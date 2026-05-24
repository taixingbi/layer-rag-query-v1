# Follow-up questions (RAG response)

After the main RAG answer and citations are produced, the pipeline can attach **`follow_up_questions`**: short suggested questions for the user to ask next. Implementation lives in [`app/rag/follow_up.py`](../app/rag/follow_up.py) (entry point: `generate_follow_ups`), invoked from [`app/rag/rag_answer.py`](../app/rag/rag_answer.py); HTTP/MCP wiring is in [`app/main.py`](../app/main.py).

## Response shape

`POST /v1/rag/query` and the MCP tool `answer_from_inference` return JSON with:

| Field | Type | Meaning |
|-------|------|---------|
| `answer` | string | Model reply with optional inline `[n]` citations. |
| `citations` | array | Passages actually cited in `answer` (`cite_id`, `chunk_id`, `source`, `text`). |
| `follow_up_questions` | array of strings | Always present; may be `[]` if disabled, on parse failure, or when generation fails. |
| `latency_ms` | object | Always present: integer millisecond timings per phase (see below). |

### `latency_ms` keys

All values are non-negative integers (wall time from `time.perf_counter()`).

| Key | Meaning |
|-----|---------|
| `total` | End-to-end wall time for `complete_rag_answer`. |
| `embed` | Query embedding HTTP call. |
| `retrieve_rerank` | Hybrid `query_chunks` + passage rerank (combined). |
| `chat` | Sum of all main RAG `chat_complete` calls (including widen retries). |
| `follow_up_chat` | Follow-up candidate generation + rerank (combined); `0` if follow-ups disabled. |

The final `complete_rag_answer done` log line still emits internal fields (`latency_embed_ms`, `latency_retrieve_ms`, …) for structured log pipelines.

## Pipeline

1. **Main RAG** (unchanged): retrieve → optional chunk rerank → chat → citations.
2. **Context summary**: From **every chunk** in the final context slice used for the last chat turn (all retrieval hits in that slice, not only inline `[n]` citations), build a compact bullet list (`source` + truncated text, ~3.5k chars max) via `_context_summary_for_followups`.
3. **Generate candidates**: Second call to `INFERENCE_URL` / `v1/chat/completions` asks the model for **only** a JSON object of the shape `{"follow_up_questions": ["…", "…"]}`. Count bounds: `min_gen = max(3, follow_up_candidates - 3)` through `max_gen = follow_up_candidates` (defaults: 5–8 when `follow_up_candidates == 8`). The parser is tolerant: it also accepts a bare JSON array, dicts using the keys `questions` / `follow_ups`, code-fenced JSON, comma-separated arrays, and even concatenated top-level values like `["Q1"]["Q2"]["Q3"]` (a known vLLM glitch); duplicates are de-duplicated by string. **Grounding:** each summary line includes the chunk **`source`** metadata; the system prompt asks for questions answerable from any of those retrieved passages (including sources the answer did not cite inline).
4. **Rerank**: [`app/http/rerank.py`](../app/http/rerank.py) `rerank_texts` scores each candidate string against `question + "\n\n" + answer` (so the cross-encoder has the full semantic target, not just the short user question), returns all indices ordered by score, then the code keeps the first **`follow_up_final`** (default **3**).

Token budget for the generator: `min(512, max(256, max_tokens))` where `max_tokens` is the same cap used for the main RAG chat for that request.

### Success log line

When generation produces at least one ranked question, `generate_follow_ups` emits a single **INFO** line `follow_up_questions_ok cand=<N> ranked=<M> reply_chars=<C>` carrying the full input/output as top-level JSON fields (one extra line per RAG request):

| Field | Type | Meaning |
|--------|------|---------|
| `follow_up_raw_reply` | string | Full assistant `content` from the generator chat call (newlines escaped, **not truncated**). |
| `follow_up_candidates_full` | array of strings | All parsed/de-duped candidate questions before rerank. |
| `follow_up_candidates_count` | integer | `len(follow_up_candidates_full)`. |
| `follow_up_ranked` | array of strings | Final questions returned to the client (rerank top-`follow_up_final`). |
| `follow_up_ranked_count` | integer | `len(follow_up_ranked)`. |
| `latency_follow_up_chat_ms` / `latency_follow_up_rerank_ms` | integer | Same milliseconds reported in the `latency_ms` block. |

Empty / failure paths emit `follow_up_questions_empty` instead (see "Fallbacks"); the two events are mutually exclusive per request.

## HTTP request body (optional fields)

| Field | Default | Constraints |
|-------|---------|-------------|
| `include_follow_up_questions` | `true` | Set `false` to skip extra chat + rerank. |
| `follow_up_candidates` | `8` | Integer in **3–12** (LLM asked for between `max(3, N-3)` and `N` questions). |
| `follow_up_final` | `3` | Integer **1–8** and **must be <= `follow_up_candidates`**. |
| `include_retrieval_hits` | `false` | Include `retrieval_hits` in the response. Equivalent aliases: `debug`, `trace_retrieval`, `return_retrieval_hits`. |

Validation errors (e.g. `follow_up_final > follow_up_candidates`) return **422** on HTTP with Pydantic detail.

## CLI (`python -m app.rag`)

- `--no-follow-ups` — same as disabling follow-ups.
- `--follow-up-candidates N` — must satisfy `complete_rag_answer` validation (3–12).
- `--follow-up-final M` — must be ≤ candidates.

Printed JSON includes `follow_up_questions` and `latency_ms`.

## Fallbacks

- **JSON parse failure** after generation → `follow_up_questions: []` (log line `follow_up_questions_empty` with `follow_up_empty_reason` in JSON, e.g. `json_invalid_bracket_slice_failed`, `parsed_not_list:dict`, `parsed_list_no_non_empty_strings`).
- **Empty model reply** (whitespace-only assistant `content`) → `[]` (same log line with `follow_up_empty_reason=empty_model_reply` at INFO).
- **No chunks** passed into follow-up generation → `[]` (`follow_up_empty_reason=no_chunks_used`).
- **Rerank HTTP / transport error** → first `follow_up_final` candidates in **generation order** (warning `follow_up rerank failed`; response is **not** empty unless generation already failed).
- **Generation exception** (HTTP error, bad response shape, etc.) → `[]` (`follow_up_empty_reason=generation_failed` plus `error_message`).
- **Client disabled follow-ups** (`include_follow_up_questions: false`) → `[]` (`follow_up_empty_reason=follow_ups_disabled_by_request`).

## Cost and latency

Each enabled request adds **one extra chat completion** and **one rerank** call (documents = candidate question strings). Uses the same `INFERENCE_*` and `RERANK_*` settings as the rest of RAG.

## Related

- [README.md](../README.md) — curl example and body overview.
- [log-json-schema.md](log-json-schema.md) — structured logging for the same service.
