# RAG evaluation

How to batch-test **layer-rag-query** end-to-end (retrieve → rerank → answer) and score retrieval quality. This service exposes **`POST /v1/rag/query`**; evaluation harnesses live in sibling repos and call that API (or the CLI) with fixed parameters.

Field-level contract: [schema.md](schema.md). Runnable smoke `curl` examples: [smoke-test.md](smoke-test.md).

---

## What to evaluate

| Layer | Question | Typical signal |
|-------|----------|----------------|
| **Retrieval** | Did the right chunk rank in the fused / reranked pool? | `retrieval_hits`, `rag.retrieval.*`, Recall@k |
| **Generation** | Is the answer grounded and correct vs gold? | `answer`, `citations`, LLM-as-judge metrics |
| **Latency** | Is the pipeline within SLO? | `latency_ms` (`embed`, `retrieve_rerank`, `chat`, `total`) |

---

## Prerequisites

1. **Qdrant** collection ingested for the target `ENV` (see [layer-rag-ingest-v1](https://github.com/taixingbi/layer-rag-ingest-v1)).
2. **Embedding** and **rerank** APIs reachable from `.env`.
3. **Chat completions** API reachable (`INFERENCE_URL` / gateway).
4. **layer-rag-query** HTTP server running, e.g.:

   ```bash
   fastmcp run app/main.py:mcp --transport http --host 0.0.0.0 --port 8000
   ```

5. Shell loads `.env` when using the CLI:

   ```bash
   set -a && source .env && set +a
   ```

---

## Recommended HTTP settings for batch eval

Use **non-stream JSON** and **single-pass** answering so each gold row maps to one deterministic chat call (no `retrieval_widen` retries).

| Body field | Eval value | Why |
|------------|------------|-----|
| `stream` | `false` | Single JSON object per row; no SSE parsing. |
| `expand_on_not_found` | `false` | One chat at initial `k`; no context-slice widen on `NOT_FOUND`. |
| `include_follow_up_questions` | `false` | Optional; skips extra chat + rerank for faster runs. |
| `include_retrieval_hits` | `true` | When scoring retrieval rank / Recall@k (see below). |
| `k` / `k_max` | e.g. `5` / `40` | Match production or experiment matrix. |

**Correlation ids:** send `X-Request-Id` and `X-Session-Id` as **headers only**. Putting `request_id`, `session_id`, or `trace_id` in the JSON body returns **400** ([smoke-test.md](smoke-test.md#correlation-headers)).

**Access control:** for HR / admin gold rows, send `X-User-Roles` (and groups/teams) as headers — not in the body. Anonymous eval uses default `anyuser` ([access-control.md](access-control.md)).

Example:

```bash
RAG_URL="${RAG_URL:-http://127.0.0.1:8000}"

curl -sS -X POST "$RAG_URL/v1/rag/query" \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: eva-req-001" \
  -H "X-Session-Id: eva-ses-001" \
  -d '{
    "question": "What is Taixing visa status?",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 40,
    "stream": false,
    "expand_on_not_found": false,
    "include_follow_up_questions": false,
    "include_retrieval_hits": true
  }'
```

Response fields used by harnesses: `answer`, `citations`, `latency_ms`, `usage`, `rag`, and optionally `retrieval_hits`.

---

## Evaluation harnesses (external repos)

### 1. End-to-end Q&A + LLM judge — [layer-rag-evaluation-v1](https://github.com/taixingbi/layer-rag-evaluation-v1)

Batch gold JSON → fill `inference-output` via `/v1/rag/query` → attach per-row **metrics** (LLM-as-judge or heuristics).

**One-shot** (RAG fill + metrics):

```bash
cd layer-rag-evaluation-v1
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 main.py \
  -i dataset/dataset-gold-test-1.0.0.json \
  -o result/dataset-gold-test-1.0.0.json \
  --base-url http://127.0.0.1:8000 \
  --rag-max-concurrency 1 \
  --judge-max-concurrency 8
```

**Standalone steps:**

```bash
python3 rag_query.py -i dataset/dataset-gold-test-1.0.0.json -o result/dataset-gold-test-1.0.0.json
python3 metric.py -i result/dataset-gold-test-1.0.0.json -o result/dataset-gold-test-1.0.0.json
```

**Gold row shape** (JSON array):

| Field | Role |
|-------|------|
| `input` | Question |
| `output` | Reference answer |
| `inference-output` | Filled by RAG (`answer` from API) |

After `metric.py`, each row includes a `metrics` object when scoring succeeds.

**Judge metrics** (when `LLM_JUDGE_URL` or `INFERENCE_URL` is set in `.env`):

| Key | Meaning |
|-----|---------|
| `faithfulness_grounding` | Grounding vs gold and citation excerpts |
| `answer_correctness` | Alignment with gold answer |
| `context_relevance` | How well excerpts support the answer |
| `answer_relevance` | Whether the answer addresses the question |
| `hallucination_rate_proxy` | Contradiction / fabrication vs gold (lower is better) |

Use `--heuristic-only` to skip the LLM judge. See that repo’s README for concurrency, retries, and `.env` variables.

> **Note:** Eval HTTP clients must send `stream: false` and correlation ids via **headers**. If a client still puts `request_id` / `session_id` in the JSON body, the server returns **400** — update the client to match the contract above.

---

### 2. Retrieval + must-contain — [layer-rag-evaluation-v1](../../layer-rag-evaluation-v1) `run_eval.py`

Gold **JSONL** generated from ingested `points_*.json` ([gold-dataset.md](../../layer-rag-evaluation-v1/docs/gold-dataset.md)). Scores substring `must_contain`, citation `source` match, and optional **Recall@k** from `retrieval_hits`.

```bash
cd layer-rag-evaluation-v1
python3 rag_gold_eval/run_eval.py \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --rag-base-url http://127.0.0.1:8000 \
  --collection-base taixing_knowledge \
  --recall-at-k 5,10,40
```

With `--skip-retrieval-hits`, retrieval rank metrics are omitted (answer-only scoring).

---

## CLI spot checks (no harness)

Same pipeline as HTTP, useful for debugging one question:

```bash
python -m app.rag "What is Taixing visa status?" -c taixing_knowledge -k 5 \
  --single-pass \
  --no-follow-ups \
  --retrieval-hits
```

| Flag | Effect |
|------|--------|
| `--single-pass` | `expand_on_not_found=False` — one chat at `k` |
| `--no-reranker` | Skip cross-encoder rerank |
| `--no-follow-ups` | Skip follow-up question generation |
| `--retrieval-hits` | Print `retrieval_hits` in stdout JSON |

Set `RAG_REQUEST_ID` / `RAG_SESSION_ID` in the environment for embedding trace correlation.

---

## `retrieval_hits` (retrieval-only debug)

Enable with any of: `include_retrieval_hits`, `debug`, `trace_retrieval`, `return_retrieval_hits`.

Each hit is a slim row (no passage text):

| Field | Description |
|-------|-------------|
| `stage` | `"retrieve"` (RRF order) or `"rerank"` (cross-encoder order) |
| `rank` | 1-based rank within that stage |
| `chunk_id` | Qdrant point id |
| `source` | Document / file label |
| `score` | RRF or rerank score (not comparable across stages) |

Full schema: [schema.md](schema.md#retrieval-hit).

The `rag` block always includes retrieval funnel counts (`retrieved_chunks`, `reranked_chunks`, `context_chunks`, `top_score`, `confidence`) without enabling `retrieval_hits`.

---

## Interpreting responses

### `rag.retrieval`

| Field | Use in eval |
|-------|-------------|
| `confidence` | `high` / `medium` / `low` — coarse retrieval quality |
| `top_score` | Best rerank (or RRF) score in context |
| `context_tokens` | Prompt size pressure |
| `embed_model` / `reranker_model` | Version pinning in reports |

### `latency_ms`

| Phase | Covers |
|-------|--------|
| `embed` | Query embedding |
| `retrieve_rerank` | Qdrant hybrid + optional rerank API |
| `chat` | Main answer completion(s); includes widen attempts when `expand_on_not_found=true` |
| `follow_up_chat` | Follow-up generator (if enabled) |
| `total` | Wall clock for the request |

For comparable batch timings, use `expand_on_not_found: false` and `include_follow_up_questions: false`.

### Answers

- Exact `NOT_FOUND` — model abstained; may trigger widen when `expand_on_not_found=true`.
- Inline `[n]` markers — citation indices; strip before string-similarity metrics.
- `citations[].excerpt` — passages the judge uses for grounding scores.

---

## Suggested regression workflow

1. **Ingest** target collection (`layer-rag-ingest-v1` `./scripts/data1.sh dev|prod`).
2. **Smoke** one question via [smoke-test.md](smoke-test.md).
3. **Retrieval eval** — `run_eval.py` on `easy_single_hop.jsonl`; track Recall@k and `must_contain_pass` rate.
4. **End-to-end eval** — `layer-rag-evaluation-v1` `main.py`; track judge means and latency.
5. **Compare** result JSON across prompt / model / `k_max` changes; keep `collection_base`, `ENV`, and embed model fixed.

Pin in eval reports: git SHA of **layer-rag-query**, **layer-rag-ingest** (KB version), `INFERENCE_MODEL`, `EMBEDDING_MODEL`, `k`, `k_max`, and access headers used.

---

## Related docs

- [schema.md](schema.md) — request/response fields
- [streaming.md](streaming.md) — SSE (not used for batch eval)
- [access-control.md](access-control.md) — `X-User-*` headers for role-gated gold
- [follow-up-questions.md](follow-up-questions.md) — disable for faster eval runs
- [layer-rag-evaluation-v1 README](https://github.com/taixingbi/layer-rag-evaluation-v1) — judge metrics and `main.py` flags
- [layer-rag-evaluation-v1 gold-dataset.md](../../layer-rag-evaluation-v1/docs/gold-dataset.md) — JSONL gold generation and `run_eval.py`
- [layer-rag-evaluation-v1 eval.md](../../layer-rag-evaluation-v1/docs/eval.md) — end-to-end eval workflow



For retrieval, I use recall@k, precision@k, MRR, and sometimes nDCG. 
For generation, I evaluate correctness, faithfulness, relevance, and citation accuracy. In production, I also track latency, token usage, user feedback, fallback rate, and no-answer rate.