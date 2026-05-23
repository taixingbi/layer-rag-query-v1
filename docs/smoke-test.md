# Smoke tests (curl)

Reference list of every HTTP endpoint exposed by **layer-rag-query** plus the upstream services it calls. Use these to verify each dependency end-to-end without writing Python.

Field-level request/response contract: [schema.md](schema.md).

All snippets assume a `.env` at the repo root (see [README.md](../README.md#configuration)). Load it once per shell:

```bash
set -a && source .env && set +a
```

When the FastMCP HTTP server is running locally, the base URL is `http://127.0.0.1:8000` (start it with `fastmcp run app/main.py:mcp --transport http --host 0.0.0.0 --port 8000`).

## Liveness / readiness / version / metrics

No `request_id` / `session_id` required for these probes.

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected: `200` with `status`, `app_name`, and `app_version` (from `APP_VERSION` in the image or package metadata).

```bash
curl -sS http://127.0.0.1:8000/version
```

Expected: `200 {"app_name":"layer-rag-query","app_version":"..."}`.

```bash
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" http://127.0.0.1:8000/ready
```

Expected when Qdrant is reachable: `200` with `status`, `app_name`, `app_version`. When Qdrant is unreachable / mis-configured: `503` with `status`, `detail`, `app_name`, `app_version`.

```bash
curl -sS http://127.0.0.1:8000/metrics | head
```

Expected: Prometheus text exposition (`rag_query_http_requests_total`, `rag_query_duration_seconds`, etc.).

## Correlation headers

`POST /v1/rag/query` and MCP `tools/call` read correlation IDs **only from HTTP headers** (not the JSON body). Putting `request_id`, `session_id`, or `trace_id` in the RAG JSON body returns **400**.

| Header | Required | Notes |
|--------|----------|-------|
| `X-Request-Id` | no | If missing or blank, the server generates a UUID for this call. Forwarded to the embedding API; appears as `request_id` in stderr JSON logs. |
| `X-Session-Id` | no | If missing or blank, the server generates a UUID for this call. Forwarded to the embedding API; appears as `session_id` in stderr JSON logs. |
| `X-Trace-Id` | no | When set, forwarded to embedding API as `X-Trace-Id` and appears as `trace_id` in stderr JSON logs (else `"-"`). Not auto-generated. |

## Access-control headers

Same header-only rule as correlation. The four dimensions drive a Qdrant payload filter on `payload.access.{roles,groups,teams}`. See [access-control.md](access-control.md) for the full semantics table (admin bypass, deny-by-default for untagged chunks, `anyuser` public default).

| Header | Required | Notes |
|--------|----------|-------|
| `X-User-Id` | no | Default `"-"`. Echoed on `200` (JSON and SSE); appears as `user_id` in stderr JSON logs. |
| `X-User-Roles` | no | Comma-separated. Default `["anyuser"]` when missing or empty (anonymous public access). `admin` (case-insensitive) bypasses the filter. |
| `X-User-Groups` | no | Comma-separated. Default `[]`. |
| `X-User-Teams` | no | Comma-separated. Default `[]`. |

Sending **no** access headers is the same as anonymous: the request asks for chunks whose `access.roles` contains `"anyuser"`. Untagged chunks (no `payload.access`) are returned only to admins.

## RAG query (`POST /v1/rag/query`)

Default JSON response: `answer`, `citations`, `follow_up_questions`, `latency_ms`, `usage`, plus `request_id`, `session_id`, `trace_id`, and `conversation_id`. On **200**, ids are echoed as `X-Request-Id`, `X-Session-Id`, `X-Trace-Id` (when sent), and `X-Conversation-Id`.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: req-abc123" \
  -H "X-Session-Id: ses-xyz789" \
  -H "X-Trace-Id: trace-001" \
  -H "X-User-Id: taixing" \
  -H "X-User-Roles: hr" \
  -H "X-User-Groups: engineering" \
  -H "X-User-Teams: rag-platform" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50
  }'
```

**Tune follow-ups:** add `"follow_up_candidates": 10` and `"follow_up_final": 5` to the JSON (`follow_up_final` must be ≤ `follow_up_candidates`, else **422**).

**Streaming (SSE):** add `-H "Accept: text/event-stream"` (or `"stream": true` in the body instead). Use `curl -N`. See [streaming.md](streaming.md). Expect `meta`, `latency` phases, `answer_delta` frames, `citations`, `follow_up_questions`, `done`.

```bash
curl -N -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-Request-Id: req-abc123" \
  -H "X-Session-Id: ses-xyz789" \
  -H "X-Trace-Id: trace-001" \
  -H "X-User-Id: taixing" \
  -H "X-User-Roles: hr" \
  -H "X-User-Groups: engineering" \
  -H "X-User-Teams: rag-platform" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50
  }'
```

**Error cases:**

```bash
# request_id in body -> HTTP 400
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" \
  -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"q","collection_base":"taixing_knowledge","request_id":"x","session_id":"y"}'

# user_roles in body -> HTTP 400
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" \
  -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"q","collection_base":"taixing_knowledge","user_roles":["admin"]}'
```

## MCP over HTTP (`/v1/mcp`)

Prefer **`POST /v1/rag/query`** for plain JSON. MCP uses the same RAG logic via `tools/call` → `rag_query` with `"stream": true` or `false` in `arguments` (same meaning as the HTTP body flag).

MCP responses are **SSE frames** (`event: message` + `data: {...}`). Parse `result.content[0].text` (JSON string when not streaming events; `{"events": [...]}` when streaming).

```bash
MCP_URL=http://127.0.0.1:8000/v1/mcp

curl -sS -X POST "${MCP_URL}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Request-Id: req-abc123" \
  -H "X-Session-Id: ses-xyz789" \
  -H "X-Trace-Id: trace-001" \
  -H "X-User-Id: taixing" \
  -H "X-User-Roles: hr" \
  -H "X-User-Groups: engineering" \
  -H "X-User-Teams: rag-platform" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "rag_query",
      "arguments": {
        "question": "what is taixing visa status in us?",
        "collection_base": "taixing_knowledge",
        "stream": true,
        "k": 5,
        "k_max": 50
      }
    }
  }'
```

Optional: `sed -n 's/^data: //p' | jq -r '.result.content[0].text' | jq .` (non-stream) or `| jq -r '.events[].type'` (stream).

## Embedding API (upstream)

`POST /v1/embeddings` on `EMBEDDING_URL`. `X-Trace-Id` forwarded only when set.

```bash
curl -sS -X POST "${EMBEDDING_URL}/v1/embeddings" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBEDDING_MODEL}\", \"input\": \"hello world\"}"
```

## Inference / chat (upstream)

`POST /v1/chat/completions` on `INFERENCE_URL`.

```bash
curl -sS -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${INFERENCE_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"where is jersey city\"}], \"max_tokens\": 50}"
```

## Qdrant (upstream)

```bash
curl -sS "${QDRANT_URL}/collections" \
  -H "api-key: ${QDRANT_API_KEY}"

curl -sS "${QDRANT_URL}/collections/taixing_knowledge_${ENV}" \
  -H "api-key: ${QDRANT_API_KEY}"
```

## Quick cheat sheet

| Purpose | Method | URL |
|---------|--------|-----|
| Liveness | `GET` | `http://127.0.0.1:8000/health` |
| Readiness | `GET` | `http://127.0.0.1:8000/ready` |
| Version | `GET` | `http://127.0.0.1:8000/version` |
| Metrics | `GET` | `http://127.0.0.1:8000/metrics` |
| RAG (JSON) | `POST` | `http://127.0.0.1:8000/v1/rag/query` |
| RAG (SSE) | `POST` | `http://127.0.0.1:8000/v1/rag/query` + `Accept: text/event-stream` |
| MCP `rag_query` | `POST` | `http://127.0.0.1:8000/v1/mcp` |
| Embedding | `POST` | `${EMBEDDING_URL}/v1/embeddings` |
| Chat | `POST` | `${INFERENCE_URL}/v1/chat/completions` |
| Qdrant | `GET` | `${QDRANT_URL}/collections` |
