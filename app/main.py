"""
FastMCP server: hybrid chunk retrieval + embeddings (stdio for Cursor / Claude).

Requires `.env` at repo root (same as `import app`). Install: ``pip install -e ".[mcp]"``

Run: ``python -m app.main`` or ``fastmcp run app/main.py:mcp``
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import httpx
import fastmcp
from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import CurrentContext
from pydantic import BaseModel, Field, ValidationError, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from app.rag.access import RagUser
from app.core.asyncio_util import run_async
from app.core.metrics import (
    metrics_content_type,
    metrics_payload,
    observe_http,
    observe_rag_query,
)
from app.core.version import APP_NAME, get_app_version
from app.http._correlation import correlation_from_request
from app.http.embed import embed_text as _embed_text_async
from app.http.inference import resolve_conversation_id
from app.http.usage import UsageTokens
from app.core.logging_config import logger
from app.qdrant.client import create_async_client, resolve_connection_params
from app.rag.mcp_tools import (
    rag_query_non_stream,
    rag_query_stream_events_mcp,
)
from app.rag.rag_answer import complete_rag_answer_stream
from app.core.request_context import bind_http_context, bind_request_context
from app.rag.mcp_http import resolve_mcp_rag_call
from app.rag.retrieval import query_chunks as _query_chunks_async

MCP_HTTP_PATH = "/v1/mcp"

_FORBIDDEN_RAG_BODY_KEYS = frozenset({
    "request_id",
    "session_id",
    "trace_id",
    "user_id",
    "user_roles",
    "user_groups",
    "user_teams",
})


def _observe_http_request(request: Request, response: Response, started: float) -> None:
    observe_http(
        request.method,
        request.url.path,
        int(response.status_code),
        time.perf_counter() - started,
    )


def _wants_sse(request: Request) -> bool:
    """``True`` when caller asked for SSE via an ``Accept: text/event-stream`` header.

    The other supported trigger is ``"stream": true`` in the JSON body, checked at the
    route handler. Query-param triggers (``?stream=1`` etc.) are intentionally not
    supported — keep streaming opt-in via header or body so URLs stay clean."""
    accept = (request.headers.get("accept") or "").lower()
    return "text/event-stream" in accept


def _sse_event(name: str, payload: dict[str, Any]) -> bytes:
    """Encode a single SSE frame as ``event: <name>\\ndata: <json>\\n\\n`` (UTF-8)."""
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {name}\ndata: {body}\n\n".encode("utf-8")


class AnswerFromInferenceBody(BaseModel):
    question: str
    collection_base: str
    k: int = Field(default=5, ge=1)
    k_max: int = Field(default=50, ge=1)
    max_tokens: int | None = None
    expand_on_not_found: bool = True
    rerank_top_n: int | None = Field(default=None, ge=1)
    rerank_return_top_k: int | None = Field(default=None, ge=1)
    retrieve_fallback_n: int | None = Field(default=None, ge=0)
    final_context_top_k: int | None = Field(default=None, ge=1)
    use_reranker: bool = True
    include_follow_up_questions: bool = True
    follow_up_candidates: int = Field(default=8, ge=3, le=12)
    follow_up_final: int = Field(default=3, ge=1, le=8)
    include_retrieval_hits: bool = False
    debug: bool = False
    trace_retrieval: bool = False
    return_retrieval_hits: bool = False
    stream: bool = True
    conversation_id: str | None = None

    @model_validator(mode="after")
    def _follow_up_final_lte_candidates(self) -> AnswerFromInferenceBody:
        if self.follow_up_final > self.follow_up_candidates:
            raise ValueError("follow_up_final must be <= follow_up_candidates")
        return self

    def wants_retrieval_hits(self) -> bool:
        """Support historical debug aliases for returning retrieval_hits."""
        return bool(
            self.include_retrieval_hits
            or self.debug
            or self.trace_retrieval
            or self.return_retrieval_hits
        )


def _answer_payload(
    *,
    answer: str,
    citations: list[dict],
    follow_up_questions: list[str],
    latency_ms: dict[str, int],
    retrieval_hits: list[dict],
    include_retrieval_hits: bool,
    request_id: str,
    session_id: str,
    trace_id: str | None,
    conversation_id: str,
    usage: dict[str, UsageTokens],
) -> dict[str, Any]:
    """Build stable HTTP/MCP response payload (conditionally including retrieval_hits)."""
    out: dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "follow_up_questions": follow_up_questions,
        "latency_ms": latency_ms,
        "usage": usage,
        "request_id": request_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
    }
    if include_retrieval_hits:
        out["retrieval_hits"] = retrieval_hits
    return out


async def answer_from_inference_payload_async(
    body: AnswerFromInferenceBody,
    *,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    user: RagUser | None = None,
    conversation_id: str,
) -> dict[str, Any]:
    """Run RAG + chat (async). Raise ``ValueError`` or ``httpx.HTTPStatusError`` on failure."""
    wants_hits = body.wants_retrieval_hits()
    return await rag_query_non_stream(
        question=body.question,
        collection_base=body.collection_base,
        request_id=request_id,
        session_id=session_id,
        k=body.k,
        k_max=body.k_max,
        max_tokens=body.max_tokens,
        expand_on_not_found=body.expand_on_not_found,
        rerank_top_n=body.rerank_top_n,
        rerank_return_top_k=body.rerank_return_top_k,
        retrieve_fallback_n=body.retrieve_fallback_n,
        final_context_top_k=body.final_context_top_k,
        use_reranker=body.use_reranker,
        include_follow_up_questions=body.include_follow_up_questions,
        follow_up_candidates=body.follow_up_candidates,
        follow_up_final=body.follow_up_final,
        include_retrieval_hits=body.include_retrieval_hits,
        debug=body.debug,
        trace_retrieval=body.trace_retrieval,
        return_retrieval_hits=body.return_retrieval_hits,
        trace_id=trace_id,
        user=user,
        conversation_id=conversation_id,
        build_payload=lambda **kw: _answer_payload(
            **kw,
            include_retrieval_hits=wants_hits,
        ),
    )


fastmcp.settings.streamable_http_path = MCP_HTTP_PATH
# Stateless: each POST /v1/mcp is self-contained (no initialize / mcp-session-id).
fastmcp.settings.stateless_http = True

mcp = FastMCP(
    "layer-rag-query",
    instructions="RAG tools: Qdrant hybrid search (dense + BM25 + RRF), embeddings, and optional "
    "full answers via INFERENCE_URL /v1/chat/completions (set in .env). "
    "Pass collection base; ENV suffix comes from .env. request_id and session_id are required for retrieval embedding calls. "
    "Use rag_query with stream=true for live RAG events by default; set stream=false for a single JSON answer (same shape as POST /v1/rag/query). "
    "(MCP progress notifications + final {\"events\": [...]}; upstream answer tokens stream as they are generated). "
    "On HTTP transport, pass X-Request-Id, X-Session-Id, X-Trace-Id, and X-User-* headers on each POST /v1/mcp call.",
)

async def _mcp_rag_query_impl(
    mcp_ctx: Context,
    *,
    question: str,
    collection_base: str,
    request_id: str = "",
    session_id: str = "",
    stream: bool = True,
    k: int = 5,
    k_max: int = 50,
    max_tokens: int | None = None,
    expand_on_not_found: bool = True,
    rerank_top_n: int | None = None,
    rerank_return_top_k: int | None = None,
    retrieve_fallback_n: int | None = None,
    final_context_top_k: int | None = None,
    use_reranker: bool = True,
    include_follow_up_questions: bool = True,
    follow_up_candidates: int = 8,
    follow_up_final: int = 3,
    include_retrieval_hits: bool = False,
    debug: bool = False,
    trace_retrieval: bool = False,
    return_retrieval_hits: bool = False,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    call_ctx = resolve_mcp_rag_call(
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    wants_hits = include_retrieval_hits or debug or trace_retrieval or return_retrieval_hits
    rag_kwargs = dict(
        question=question,
        collection_base=collection_base,
        request_id=call_ctx.request_id,
        session_id=call_ctx.session_id,
        k=k,
        k_max=k_max,
        max_tokens=max_tokens,
        expand_on_not_found=expand_on_not_found,
        rerank_top_n=rerank_top_n,
        rerank_return_top_k=rerank_return_top_k,
        retrieve_fallback_n=retrieve_fallback_n,
        final_context_top_k=final_context_top_k,
        use_reranker=use_reranker,
        include_follow_up_questions=include_follow_up_questions,
        follow_up_candidates=follow_up_candidates,
        follow_up_final=follow_up_final,
        include_retrieval_hits=include_retrieval_hits,
        debug=debug,
        trace_retrieval=trace_retrieval,
        return_retrieval_hits=return_retrieval_hits,
        trace_id=call_ctx.trace_id,
        user=call_ctx.user,
        conversation_id=call_ctx.conversation_id,
    )
    with bind_request_context(
        call_ctx.request_id,
        call_ctx.session_id,
        trace_id=call_ctx.trace_id,
        user_id=call_ctx.user.id,
        conversation_id=call_ctx.conversation_id,
    ):
        if stream:
            return await rag_query_stream_events_mcp(mcp_ctx, **rag_kwargs)
        return await rag_query_non_stream(
            **rag_kwargs,
            build_payload=lambda **kw: _answer_payload(
                **kw,
                include_retrieval_hits=wants_hits,
            ),
        )


@mcp.tool
def retrieve_chunks(
    query: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    k: int = 5,
) -> list[dict]:
    """Hybrid retrieval from Qdrant. collection_base is suffixed with ENV from .env (e.g. taixing_knowledge + dev → taixing_knowledge_dev)."""
    return run_async(
        _query_chunks_async(
            query,
            collection_base,
            k=k,
            request_id=request_id,
            session_id=session_id,
        )
    )


@mcp.tool
def embed_text(
    text: str,
    request_id: str,
    session_id: str,
) -> list[float]:
    """Embed a single string via the configured /v1/embeddings API. Returns the embedding vector."""
    return run_async(_embed_text_async(text, request_id=request_id, session_id=session_id))


@mcp.tool
async def rag_query(
    question: str,
    collection_base: str,
    request_id: str = "",
    session_id: str = "",
    stream: bool = True,
    k: int = 5,
    k_max: int = 50,
    max_tokens: int | None = None,
    expand_on_not_found: bool = True,
    rerank_top_n: int | None = None,
    rerank_return_top_k: int | None = None,
    retrieve_fallback_n: int | None = None,
    final_context_top_k: int | None = None,
    use_reranker: bool = True,
    include_follow_up_questions: bool = True,
    follow_up_candidates: int = 8,
    follow_up_final: int = 3,
    include_retrieval_hits: bool = False,
    debug: bool = False,
    trace_retrieval: bool = False,
    return_retrieval_hits: bool = False,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    mcp_ctx: Context = CurrentContext(),
) -> dict[str, Any]:
    """RAG answer via MCP. Same parameters as ``POST /v1/rag/query``.

    Default is ``stream=true`` for live RAG events over MCP progress notifications (see ``docs/streaming.md``),
    plus a final ``{"streamed": true, "events": [...]}`` tool result.
    Set ``stream=false`` for one JSON object (``answer``, ``citations``, …).

    On HTTP transport, set correlation and access headers on the MCP request (they override
    ``request_id`` / ``session_id`` / ``trace_id`` tool arguments). See ``docs/smoke-test.md``.
    """
    return await _mcp_rag_query_impl(
        mcp_ctx,
        question=question,
        collection_base=collection_base,
        request_id=request_id,
        session_id=session_id,
        stream=stream,
        k=k,
        k_max=k_max,
        max_tokens=max_tokens,
        expand_on_not_found=expand_on_not_found,
        rerank_top_n=rerank_top_n,
        rerank_return_top_k=rerank_return_top_k,
        retrieve_fallback_n=retrieve_fallback_n,
        final_context_top_k=final_context_top_k,
        use_reranker=use_reranker,
        include_follow_up_questions=include_follow_up_questions,
        follow_up_candidates=follow_up_candidates,
        follow_up_final=follow_up_final,
        include_retrieval_hits=include_retrieval_hits,
        debug=debug,
        trace_retrieval=trace_retrieval,
        return_retrieval_hits=return_retrieval_hits,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )


@mcp.tool
async def rag_query_stream(
    question: str,
    collection_base: str,
    request_id: str = "",
    session_id: str = "",
    k: int = 5,
    k_max: int = 50,
    max_tokens: int | None = None,
    expand_on_not_found: bool = True,
    rerank_top_n: int | None = None,
    rerank_return_top_k: int | None = None,
    retrieve_fallback_n: int | None = None,
    final_context_top_k: int | None = None,
    use_reranker: bool = True,
    include_follow_up_questions: bool = True,
    follow_up_candidates: int = 8,
    follow_up_final: int = 3,
    include_retrieval_hits: bool = False,
    debug: bool = False,
    trace_retrieval: bool = False,
    return_retrieval_hits: bool = False,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    mcp_ctx: Context = CurrentContext(),
) -> dict[str, Any]:
    """Alias for :func:`rag_query` with ``stream=true`` (kept for backward compatibility)."""
    return await _mcp_rag_query_impl(
        mcp_ctx,
        question=question,
        collection_base=collection_base,
        request_id=request_id,
        session_id=session_id,
        stream=True,
        k=k,
        k_max=k_max,
        max_tokens=max_tokens,
        expand_on_not_found=expand_on_not_found,
        rerank_top_n=rerank_top_n,
        rerank_return_top_k=rerank_return_top_k,
        retrieve_fallback_n=retrieve_fallback_n,
        final_context_top_k=final_context_top_k,
        use_reranker=use_reranker,
        include_follow_up_questions=include_follow_up_questions,
        follow_up_candidates=follow_up_candidates,
        follow_up_final=follow_up_final,
        include_retrieval_hits=include_retrieval_hits,
        debug=debug,
        trace_retrieval=trace_retrieval,
        return_retrieval_hits=return_retrieval_hits,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )


@mcp.tool
async def answer_from_inference(
    question: str,
    collection_base: str,
    request_id: str = "",
    session_id: str = "",
    k: int = 5,
    k_max: int = 50,
    max_tokens: int | None = None,
    expand_on_not_found: bool = True,
    rerank_top_n: int | None = None,
    rerank_return_top_k: int | None = None,
    retrieve_fallback_n: int | None = None,
    final_context_top_k: int | None = None,
    use_reranker: bool = True,
    include_follow_up_questions: bool = True,
    follow_up_candidates: int = 8,
    follow_up_final: int = 3,
    include_retrieval_hits: bool = False,
    debug: bool = False,
    trace_retrieval: bool = False,
    return_retrieval_hits: bool = False,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    mcp_ctx: Context = CurrentContext(),
) -> dict[str, Any]:
    """Alias for :func:`rag_query` with ``stream=false`` (kept for backward compatibility)."""
    return await _mcp_rag_query_impl(
        mcp_ctx,
        question=question,
        collection_base=collection_base,
        request_id=request_id,
        session_id=session_id,
        stream=False,
        k=k,
        k_max=k_max,
        max_tokens=max_tokens,
        expand_on_not_found=expand_on_not_found,
        rerank_top_n=rerank_top_n,
        rerank_return_top_k=rerank_return_top_k,
        retrieve_fallback_n=retrieve_fallback_n,
        final_context_top_k=final_context_top_k,
        use_reranker=use_reranker,
        include_follow_up_questions=include_follow_up_questions,
        follow_up_candidates=follow_up_candidates,
        follow_up_final=follow_up_final,
        include_retrieval_hits=include_retrieval_hits,
        debug=debug,
        trace_retrieval=trace_retrieval,
        return_retrieval_hits=return_retrieval_hits,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )


@mcp.custom_route("/v1/rag/query", methods=["POST"])
async def answer_from_inference_http(request: Request) -> Response:
    """JSON body for ``curl`` when using FastMCP ``--transport http``.

    Correlation: ``X-Request-Id``, ``X-Session-Id``, ``X-Trace-Id`` (optional). If request or
    session id headers are missing or blank, new UUIDs are generated for this call only.
    Do not send ``request_id``, ``session_id``, or ``trace_id`` in the JSON body (400).

    Access control: ``X-User-Id`` / ``X-User-Roles`` / ``X-User-Groups`` / ``X-User-Teams``
    (all optional). Roles default to ``["anyuser"]`` when absent so chunks tagged
    ``access.roles=["anyuser"]`` are the public set. ``admin`` role bypasses filtering.
    These four fields must NOT appear in the JSON body (400). See ``docs/access-control.md``.

    Default behavior is streaming when the JSON body omits ``stream`` (``"stream": true`` by default).
    Set ``"stream": false`` for single-shot ``application/json``. ``Accept: text/event-stream``
    also forces streaming. See ``docs/streaming.md`` for the event
    sequence and error semantics.

    Optional JSON field ``conversation_id`` threads the chat turn through the inference
    gateway (same contract as ``layer-gateway-inference-v1``): forwarded on every
    ``/v1/chat/completions`` body; omitted or blank values get a server-generated
    ``conv_<hex>`` id, echoed as ``X-Conversation-Id`` and in the JSON response
  (along with ``request_id``, ``session_id``, and ``trace_id`` in the body).
    """
    http_t0 = time.perf_counter()
    method = request.method
    path = request.url.path

    try:
        data = await request.json()
    except Exception:
        response = JSONResponse({"detail": "Invalid JSON"}, status_code=400)
        _observe_http_request(request, response, http_t0)
        return response
    if not isinstance(data, dict):
        response = JSONResponse({"detail": "JSON body must be an object"}, status_code=400)
        _observe_http_request(request, response, http_t0)
        return response
    if _FORBIDDEN_RAG_BODY_KEYS & data.keys():
        response = JSONResponse(
            {
                "detail": (
                    "request_id, session_id, and trace_id must not appear in the JSON body; "
                    "use X-Request-Id, X-Session-Id, and X-Trace-Id headers instead."
                )
            },
            status_code=400,
        )
        _observe_http_request(request, response, http_t0)
        return response
    request_id, session_id, trace_id = correlation_from_request(request)
    if not request_id:
        request_id = str(uuid.uuid4())
    if not session_id:
        session_id = str(uuid.uuid4())
    user = RagUser.from_headers(request)
    try:
        body = AnswerFromInferenceBody.model_validate(data)
    except ValidationError as e:
        response = JSONResponse({"detail": e.errors()}, status_code=422)
        _observe_http_request(request, response, http_t0)
        return response

    conversation_id = resolve_conversation_id(body.conversation_id)

    if _wants_sse(request) or body.stream:
        stream_latency_ms: dict[str, int] = {}

        async def sse_iter():
            """Yield SSE frames from ``complete_rag_answer_stream``. Once headers flushed
            we're locked at HTTP 200, so any error is surfaced in-band as ``error`` +
            ``done`` (per ``docs/streaming.md``)."""
            with bind_http_context(method, path, status="200"):
                try:
                    async for ev in complete_rag_answer_stream(
                        body.question,
                        body.collection_base,
                        request_id,
                        session_id,
                        k=body.k,
                        k_max=body.k_max,
                        max_tokens=body.max_tokens,
                        expand_on_not_found=body.expand_on_not_found,
                        rerank_top_n=body.rerank_top_n,
                        rerank_return_top_k=body.rerank_return_top_k,
                        retrieve_fallback_n=body.retrieve_fallback_n,
                        final_context_top_k=body.final_context_top_k,
                        use_reranker=body.use_reranker,
                        include_follow_up_questions=body.include_follow_up_questions,
                        follow_up_candidates=body.follow_up_candidates,
                        follow_up_final=body.follow_up_final,
                        include_retrieval_hits=body.wants_retrieval_hits(),
                        trace_id=trace_id,
                        user=user,
                        conversation_id=conversation_id,
                    ):
                        ev_name = ev.pop("type")
                        if ev_name == "latency":
                            phase = ev.get("phase")
                            ms = ev.get("ms")
                            if isinstance(phase, str) and isinstance(ms, int):
                                stream_latency_ms[phase] = ms
                        yield _sse_event(ev_name, ev)
                    observe_rag_query(
                        status_code=200,
                        stream=True,
                        latency_ms=stream_latency_ms or None,
                    )
                except asyncio.CancelledError:
                    # Client closed the connection (Pause / abort / tab close). Don't
                    # try to emit `error` / `done` — the socket is already gone — but
                    # do log a structured WARNING and re-raise so the upstream
                    # ``chat_complete_stream`` cancellation chain runs to completion
                    # (closes httpx stream → TCP RST to vLLM → frees GPU slot).
                    logger.warning(
                        "rag_query stream cancelled by client",
                        extra={"reason": "client_cancelled"},
                    )
                    raise
                except ValueError as e:
                    yield _sse_event("error", {"detail": str(e)})
                    yield _sse_event("done", {})
                except httpx.HTTPStatusError as e:
                    yield _sse_event(
                        "error",
                        {"detail": e.response.text or str(e)},
                    )
                    yield _sse_event("done", {})
                except Exception as e:
                    logger.exception(
                        "complete_rag_answer_stream crashed",
                        extra={"error_type": type(e).__name__, "error_message": str(e)},
                    )
                    yield _sse_event(
                        "error",
                        {"detail": f"{type(e).__name__}: {e}"},
                    )
                    yield _sse_event("done", {})

        sse_headers: dict[str, str] = {
            "X-Request-Id": request_id,
            "X-Session-Id": session_id,
            "X-User-Id": user.id,
            "X-Conversation-Id": conversation_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if trace_id:
            sse_headers["X-Trace-Id"] = trace_id
        response = StreamingResponse(
            sse_iter(),
            media_type="text/event-stream",
            headers=sse_headers,
        )
        _observe_http_request(request, response, http_t0)
        return response

    try:
        # method/path/status for stderr JSON lines (matches ASGI access log when happy path).
        with bind_http_context(method, path, status="200"):
            out = await answer_from_inference_payload_async(
                body,
                request_id=request_id,
                session_id=session_id,
                trace_id=trace_id,
                user=user,
                conversation_id=conversation_id,
            )
    except ValueError as e:
        response = JSONResponse({"detail": str(e)}, status_code=400)
        _observe_http_request(request, response, http_t0)
        return response
    except httpx.HTTPStatusError as e:
        response = JSONResponse(
            {"detail": e.response.text or str(e)},
            status_code=502,
        )
        _observe_http_request(request, response, http_t0)
        return response
    hdrs: dict[str, str] = {
        "X-Request-Id": request_id,
        "X-Session-Id": session_id,
        "X-User-Id": user.id,
        "X-Conversation-Id": conversation_id,
    }
    if trace_id:
        hdrs["X-Trace-Id"] = trace_id
    response = JSONResponse(out, headers=hdrs)
    _observe_http_request(request, response, http_t0)
    return response


@mcp.custom_route("/version", methods=["GET"], include_in_schema=False)
async def version(request: Request) -> JSONResponse:
    """Build identity for probes and dashboards."""
    http_t0 = time.perf_counter()
    with bind_http_context(request.method, request.url.path, status="200"):
        response = JSONResponse(
            {
                "app_name": APP_NAME,
                "app_version": get_app_version(),
            }
        )
    _observe_http_request(request, response, http_t0)
    return response


@mcp.custom_route("/metrics", methods=["GET"], include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Prometheus scrape endpoint."""
    http_t0 = time.perf_counter()
    response = Response(
        content=metrics_payload(),
        media_type=metrics_content_type(),
    )
    _observe_http_request(request, response, http_t0)
    return response


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(request: Request) -> JSONResponse:
    """Liveness: always 200 while the process is up."""
    http_t0 = time.perf_counter()
    with bind_http_context(request.method, request.url.path, status="200"):
        response = JSONResponse(
            {
                "status": "ok",
                "app_name": APP_NAME,
                "app_version": get_app_version(),
            }
        )
    _observe_http_request(request, response, http_t0)
    return response


@mcp.custom_route("/ready", methods=["GET"], include_in_schema=False)
async def ready(request: Request) -> JSONResponse:
    """Readiness: 200 when Qdrant responds to ``get_collections``, else 503."""
    http_t0 = time.perf_counter()
    url, api_key = resolve_connection_params()
    client = create_async_client(url, api_key)
    try:
        try:
            await client.get_collections()
        except Exception as e:
            with bind_http_context(request.method, request.url.path, status="503"):
                logger.warning(
                    "ready probe failed",
                    extra={"error_type": type(e).__name__, "error_message": str(e)},
                )
            response = JSONResponse(
                {
                    "status": "not_ready",
                    "detail": type(e).__name__,
                    "app_name": APP_NAME,
                    "app_version": get_app_version(),
                },
                status_code=503,
            )
            _observe_http_request(request, response, http_t0)
            return response
        with bind_http_context(request.method, request.url.path, status="200"):
            response = JSONResponse(
                {
                    "status": "ready",
                    "app_name": APP_NAME,
                    "app_version": get_app_version(),
                }
            )
        _observe_http_request(request, response, http_t0)
        return response
    finally:
        await client.close()


if __name__ == "__main__":
    mcp.run(path=MCP_HTTP_PATH, stateless_http=True)
