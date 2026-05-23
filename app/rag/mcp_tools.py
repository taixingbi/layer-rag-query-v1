"""MCP tool implementations for non-stream and stream RAG queries."""

from __future__ import annotations

from typing import Any

from app.core.metrics import observe_rag_query
from app.rag.access import RagUser
from app.rag.rag_answer import complete_rag_answer, complete_rag_answer_stream


def _wants_retrieval_hits(
    *,
    include_retrieval_hits: bool,
    debug: bool,
    trace_retrieval: bool,
    return_retrieval_hits: bool,
) -> bool:
    return bool(
        include_retrieval_hits or debug or trace_retrieval or return_retrieval_hits
    )


async def rag_query_non_stream(
    *,
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
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
    user: RagUser | None = None,
    conversation_id: str,
    build_payload,
) -> dict[str, Any]:
    """Run :func:`complete_rag_answer` and return the HTTP-shaped JSON body."""
    if follow_up_final > follow_up_candidates:
        raise ValueError("follow_up_final must be <= follow_up_candidates")
    if k_max < k:
        raise ValueError("k_max must be >= k")
    wants_hits = _wants_retrieval_hits(
        include_retrieval_hits=include_retrieval_hits,
        debug=debug,
        trace_retrieval=trace_retrieval,
        return_retrieval_hits=return_retrieval_hits,
    )
    answer, citations, follow_up_questions, latency_ms, retrieval_hits, usage = (
        await complete_rag_answer(
            question,
            collection_base,
            request_id,
            session_id,
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
            include_retrieval_hits=wants_hits,
            trace_id=trace_id,
            user=user,
            conversation_id=conversation_id,
        )
    )
    payload = build_payload(
        answer=answer,
        citations=citations,
        follow_up_questions=follow_up_questions,
        latency_ms=latency_ms,
        retrieval_hits=retrieval_hits,
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
        usage=usage,
    )
    observe_rag_query(status_code=200, stream=False, latency_ms=latency_ms)
    return payload


async def rag_query_stream_events(
    *,
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
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
    user: RagUser | None = None,
    conversation_id: str,
) -> dict[str, Any]:
    """Run :func:`complete_rag_answer_stream` and return all SSE-shaped events as JSON.

    Each entry is ``{"type": "<event>", ...fields}`` (same shapes as ``docs/streaming.md``).
    On pipeline failure after ``meta``, expect a terminal ``error`` then ``done`` event.
    """
    if follow_up_final > follow_up_candidates:
        raise ValueError("follow_up_final must be <= follow_up_candidates")
    if k_max < k:
        raise ValueError("k_max must be >= k")
    wants_hits = _wants_retrieval_hits(
        include_retrieval_hits=include_retrieval_hits,
        debug=debug,
        trace_retrieval=trace_retrieval,
        return_retrieval_hits=return_retrieval_hits,
    )
    events: list[dict[str, Any]] = []
    stream_latency_ms: dict[str, int] = {}
    async for ev in complete_rag_answer_stream(
        question,
        collection_base,
        request_id,
        session_id,
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
        include_retrieval_hits=wants_hits,
        trace_id=trace_id,
        user=user,
        conversation_id=conversation_id,
    ):
        ev_type = ev.pop("type")
        if ev_type == "latency":
            phase = ev.get("phase")
            ms = ev.get("ms")
            if isinstance(phase, str) and isinstance(ms, int):
                stream_latency_ms[phase] = ms
        events.append({"type": ev_type, **ev})
    observe_rag_query(
        status_code=200,
        stream=True,
        latency_ms=stream_latency_ms or None,
    )
    return {"events": events}
