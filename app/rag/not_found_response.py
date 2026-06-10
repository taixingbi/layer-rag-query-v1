"""Structured NOT_FOUND responses: deterministic search summary + LLM result and suggestions."""

from __future__ import annotations

import json
import time
from typing import Any

from app.core.config import get_follow_up_min_context_rerank_score
from app.core.logging_config import logger
from app.http.inference import chat_complete
from app.http.usage import UsageTokens
from app.rag.follow_up import (
    _context_summary_for_followups,
    _filter_follow_ups_by_context_rerank,
    _preview_for_log,
    _rerank_follow_up_strings,
)

_NOT_FOUND_GEN_MAX_TOKENS = 512


def build_search_summary(chunks_used: list[dict], *, k_used: int) -> dict[str, Any]:
    """Factual retrieval summary for chat UI (no LLM)."""
    sources: list[str] = []
    seen: set[str] = set()
    for chunk in chunks_used:
        src = (chunk.get("source") or "").strip()
        if not src or src in seen:
            continue
        seen.add(src)
        sources.append(src)
    sources.sort()
    chunk_count = max(0, min(k_used, len(chunks_used)))
    return {
        "chunk_count": chunk_count,
        "sources": sources,
    }


def _parse_not_found_json(raw: str) -> tuple[str, list[str]]:
    text = (raw or "").strip()
    if not text:
        return "", []
    if "```" in text:
        low = text.lower()
        if "```json" in low:
            start = low.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        else:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text[:500], []
    if not isinstance(obj, dict):
        return text[:500], []
    result = obj.get("result") or obj.get("summary") or obj.get("message")
    result_text = result.strip() if isinstance(result, str) else ""
    follow_raw = obj.get("follow_up_questions") or obj.get("suggested_questions") or []
    follow_ups: list[str] = []
    if isinstance(follow_raw, list):
        for item in follow_raw:
            if isinstance(item, str) and item.strip():
                follow_ups.append(item.strip())
    return result_text, follow_ups


async def generate_not_found_response(
    *,
    question: str,
    chunks_used: list[dict],
    search_summary: dict[str, Any],
    infer_base: str,
    model: str,
    max_tokens: int,
    rerank_url: str,
    rerank_model: str,
    follow_up_candidates: int,
    follow_up_final: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[str, list[str], int, int, UsageTokens | None]:
    """
    One LLM call for a grounded miss explanation plus answerable follow-ups.

    Returns ``(result_text, follow_ups, chat_ms, rerank_ms, usage)``.
    """
    if not chunks_used:
        return (
            "I couldn't find that in the knowledge base.",
            [],
            0,
            0,
            None,
        )

    context_summary = _context_summary_for_followups(chunks_used)
    sources_line = ", ".join(search_summary.get("sources") or []) or "unknown sources"
    chunk_count = int(search_summary.get("chunk_count") or len(chunks_used))
    min_gen = max(2, min(follow_up_final, follow_up_candidates - 2))
    max_gen = follow_up_candidates

    sys = (
        "Return ONLY one valid JSON object with keys "
        '"result" (string) and "follow_up_questions" (array of strings). '
        "No markdown fences or commentary. "
        f'"result" must be one or two sentences explaining that the retrieved passages '
        f"do not answer the user's question — mention the topic they asked about. "
        "Do not claim the passages contain the answer. "
        f'"follow_up_questions" must have {min_gen}-{max_gen} entries, each under 120 characters, '
        "each directly answerable from the retrieved passages below only."
    )
    user = (
        f"User question:\n{question}\n\n"
        f"Search covered {chunk_count} knowledge chunk(s) across: {sources_line}.\n\n"
        f"Retrieved passages (only ground truth):\n{context_summary}\n\n"
        "Return JSON:\n"
        '{"result": "...", "follow_up_questions": ["...", "..."]}'
    )

    t0 = time.perf_counter()
    chat_result = await chat_complete(
        base_url=infer_base,
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=min(_NOT_FOUND_GEN_MAX_TOKENS, max(256, max_tokens)),
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    chat_ms = int(round((time.perf_counter() - t0) * 1000))
    result_text, candidates = _parse_not_found_json(chat_result.content)
    if not result_text:
        result_text = "I couldn't find that in the knowledge base."
        logger.info(
            "not_found_response_empty_result using fallback",
            extra={"raw_preview": _preview_for_log(chat_result.content or "")},
        )

    rerank_ms = 0
    follow_ups: list[str] = []
    if candidates:
        min_score = get_follow_up_min_context_rerank_score()
        t_rr = time.perf_counter()
        grounded, _debug = await _filter_follow_ups_by_context_rerank(
            candidates=candidates,
            chunks=chunks_used,
            rerank_url=rerank_url,
            rerank_model=rerank_model,
            min_score=min_score,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
        rerank_ms = int(round((time.perf_counter() - t_rr) * 1000))
        if grounded:
            follow_ups = await _rerank_follow_up_strings(
                context_summary=context_summary,
                question=question,
                candidates=grounded,
                rerank_url=rerank_url,
                rerank_model=rerank_model,
                top_n=follow_up_final,
                request_id=request_id,
                session_id=session_id,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        else:
            logger.info(
                "not_found_follow_ups_empty reason=context_rerank_filtered_all",
                extra={"candidate_count": len(candidates)},
            )

    return result_text, follow_ups[:follow_up_final], chat_ms, rerank_ms, chat_result.usage
