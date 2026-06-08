"""Follow-up question generation: extra LLM call + reranker over candidate strings.

Public surface: ``generate_follow_ups`` — invoked by ``app.rag.rag_answer.complete_rag_answer``
after the main RAG answer is produced. All failures are logged and degrade to ``[]`` so
the primary RAG response shape stays stable.
"""
from __future__ import annotations

import json
import logging
import time

from app.core.config import get_follow_up_min_context_rerank_score
from app.http.inference import chat_complete
from app.http.usage import UsageTokens
from app.http.rerank import rerank_texts
from app.core.logging_config import logger

_FOLLOW_UP_GEN_MAX_TOKENS_CAP = 512
_NOT_FOUND_REPLY = "NOT_FOUND"
_CONTEXT_SUMMARY_MAX_CHARS = 12_000


def _elapsed_ms(since: float) -> int:
    """Wall time in milliseconds from ``time.perf_counter()`` mark ``since``."""
    return int(round((time.perf_counter() - since) * 1000))


def _preview_for_log(text: str, *, max_chars: int = 400) -> str:
    """One-line safe truncation for stderr JSON (newlines escaped)."""
    s = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(s) > max_chars:
        return s[:max_chars] + "…"
    return s


def _sanitize_for_log(text: str) -> str:
    """Single-line safe form (newlines escaped) for stderr JSON; no truncation."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def _context_summary_for_followups(
    chunks: list[dict],
    *,
    max_chars: int = _CONTEXT_SUMMARY_MAX_CHARS,
) -> str:
    """Numbered passage block for follow-up prompts (same chunks as answer context)."""
    parts: list[str] = []
    size = 0
    cite_id = 0
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        src = (c.get("source") or "").strip() or "(unknown source)"
        cite_id += 1
        block = f"[{cite_id}] {src}\n{text}"
        if size + len(block) + 2 > max_chars:
            break
        parts.append(block)
        size += len(block) + 2
    return "\n\n".join(parts) if parts else "(no context)"


def _chunk_passage_texts(chunks: list[dict]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if text:
            out.append(text)
    return out


_FOLLOW_UP_OBJECT_KEYS = ("follow_up_questions", "questions", "follow_ups")


def _coerce_to_strings(value: object) -> list[str]:
    """Pull question strings from supported shapes: list, or dict with one of the known keys."""
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str)]
    if isinstance(value, dict):
        for key in _FOLLOW_UP_OBJECT_KEYS:
            v = value.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        for v in value.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return list(v)
    return []


def _parse_follow_up_json(raw: str) -> tuple[list[str], str | None]:
    """
    Parse model output into a list of non-empty question strings.

    Accepts the canonical shape ``{"follow_up_questions": ["…", "…"]}`` and
    legacy bare arrays. Recovers from common LLM glitches: code fences,
    concatenated top-level JSON values (``["Q1"]["Q2"]["Q3"]``), and
    array-fragment slices.

    When the returned list is empty, the second element is a stable machine
    reason for logging (``None`` when the list is non-empty).
    """
    text = raw.strip()
    if not text:
        return [], "empty_model_reply"
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
    if not text.strip():
        return [], "empty_after_code_fence_strip"

    decoder = json.JSONDecoder()
    values: list[object] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        values.append(value)
        i = end

    if not values:
        i0 = text.find("[")
        i1 = text.rfind("]")
        if i0 == -1 or i1 <= i0:
            return [], "json_invalid_no_array_slice"
        try:
            values = [json.loads(text[i0 : i1 + 1])]
        except json.JSONDecodeError:
            return [], "json_invalid_bracket_slice_failed"

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for s in _coerce_to_strings(value):
            t = s.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    if not out:
        if len(values) == 1 and not isinstance(values[0], (list, dict)):
            return [], f"parsed_not_list:{type(values[0]).__name__}"
        return [], "parsed_list_no_non_empty_strings"
    return out, None


async def _generate_follow_up_candidates(
    *,
    question: str,
    context_summary: str,
    infer_base: str,
    model: str,
    min_count: int,
    max_count: int,
    max_tokens: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[list[str], str, UsageTokens | None]:
    """One chat call: returns ``(candidates, raw, usage)``."""
    sys = (
        "Return ONLY one valid JSON object with a single key "
        '"follow_up_questions" whose value is an array of strings. '
        "No markdown, no commentary, no extra keys, no multiple objects. "
        f"Produce between {min_count} and {max_count} distinct questions, "
        "each under 120 characters. "
        "The retrieved passages below are the ONLY knowledge available. "
        "Each follow-up must be directly answerable from that text alone — "
        "do not ask about topics, policies, or facts not stated in the passages. "
        "Prefer drill-downs on details already present (who/what/when/where/how). "
        "Do not repeat the original question verbatim."
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"Retrieved passages (ground truth — follow-ups must be answerable from this text only):\n"
        f"{context_summary}\n\n"
        f'Return EXACTLY this shape with {min_count}-{max_count} entries:\n'
        '{"follow_up_questions": ["question 1", "question 2", "question 3"]}'
    )
    chat_result = await chat_complete(
        base_url=infer_base,
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    raw = chat_result.content
    candidates, empty_reason = _parse_follow_up_json(raw)
    if not candidates:
        preview = _preview_for_log(raw) if raw.strip() else "-"
        lvl = logging.WARNING if empty_reason != "empty_model_reply" else logging.INFO
        logger.log(
            lvl,
            "follow_up_questions_empty reason=%s reply_chars=%s raw_preview=%s",
            empty_reason,
            len(raw),
            preview,
            extra={"follow_up_empty_reason": empty_reason},
        )
    return candidates, raw, chat_result.usage


async def _filter_follow_ups_by_context_rerank(
    *,
    candidates: list[str],
    chunks: list[dict],
    rerank_url: str,
    rerank_model: str,
    min_score: float,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Keep candidates whose best passage rerank score meets ``min_score``.

    Returns ``(kept_questions, debug_rows)`` where each debug row has
    ``question``, ``top_score``, and ``kept``.
    """
    passages = _chunk_passage_texts(chunks)
    if not candidates or not passages:
        return [], []

    kept: list[str] = []
    debug_rows: list[dict] = []
    for q in candidates:
        try:
            rows = await rerank_texts(
                base_url=rerank_url,
                model=rerank_model,
                query=q,
                documents=passages,
                top_n=1,
                request_id=request_id,
                session_id=session_id,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.warning(
                "follow_up context filter rerank failed question=%s reason=%s",
                _preview_for_log(q, max_chars=80),
                str(e),
            )
            debug_rows.append({"question": q, "top_score": None, "kept": False})
            continue
        top_score = float(rows[0].get("score", 0.0)) if rows else 0.0
        ok = top_score >= min_score
        debug_rows.append({"question": q, "top_score": top_score, "kept": ok})
        if ok:
            kept.append(q)
    return kept, debug_rows


async def _rerank_follow_up_strings(
    *,
    context_summary: str,
    question: str,
    candidates: list[str],
    rerank_url: str,
    rerank_model: str,
    top_n: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> list[str]:
    """Rerank grounded candidates; query prefers retrieved passages over the answer."""
    if not candidates:
        return []
    n = min(top_n, len(candidates))
    rerank_query = context_summary.strip() or question
    try:
        rows = await rerank_texts(
            base_url=rerank_url,
            model=rerank_model,
            query=rerank_query,
            documents=candidates,
            top_n=n,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.warning("follow_up rerank failed reason=%s", str(e))
        return candidates[:n]
    out: list[str] = []
    seen: set[int] = set()
    for row in rows:
        idx = row.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates) or idx in seen:
            continue
        seen.add(idx)
        out.append(candidates[idx])
        if len(out) >= n:
            break
    if len(out) < n:
        for i, s in enumerate(candidates):
            if i in seen:
                continue
            out.append(s)
            if len(out) >= n:
                break
    return out if out else candidates[:n]


async def generate_follow_ups(
    *,
    question: str,
    answer: str,
    chunks_used: list[dict],
    follow_up_candidates: int,
    follow_up_final: int,
    infer_base: str,
    model: str,
    max_tokens_main: int,
    rerank_url: str,
    rerank_model: str,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[list[str], int, int, UsageTokens | None]:
    """Returns ``(questions, chat_ms, rerank_ms, chat_usage)``; times are zero when skipped."""
    if not chunks_used:
        logger.info(
            "follow_up_questions_empty reason=no_chunks_used",
            extra={"follow_up_empty_reason": "no_chunks_used"},
        )
        return [], 0, 0, None
    if not answer or answer.strip() == _NOT_FOUND_REPLY:
        logger.info(
            "follow_up_questions_empty reason=answer_not_found",
            extra={"follow_up_empty_reason": "answer_not_found"},
        )
        return [], 0, 0, None

    min_gen = max(3, follow_up_candidates - 3)
    max_gen = follow_up_candidates
    if min_gen > max_gen:
        min_gen = max_gen
    summary = _context_summary_for_followups(chunks_used)
    min_context_score = get_follow_up_min_context_rerank_score()
    gen_budget = min(_FOLLOW_UP_GEN_MAX_TOKENS_CAP, max(256, max_tokens_main))
    gen_t0 = time.perf_counter()
    try:
        candidates, raw, chat_usage = await _generate_follow_up_candidates(
            question=question,
            context_summary=summary,
            infer_base=infer_base,
            model=model,
            min_count=min_gen,
            max_count=max_gen,
            max_tokens=gen_budget,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.warning(
            "follow_up_questions_empty reason=generation_failed detail=%s",
            str(e),
            extra={"follow_up_empty_reason": "generation_failed", "error_message": str(e)},
        )
        return [], _elapsed_ms(gen_t0), 0, None
    gen_ms = _elapsed_ms(gen_t0)
    if not candidates:
        return [], gen_ms, 0, chat_usage

    rr_t0 = time.perf_counter()
    grounded, filter_debug = await _filter_follow_ups_by_context_rerank(
        candidates=candidates,
        chunks=chunks_used,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
        min_score=min_context_score,
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    if not grounded:
        logger.info(
            "follow_up_questions_empty reason=context_rerank_filtered_all min_score=%s",
            min_context_score,
            extra={
                "follow_up_empty_reason": "context_rerank_filtered_all",
                "follow_up_filter_debug": filter_debug,
            },
        )
        return [], gen_ms, _elapsed_ms(rr_t0), chat_usage

    ranked = await _rerank_follow_up_strings(
        context_summary=summary,
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
    rr_ms = _elapsed_ms(rr_t0)
    if ranked:
        logger.info(
            "follow_up_questions_ok cand=%s grounded=%s ranked=%s min_context_score=%s",
            len(candidates),
            len(grounded),
            len(ranked),
            min_context_score,
            extra={
                "follow_up_raw_reply": _sanitize_for_log(raw),
                "follow_up_candidates_full": list(candidates),
                "follow_up_candidates_count": len(candidates),
                "follow_up_grounded": list(grounded),
                "follow_up_filter_debug": filter_debug,
                "follow_up_ranked": list(ranked),
                "follow_up_ranked_count": len(ranked),
                "latency_follow_up_chat_ms": gen_ms,
                "latency_follow_up_rerank_ms": rr_ms,
            },
        )
    return ranked, gen_ms, rr_ms, chat_usage
