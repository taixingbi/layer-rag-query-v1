"""Unit tests for grounded follow-up question helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.rag.follow_up import (
    _context_summary_for_followups,
    _filter_follow_ups_by_context_rerank,
    _parse_follow_up_json,
    generate_follow_ups,
)


def test_context_summary_includes_full_passage_text():
    chunks = [
        {
            "source": "personal_profile",
            "text": "Q: What is visa status?\nA: H4 EAD. No sponsorship required.",
        }
    ]
    summary = _context_summary_for_followups(chunks)
    assert "[1] personal_profile" in summary
    assert "H4 EAD" in summary
    assert "sponsorship required" in summary


def test_parse_follow_up_json_object_shape():
    raw = '{"follow_up_questions": ["What is the EAD renewal process?", "Is sponsorship required?"]}'
    out, reason = _parse_follow_up_json(raw)
    assert reason is None
    assert len(out) == 2


def test_filter_follow_ups_by_context_rerank_keeps_high_scores():
    chunks = [{"text": "Q: visa?\nA: H4 EAD only."}]

    async def run() -> tuple[list[str], list[dict]]:
        with patch("app.rag.follow_up.rerank_texts", new_callable=AsyncMock) as mock_rr:
            mock_rr.side_effect = [
                [{"index": 0, "score": 0.9}],
                [{"index": 0, "score": 0.1}],
            ]
            return await _filter_follow_ups_by_context_rerank(
                candidates=["What is H4 EAD?", "Can Taixing get H1B sponsorship?"],
                chunks=chunks,
                rerank_url="http://rerank",
                rerank_model="m",
                min_score=0.35,
                request_id="r",
                session_id="s",
            )

    kept, debug = asyncio.run(run())
    assert kept == ["What is H4 EAD?"]
    assert debug[0]["kept"] is True
    assert debug[1]["kept"] is False


def test_generate_follow_ups_skips_not_found():
    async def run():
        return await generate_follow_ups(
            question="q",
            answer="NOT_FOUND",
            chunks_used=[{"text": "some context", "source": "doc"}],
            follow_up_candidates=8,
            follow_up_final=3,
            infer_base="http://infer",
            model="m",
            max_tokens_main=512,
            rerank_url="http://rerank",
            rerank_model="rm",
            request_id="r",
            session_id="s",
        )

    out, chat_ms, rr_ms, usage = asyncio.run(run())
    assert out == []
    assert chat_ms == 0
    assert rr_ms == 0
    assert usage is None
