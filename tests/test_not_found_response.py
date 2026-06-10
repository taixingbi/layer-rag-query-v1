"""Unit tests for NOT_FOUND search summary (no inference)."""

from __future__ import annotations

from app.rag.not_found_response import build_search_summary


def test_build_search_summary_dedupes_sources():
    chunks = [
        {"source": "employment_history", "text": "a"},
        {"source": "personal_profile", "text": "b"},
        {"source": "personal_profile", "text": "c"},
        {"source": "education", "text": "d"},
    ]
    summary = build_search_summary(chunks, k_used=4)
    assert summary["chunk_count"] == 4
    assert summary["sources"] == ["education", "employment_history", "personal_profile"]


def test_build_search_summary_respects_k_used():
    chunks = [{"source": f"src{i}", "text": "x"} for i in range(10)]
    summary = build_search_summary(chunks, k_used=5)
    assert summary["chunk_count"] == 5
