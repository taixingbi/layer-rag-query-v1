"""RAG latency_ms response shape."""

from __future__ import annotations

from app.rag.latency import LATENCY_PHASES, build_latency_ms


def test_build_latency_ms_keys_and_sums() -> None:
    out = build_latency_ms(
        embed_ms=100,
        retrieve_ms=50,
        chunk_rerank_ms=30,
        chat_ms=600,
        follow_up_chat_ms=200,
        follow_up_rerank_ms=20,
        total_ms=1000,
    )
    assert tuple(out.keys()) == LATENCY_PHASES
    assert out["github_readme"] == 100
    assert out["github_search"] == 80
    assert out["follow_up_chat"] == 220
    assert out["chat"] == 600
    assert out["total"] == 1000
