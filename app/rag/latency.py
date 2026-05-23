"""RAG response ``latency_ms`` shape (stable keys for HTTP, MCP, and SSE)."""

from __future__ import annotations

# Public phase keys (match cross-service agent tooling conventions).
LATENCY_GITHUB_README = "github_readme"
LATENCY_GITHUB_SEARCH = "github_search"
LATENCY_CHAT = "chat"
LATENCY_FOLLOW_UP_CHAT = "follow_up_chat"
LATENCY_TOTAL = "total"

LATENCY_PHASES = (
    LATENCY_GITHUB_README,
    LATENCY_GITHUB_SEARCH,
    LATENCY_CHAT,
    LATENCY_FOLLOW_UP_CHAT,
    LATENCY_TOTAL,
)


def build_latency_ms(
    *,
    embed_ms: int,
    retrieve_ms: int,
    chunk_rerank_ms: int,
    chat_ms: int,
    follow_up_chat_ms: int,
    follow_up_rerank_ms: int,
    total_ms: int,
) -> dict[str, int]:
    """Map internal timings to the external ``latency_ms`` object."""
    return {
        LATENCY_GITHUB_README: embed_ms,
        LATENCY_GITHUB_SEARCH: retrieve_ms + chunk_rerank_ms,
        LATENCY_CHAT: chat_ms,
        LATENCY_FOLLOW_UP_CHAT: follow_up_chat_ms + follow_up_rerank_ms,
        LATENCY_TOTAL: total_ms,
    }
