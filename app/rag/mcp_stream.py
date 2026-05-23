"""Live RAG event streaming over MCP HTTP (progress notifications)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Context


def mcp_progress_token(ctx: Context) -> str | int:
    """Token for ``notifications/progress`` (client token or this request's id)."""
    if ctx.request_context and ctx.request_context.meta:
        token = ctx.request_context.meta.progressToken
        if token is not None:
            return token
    return ctx.request_id


async def emit_mcp_rag_event(ctx: Context, seq: int, event: dict[str, Any]) -> None:
    """Push one RAG SSE-shaped event to the client during ``tools/call`` (streamable HTTP).

    Each notification's ``message`` is a JSON object with a ``type`` field (same as
    ``docs/streaming.md`` event names). Clients see multiple ``event: message`` frames
    before the final tool result.
    """
    await ctx.session.send_progress_notification(
        progress_token=mcp_progress_token(ctx),
        progress=float(seq),
        message=json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        related_request_id=ctx.request_id,
    )
