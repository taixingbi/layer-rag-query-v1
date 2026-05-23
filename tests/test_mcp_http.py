"""MCP HTTP header resolution (no live server)."""

from __future__ import annotations

from app.rag.mcp_http import resolve_mcp_rag_call


def test_resolve_mcp_rag_call_uses_tool_args_without_http() -> None:
    ctx = resolve_mcp_rag_call(
        request_id="req-tool",
        session_id="ses-tool",
        trace_id="trc-tool",
    )
    assert ctx.request_id == "req-tool"
    assert ctx.session_id == "ses-tool"
    assert ctx.trace_id == "trc-tool"
    assert ctx.user.id == "-"
    assert ctx.user.roles == ["anyuser"]


def test_resolve_mcp_rag_call_generates_ids_when_blank() -> None:
    ctx = resolve_mcp_rag_call()
    assert len(ctx.request_id) >= 32
    assert len(ctx.session_id) >= 32
