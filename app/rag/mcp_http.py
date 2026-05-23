"""Resolve correlation and access-control for MCP RAG tools over HTTP."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.http._correlation import correlation_from_request
from app.http.inference import resolve_conversation_id
from app.rag.access import RagUser


@dataclass(frozen=True)
class McpRagCallContext:
    request_id: str
    session_id: str
    trace_id: str | None
    user: RagUser
    conversation_id: str


def resolve_mcp_rag_call(
    *,
    request_id: str = "",
    session_id: str = "",
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> McpRagCallContext:
    """Merge tool arguments with inbound HTTP headers (headers win when set).

    On **stdio** transport there is no HTTP request; ``request_id`` / ``session_id`` tool
    arguments are used, with UUIDs generated when blank. On **HTTP** transport, set
    ``X-Request-Id``, ``X-Session-Id``, ``X-Trace-Id``, and access headers on each
    ``POST /v1/mcp`` call (same as ``POST /v1/rag/query``).
    """
    rid = (request_id or "").strip()
    sid = (session_id or "").strip()
    tid = (trace_id or "").strip() or None
    user: RagUser | None = None

    try:
        from fastmcp.server.dependencies import get_http_request

        http_request = get_http_request()
    except Exception:
        http_request = None

    if http_request is not None:
        hdr_rid, hdr_sid, hdr_tid = correlation_from_request(http_request)
        if hdr_rid:
            rid = hdr_rid
        if hdr_sid:
            sid = hdr_sid
        if hdr_tid is not None:
            tid = hdr_tid
        user = RagUser.from_headers(http_request)

    if not rid:
        rid = str(uuid.uuid4())
    if not sid:
        sid = str(uuid.uuid4())

    if user is None:
        user = RagUser()

    return McpRagCallContext(
        request_id=rid,
        session_id=sid,
        trace_id=tid,
        user=user,
        conversation_id=resolve_conversation_id(conversation_id),
    )
