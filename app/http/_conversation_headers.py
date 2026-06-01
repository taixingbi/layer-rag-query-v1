"""Parse upstream thread headers (orchestrator / gateway)."""

from __future__ import annotations

from starlette.requests import Request


def is_new_conversation_from_header(request: Request) -> bool | None:
    """``X-Is-New-Conversation: true|false`` when set by upstream; else ``None``."""
    raw = (request.headers.get("x-is-new-conversation") or "").strip().lower()
    if raw == "true":
        return True
    if raw == "false":
        return False
    return None


def resolve_thread_from_request(
    request: Request,
    body_conversation_id: str | None,
) -> tuple[str, bool]:
    """Return (resolved conversation_id, is_new_conversation)."""
    header_cid = (request.headers.get("x-conversation-id") or "").strip()
    body_raw = (body_conversation_id or "").strip()
    had_client_thread = bool(body_raw or header_cid)
    header_flag = is_new_conversation_from_header(request)
    if header_flag is not None:
        is_new = header_flag
    else:
        is_new = not had_client_thread
    from app.http.inference import resolve_conversation_id

    conversation_id = resolve_conversation_id(body_raw or header_cid or None)
    return conversation_id, is_new
