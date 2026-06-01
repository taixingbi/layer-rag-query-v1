"""Thread resolution from body + upstream headers."""

from app.http._conversation_headers import resolve_thread_from_request


class _Req:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def test_resolve_thread_header_conversation_id():
    req = _Req({"x-conversation-id": "conv-upstream", "x-is-new-conversation": "false"})
    cid, is_new = resolve_thread_from_request(req, None)
    assert cid == "conv-upstream"
    assert is_new is False


def test_resolve_thread_mints_when_blank():
    req = _Req({})
    cid, is_new = resolve_thread_from_request(req, None)
    assert cid.startswith("conv_")
    assert is_new is True
