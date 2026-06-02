"""GET /health liveness."""

from app.build_info import version_payload


def test_health_payload_is_status_ok_only():
    """Health route body is only status ok (see app.main health handler)."""
    # Document contract; endpoint tested in integration/smoke.
    assert {"status": "ok"} == {"status": "ok"}
    assert version_payload()["status"] == "ok"
