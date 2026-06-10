"""NOT_FOUND sentinel → user-facing copy."""

from __future__ import annotations

from app.rag.rag_answer import _NOT_FOUND_REPLY, _NOT_FOUND_USER_MESSAGE, user_facing_answer


def test_user_facing_answer_maps_sentinel():
    assert user_facing_answer(_NOT_FOUND_REPLY) == _NOT_FOUND_USER_MESSAGE
    assert user_facing_answer(f"  {_NOT_FOUND_REPLY}  ") == _NOT_FOUND_USER_MESSAGE
    assert user_facing_answer("") == _NOT_FOUND_USER_MESSAGE


def test_user_facing_answer_passthrough():
    assert user_facing_answer("Taixing works at Saks.") == "Taixing works at Saks."


def test_parse_not_found_json():
    from app.rag.not_found_response import _parse_not_found_json

    raw = '{"result": "No hobby info.", "follow_up_questions": ["Where does he work?"]}'
    result, follow_ups = _parse_not_found_json(raw)
    assert result == "No hobby info."
    assert follow_ups == ["Where does he work?"]
