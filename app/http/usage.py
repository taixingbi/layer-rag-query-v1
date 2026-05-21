"""OpenAI-compatible token usage parsing and aggregation."""

from __future__ import annotations

from typing import TypedDict


class UsageTokens(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


ZERO_USAGE: UsageTokens = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def parse_usage(data: dict) -> UsageTokens | None:
    """Read top-level ``usage`` from a chat-completions JSON body."""
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    try:
        prompt = int(raw.get("prompt_tokens") or 0)
        completion = int(raw.get("completion_tokens") or 0)
        total = int(raw.get("total_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if total == 0 and prompt == 0 and completion == 0:
        return None
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def merge_usage(*parts: UsageTokens | None) -> UsageTokens:
    """Element-wise sum; ``None`` counts as zero."""
    prompt = completion = total = 0
    for p in parts:
        if not p:
            continue
        prompt += p["prompt_tokens"]
        completion += p["completion_tokens"]
        total += p["total_tokens"]
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def build_usage_payload(
    chat: UsageTokens | None,
    follow_up_chat: UsageTokens | None,
) -> dict[str, UsageTokens]:
    """Mirror ``latency_ms`` layout: ``chat``, ``follow_up_chat``, ``total``."""
    chat_u = chat or ZERO_USAGE
    fu_u = follow_up_chat or ZERO_USAGE
    return {
        "chat": chat_u,
        "follow_up_chat": fu_u,
        "total": merge_usage(chat_u, fu_u),
    }
