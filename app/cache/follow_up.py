"""Follow-up question list cache."""

from __future__ import annotations

import json

from app.cache.client import get_redis
from app.cache.keys import build_follow_up_key
from app.core.config import get_cache_follow_up_ttl_seconds
from app.core.metrics import observe_cache_op


async def get_cached_follow_ups(key: str) -> list[str] | None:
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if not raw:
            observe_cache_op("follow_up", "miss")
            return None
        data = json.loads(raw.decode("utf-8"))
        questions = data.get("questions")
        if not isinstance(questions, list):
            observe_cache_op("follow_up", "error")
            return None
        out = [q for q in questions if isinstance(q, str) and q.strip()]
        if not out:
            observe_cache_op("follow_up", "miss")
            return None
        observe_cache_op("follow_up", "hit")
        return out
    except Exception:
        observe_cache_op("follow_up", "error")
        return None


async def set_cached_follow_ups(key: str, questions: list[str]) -> None:
    if not questions:
        return
    client = await get_redis()
    if client is None:
        return
    ttl = get_cache_follow_up_ttl_seconds()
    payload = json.dumps({"questions": questions}).encode("utf-8")
    try:
        await client.setex(key, ttl, payload)
    except Exception:
        observe_cache_op("follow_up", "error")


def follow_up_cache_key(
    *,
    collection_base: str,
    question: str,
    answer: str,
    chunks: list[dict],
    acl: str,
    cfg_hash: str,
) -> str:
    return build_follow_up_key(
        collection_base=collection_base,
        question=question,
        answer=answer,
        chunks=chunks,
        acl=acl,
        cfg_hash=cfg_hash,
    )
