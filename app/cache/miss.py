"""Structured NOT_FOUND response bundle cache."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.cache.client import get_redis
from app.cache.keys import build_miss_key
from app.core.config import get_cache_miss_ttl_seconds
from app.core.metrics import observe_cache_op


@dataclass(frozen=True)
class MissBundle:
    result: str
    follow_up_questions: list[str]


async def get_cached_miss(key: str) -> MissBundle | None:
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if not raw:
            observe_cache_op("miss", "miss")
            return None
        data = json.loads(raw.decode("utf-8"))
        result = data.get("result")
        follow_raw = data.get("follow_up_questions")
        if not isinstance(result, str) or not result.strip():
            observe_cache_op("miss", "error")
            return None
        follow_ups: list[str] = []
        if isinstance(follow_raw, list):
            follow_ups = [q for q in follow_raw if isinstance(q, str) and q.strip()]
        observe_cache_op("miss", "hit")
        return MissBundle(result=result.strip(), follow_up_questions=follow_ups)
    except Exception:
        observe_cache_op("miss", "error")
        return None


async def set_cached_miss(key: str, *, result: str, follow_up_questions: list[str]) -> None:
    if not result.strip():
        return
    client = await get_redis()
    if client is None:
        return
    ttl = get_cache_miss_ttl_seconds()
    payload = json.dumps(
        {"result": result.strip(), "follow_up_questions": follow_up_questions}
    ).encode("utf-8")
    try:
        await client.setex(key, ttl, payload)
    except Exception:
        observe_cache_op("miss", "error")


def miss_cache_key(
    *,
    collection_base: str,
    question: str,
    chunks: list[dict],
    acl: str,
    cfg_hash: str,
) -> str:
    return build_miss_key(
        collection_base=collection_base,
        question=question,
        chunks=chunks,
        acl=acl,
        cfg_hash=cfg_hash,
    )
