"""Layer 1 — query embedding vector cache."""

from __future__ import annotations

import struct

from app.cache.client import get_redis
from app.cache.keys import build_embed_key
from app.core.config import VECTOR_SIZE, get_cache_embed_ttl_seconds
from app.core.metrics import observe_cache_op


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_vector(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


async def get_cached_embedding(
    text: str,
    *,
    model: str | None = None,
) -> list[float] | None:
    client = await get_redis()
    if client is None:
        return None
    key = build_embed_key(text, model=model)
    try:
        raw = await client.get(key)
        if not raw:
            observe_cache_op("embed", "miss")
            return None
        vec = unpack_vector(raw)
        if len(vec) != VECTOR_SIZE:
            observe_cache_op("embed", "error")
            return None
        observe_cache_op("embed", "hit")
        return vec
    except Exception:
        observe_cache_op("embed", "error")
        return None


async def set_cached_embedding(
    text: str,
    vector: list[float],
    *,
    model: str | None = None,
) -> None:
    if len(vector) != VECTOR_SIZE:
        return
    client = await get_redis()
    if client is None:
        return
    key = build_embed_key(text, model=model)
    ttl = get_cache_embed_ttl_seconds()
    try:
        await client.setex(key, ttl, pack_vector(vector))
    except Exception:
        observe_cache_op("embed", "error")
