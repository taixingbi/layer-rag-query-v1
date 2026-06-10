"""Optional async Redis client (fail-open when unset or unreachable)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.config import cache_enabled, get_redis_url

if TYPE_CHECKING:
    from redis.asyncio import Redis

_log = logging.getLogger("layer_rag_query.cache")
_redis: Redis | None = None
_redis_failed = False


async def get_redis() -> Redis | None:
    global _redis, _redis_failed
    if not cache_enabled():
        return None
    if _redis_failed:
        return None
    if _redis is not None:
        return _redis
    url = get_redis_url()
    if not url:
        return None
    try:
        from redis.asyncio import Redis as AsyncRedis

        _redis = AsyncRedis.from_url(url, decode_responses=False)
        await _redis.ping()
        return _redis
    except Exception as e:
        _redis_failed = True
        _log.warning("redis_connect_failed error=%s", str(e))
        return None


async def ping_redis() -> tuple[bool, str | None]:
    """Return (ok, error_detail). Used by /cache/health."""
    if not cache_enabled():
        return True, "cache_disabled"
    client = await get_redis()
    if client is None:
        url = get_redis_url()
        if not url:
            return True, "redis_not_configured"
        return False, "redis_unreachable"
    try:
        await client.ping()
        return True, None
    except Exception as e:
        return False, str(e)[:200]
