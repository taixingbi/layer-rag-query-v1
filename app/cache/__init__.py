"""Redis-backed RAG caches (optional; disabled without REDIS_URL)."""

from app.cache.client import ping_redis
from app.cache.embedding import get_cached_embedding, set_cached_embedding
from app.cache.follow_up import follow_up_cache_key, get_cached_follow_ups, set_cached_follow_ups
from app.cache.miss import MissBundle, get_cached_miss, miss_cache_key, set_cached_miss

__all__ = [
    "MissBundle",
    "follow_up_cache_key",
    "get_cached_embedding",
    "get_cached_follow_ups",
    "get_cached_miss",
    "miss_cache_key",
    "ping_redis",
    "set_cached_embedding",
    "set_cached_follow_ups",
    "set_cached_miss",
]
