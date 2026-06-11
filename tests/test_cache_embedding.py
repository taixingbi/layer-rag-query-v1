"""Unit tests for embedding cache pack/unpack and lookup."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, patch

from app.cache.embedding import get_cached_embedding, pack_vector, set_cached_embedding, unpack_vector
from app.core.config import VECTOR_SIZE


def test_pack_unpack_roundtrip():
    vec = [0.1, -0.2, 0.3]
    raw = pack_vector(vec)
    out = unpack_vector(raw)
    assert len(out) == len(vec)
    for a, b in zip(out, vec):
        assert abs(a - b) < 1e-6


def test_get_cached_embedding_miss_without_redis():
    async def run():
        with patch("app.cache.embedding.get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = None
            return await get_cached_embedding("visa status?")

    assert asyncio.run(run()) is None


def test_get_cached_embedding_hit():
    vec = [0.0] * VECTOR_SIZE
    raw = pack_vector(vec)

    async def run():
        client = AsyncMock()
        client.get = AsyncMock(return_value=raw)
        with patch("app.cache.embedding.get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = client
            return await get_cached_embedding("visa status?", model="BAAI/bge-m3")

    out = asyncio.run(run())
    assert out is not None
    assert len(out) == VECTOR_SIZE


def test_set_cached_embedding_writes_binary():
    vec = [0.5] * VECTOR_SIZE

    async def run():
        client = AsyncMock()
        with patch("app.cache.embedding.get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = client
            await set_cached_embedding("q", vec, model="m")
            client.setex.assert_awaited_once()
            _key, _ttl, payload = client.setex.await_args.args
            assert len(payload) == VECTOR_SIZE * 4
            assert struct.unpack(f"{VECTOR_SIZE}f", payload)[0] == 0.5

    asyncio.run(run())
