"""Unit tests for follow-up cache helpers."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.cache.follow_up import get_cached_follow_ups, set_cached_follow_ups


def test_get_cached_follow_ups_hit():
    payload = json.dumps({"questions": ["Q1?", "Q2?"]}).encode("utf-8")

    async def run():
        client = AsyncMock()
        client.get = AsyncMock(return_value=payload)
        with patch("app.cache.follow_up.get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = client
            return await get_cached_follow_ups("followup:test")

    assert asyncio.run(run()) == ["Q1?", "Q2?"]


def test_set_cached_follow_ups_skips_empty():
    async def run():
        client = AsyncMock()
        with patch("app.cache.follow_up.get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = client
            await set_cached_follow_ups("k", [])
            client.setex.assert_not_awaited()

    asyncio.run(run())
