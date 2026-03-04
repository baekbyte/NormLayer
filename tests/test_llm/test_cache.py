"""Tests for JudgmentCache — LRU with TTL."""

import time
from unittest.mock import patch

from normlayer.llm.cache import JudgmentCache


class TestCacheBasics:
    def test_put_and_get(self):
        cache = JudgmentCache()
        cache.put("prompt1", "result1")
        assert cache.get("prompt1") == "result1"

    def test_miss_returns_none(self):
        cache = JudgmentCache()
        assert cache.get("nonexistent") is None

    def test_len(self):
        cache = JudgmentCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2

    def test_clear(self):
        cache = JudgmentCache()
        cache.put("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None


class TestCacheTTL:
    def test_expired_entry_returns_none(self):
        cache = JudgmentCache(ttl_seconds=0.1)
        cache.put("prompt", "value")
        # Simulate time passing
        with patch("normlayer.llm.cache.time.monotonic", return_value=time.monotonic() + 1.0):
            assert cache.get("prompt") is None

    def test_fresh_entry_returned(self):
        cache = JudgmentCache(ttl_seconds=3600)
        cache.put("prompt", "value")
        assert cache.get("prompt") == "value"


class TestCacheLRU:
    def test_eviction_when_full(self):
        cache = JudgmentCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_lru_order(self):
        cache = JudgmentCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # refresh "a"
        cache.put("c", 3)  # should evict "b" (least recently used)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
