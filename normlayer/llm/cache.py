"""In-memory LRU cache with TTL for LLM judgment results."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any


class JudgmentCache:
    """Thread-safe in-memory LRU cache with per-entry TTL.

    Used by ``LLMJudge`` to avoid redundant LLM calls for identical prompts.
    Cache keys are SHA-256 hashes of the full prompt text.

    Args:
        max_size: Maximum number of entries before LRU eviction (default 256).
        ttl_seconds: Time-to-live for each entry in seconds (default 3600).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    @staticmethod
    def _make_key(prompt: str) -> str:
        """Create a SHA-256 hash key from prompt text.

        Args:
            prompt: The full prompt string.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Any | None:
        """Retrieve a cached result if it exists and hasn't expired.

        Args:
            prompt: The prompt text used as the cache key.

        Returns:
            The cached value, or None on miss or expiry.
        """
        key = self._make_key(prompt)
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.monotonic() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def put(self, prompt: str, value: Any) -> None:
        """Store a result in the cache, evicting the oldest entry if full.

        Args:
            prompt: The prompt text used as the cache key.
            value: The result to cache.
        """
        key = self._make_key(prompt)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
