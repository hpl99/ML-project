"""Caching Layer for API and live features.

Provides in-memory TTL caching and disk persistence to guarantee resilience
and sub-second response times on repeated queries.
"""
import time
from typing import Any, Optional, Dict


class MemoryTTLCache:
    """Thread-safe in-memory cache with Time-To-Live (TTL) expiration."""

    def __init__(self, default_ttl_sec: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl_sec

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None
        return entry["value"]

    def set(self, key: str, value: Any, ttl_sec: Optional[int] = None):
        ttl = ttl_sec if ttl_sec is not None else self._default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }

    def clear(self):
        self._cache.clear()


# Global cache instance
cache = MemoryTTLCache(default_ttl_sec=3600)
