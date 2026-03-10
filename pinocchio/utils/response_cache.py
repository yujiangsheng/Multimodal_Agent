"""Response Cache — LRU cache with TTL for LLM responses.

Caches LLM responses keyed by a fingerprint of the input messages,
avoiding redundant API calls for repeated or near-identical queries.

Features
--------
- **LRU eviction** — least-recently-used entries evicted when capacity reached.
- **TTL expiry** — entries expire after a configurable time-to-live.
- **Hit / miss metrics** — track cache effectiveness.
- **Thread-safe** — safe for concurrent access.

Usage
-----
>>> from pinocchio.utils.response_cache import ResponseCache
>>> cache = ResponseCache(capacity=200, ttl_seconds=600)
>>> cache.get("key")       # None (miss)
>>> cache.put("key", "response text")
>>> cache.get("key")       # "response text" (hit)
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


# =====================================================================
# Cache entry
# =====================================================================

@dataclass
class _CacheEntry:
    """Internal cache entry with value and expiry timestamp."""
    value: str
    expires_at: float
    created_at: float = field(default_factory=time.monotonic)


# =====================================================================
# Response Cache
# =====================================================================

class ResponseCache:
    """Thread-safe LRU response cache with TTL expiry.

    Parameters
    ----------
    capacity : int
        Maximum number of cached entries.
    ttl_seconds : float
        Time-to-live for each cache entry (seconds).  Entries older
        than this are treated as misses and evicted lazily.
    """

    def __init__(
        self,
        capacity: int = 256,
        ttl_seconds: float = 600.0,
    ) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Look up a cached response.  Returns ``None`` on miss."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() > entry.expires_at:
                # Expired — evict
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: str) -> None:
        """Store a response in the cache."""
        now = time.monotonic()
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = _CacheEntry(value=value, expires_at=now + self._ttl)
            else:
                if len(self._store) >= self._capacity:
                    self._store.popitem(last=False)  # evict LRU
                self._store[key] = _CacheEntry(value=value, expires_at=now + self._ttl)

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry.  Returns True if found."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries and reset metrics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(messages: list[dict[str, Any]], **extra: Any) -> str:
        """Generate a cache key from a message list.

        Uses a SHA-256 hash of the JSON-serialised messages plus any
        extra keyword arguments (temperature, json_mode, etc.).
        """
        payload = {"messages": messages}
        payload.update(extra)
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a float in [0, 1]."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": self.size,
            "capacity": self._capacity,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }
