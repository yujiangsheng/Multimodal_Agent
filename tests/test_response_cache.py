"""Tests for ResponseCache — LRU cache with TTL for LLM responses."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from pinocchio.utils.response_cache import ResponseCache


class TestResponseCacheBasics:
    """Core get/put/eviction behaviour."""

    def test_put_and_get(self):
        cache = ResponseCache(capacity=10, ttl_seconds=60)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_miss(self):
        cache = ResponseCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = ResponseCache(capacity=10, ttl_seconds=0.05)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"
        time.sleep(0.1)
        assert cache.get("k1") is None  # expired

    def test_lru_eviction(self):
        cache = ResponseCache(capacity=2, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    def test_access_renews_lru_order(self):
        cache = ResponseCache(capacity=2, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.get("k1")  # renew k1
        cache.put("k3", "v3")  # should evict k2, not k1
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None

    def test_overwrite_existing_key(self):
        cache = ResponseCache(capacity=10, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.put("k1", "v2")
        assert cache.get("k1") == "v2"
        assert cache.size == 1

    def test_invalidate(self):
        cache = ResponseCache()
        cache.put("k1", "v1")
        assert cache.invalidate("k1") is True
        assert cache.get("k1") is None
        assert cache.invalidate("k1") is False

    def test_clear(self):
        cache = ResponseCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestResponseCacheMetrics:
    """Hit/miss tracking and statistics."""

    def test_hit_miss_tracking(self):
        cache = ResponseCache()
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        assert cache.hits == 1
        assert cache.misses == 1
        assert abs(cache.hit_rate - 0.5) < 0.01

    def test_hit_rate_zero_when_empty(self):
        cache = ResponseCache()
        assert cache.hit_rate == 0.0

    def test_stats(self):
        cache = ResponseCache(capacity=100, ttl_seconds=300)
        s = cache.stats()
        assert s["capacity"] == 100
        assert s["ttl_seconds"] == 300
        assert s["size"] == 0
        assert s["hits"] == 0
        assert s["misses"] == 0


class TestResponseCacheKeyGeneration:
    """Tests for make_key."""

    def test_make_key_deterministic(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = ResponseCache.make_key(msgs)
        k2 = ResponseCache.make_key(msgs)
        assert k1 == k2

    def test_make_key_different_for_different_messages(self):
        k1 = ResponseCache.make_key([{"role": "user", "content": "hello"}])
        k2 = ResponseCache.make_key([{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_make_key_includes_extra_params(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = ResponseCache.make_key(msgs, temperature=0.7)
        k2 = ResponseCache.make_key(msgs, temperature=0.9)
        assert k1 != k2


class TestResponseCacheThreadSafety:
    """Concurrent access should not corrupt state."""

    def test_concurrent_put_get(self):
        cache = ResponseCache(capacity=100, ttl_seconds=60)
        errors = []

        def worker(start: int):
            try:
                for i in range(start, start + 50):
                    cache.put(f"k{i}", f"v{i}")
                    cache.get(f"k{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cache.size <= 200


class TestOrchestratorIntegration:
    """Verify the three features are wired into the orchestrator."""

    def _make_orchestrator(self):
        from unittest.mock import MagicMock, patch

        with patch("pinocchio.orchestrator.LLMClient") as MockLLM, \
             patch("pinocchio.orchestrator.PinocchioAgent"), \
             patch("pinocchio.orchestrator.MemoryManager"), \
             patch("pinocchio.orchestrator.PinocchioLogger"):
            mock_llm_instance = MagicMock()
            mock_llm_instance.num_ctx = 8192
            MockLLM.return_value = mock_llm_instance

            from pinocchio.orchestrator import Pinocchio
            return Pinocchio()

    def test_has_input_guard(self):
        p = self._make_orchestrator()
        assert hasattr(p, "_input_guard")

    def test_has_context_manager(self):
        p = self._make_orchestrator()
        assert hasattr(p, "_context_manager")

    def test_has_response_cache(self):
        p = self._make_orchestrator()
        assert hasattr(p, "_response_cache")

    def test_status_includes_new_features(self):
        p = self._make_orchestrator()
        status = p.status()
        assert "context_manager" in status
        assert "response_cache" in status

    def test_reset_clears_cache(self):
        p = self._make_orchestrator()
        p._response_cache.put("test", "value")
        p.reset()
        assert p._response_cache.size == 0
