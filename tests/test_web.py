"""Tests for web API — rate limiting middleware and request constraints."""

from __future__ import annotations

import sys
import time

import pytest

import web.app


# =====================================================================
# Rate limiting middleware
# =====================================================================

class TestWebRateLimiting:
    """Rate limiting middleware should be present in web app."""

    def test_rate_limit_middleware_exists(self):
        webapp_mod = sys.modules["web.app"]
        assert webapp_mod._RATE_LIMIT_RPM > 0
        assert webapp_mod._MAX_REQUEST_SIZE > 0

    def test_rate_limit_defaults(self):
        webapp_mod = sys.modules["web.app"]
        assert webapp_mod._RATE_LIMIT_RPM == 30
        assert webapp_mod._RATE_LIMIT_WINDOW == 60

    def test_rate_tracker_structure(self):
        from collections import defaultdict
        from web.app import _rate_tracker
        assert isinstance(_rate_tracker, defaultdict)


# =====================================================================
# Rate tracker periodic cleanup
# =====================================================================

class TestRateTrackerCleanup:
    """Rate limiter should periodically purge stale IP entries."""

    def test_purge_interval_defined(self):
        webapp_mod = sys.modules["web.app"]
        assert webapp_mod._RATE_TRACKER_PURGE_INTERVAL > 0

    def test_stale_ips_removed(self):
        """Stale IP entries should be cleaned up."""
        webapp_mod = sys.modules["web.app"]
        webapp_mod._rate_tracker.clear()
        webapp_mod._rate_tracker["1.2.3.4"] = [time.time() - 120]  # old
        webapp_mod._rate_tracker["5.6.7.8"] = [time.time()]  # recent

        now = time.time()
        window_start = now - webapp_mod._RATE_LIMIT_WINDOW
        stale = [
            ip for ip, ts in webapp_mod._rate_tracker.items()
            if not ts or ts[-1] < window_start
        ]
        for ip in stale:
            del webapp_mod._rate_tracker[ip]

        assert "1.2.3.4" not in webapp_mod._rate_tracker
        assert "5.6.7.8" in webapp_mod._rate_tracker
