"""Tests for request_dedup.py — request deduplication."""

import time

import pytest

from yunshu_engine.request_dedup import (
    DeduplicationEntry,
    RequestDeduplicator,
)


class TestDeduplicationEntry:
    def test_fan_out(self):
        entry = DeduplicationEntry(content_hash="abc", request_ids=["r1", "r2", "r3"])
        assert entry.fan_out == 3

    def test_age_ms(self):
        entry = DeduplicationEntry(content_hash="abc", created_at=time.monotonic() - 1.0)
        assert entry.age_ms >= 900

    def test_is_completed(self):
        entry = DeduplicationEntry(content_hash="abc")
        assert not entry.is_completed
        entry.completed_at = time.monotonic()
        assert entry.is_completed

    def test_default_values(self):
        entry = DeduplicationEntry(content_hash="abc")
        assert entry.fan_out == 0
        assert entry.primary_request_id == ""
        assert entry.model == ""


class TestRequestDeduplicator:
    def test_compute_hash_deterministic(self):
        h1 = RequestDeduplicator.compute_hash("model-a", "hello", temperature=0.7)
        h2 = RequestDeduplicator.compute_hash("model-a", "hello", temperature=0.7)
        assert h1 == h2

    def test_compute_hash_different_inputs(self):
        h1 = RequestDeduplicator.compute_hash("model-a", "hello")
        h2 = RequestDeduplicator.compute_hash("model-b", "hello")
        assert h1 != h2

    def test_compute_hash_different_sampling(self):
        h1 = RequestDeduplicator.compute_hash("model-a", "hello", temperature=0.7)
        h2 = RequestDeduplicator.compute_hash("model-a", "hello", temperature=1.0)
        assert h1 != h2

    def test_compute_hash_token_ids(self):
        h1 = RequestDeduplicator.compute_hash("model-a", [1, 2, 3])
        h2 = RequestDeduplicator.compute_hash("model-a", "hello")
        assert h1 != h2

    def test_register_new_request(self):
        dedup = RequestDeduplicator()
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        entry = dedup.register("r1", h, model="model-a")
        assert entry.primary_request_id == "r1"
        assert entry.fan_out == 1

    def test_deduplicate_matching_request(self):
        dedup = RequestDeduplicator(window_ms=10000.0)
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        result = dedup.check("r2", h)
        assert result is not None
        assert "r2" in result.request_ids
        assert result.fan_out == 2

    def test_no_dedup_different_hash(self):
        dedup = RequestDeduplicator()
        h1 = RequestDeduplicator.compute_hash("model-a", "hello")
        h2 = RequestDeduplicator.compute_hash("model-a", "world")
        dedup.register("r1", h1)
        result = dedup.check("r2", h2)
        assert result is None

    def test_no_dedup_completed_entry(self):
        dedup = RequestDeduplicator()
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        dedup.complete(h)
        result = dedup.check("r2", h)
        assert result is None

    def test_no_dedup_outside_window(self):
        dedup = RequestDeduplicator(window_ms=0.001)
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        time.sleep(0.01)
        result = dedup.check("r2", h)
        assert result is None

    def test_fan_out_limit(self):
        dedup = RequestDeduplicator(max_fan_out=2, window_ms=10000.0)
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        dedup.check("r2", h)  # OK
        result = dedup.check("r3", h)  # Exceeds fan_out
        assert result is None

    def test_complete_returns_all_ids(self):
        dedup = RequestDeduplicator(window_ms=10000.0)
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        dedup.check("r2", h)
        dedup.check("r3", h)
        ids = dedup.complete(h)
        assert len(ids) == 3
        assert "r1" in ids
        assert "r2" in ids
        assert "r3" in ids

    def test_complete_nonexistent(self):
        dedup = RequestDeduplicator()
        ids = dedup.complete("nonexistent")
        assert ids == []

    def test_max_entries_eviction(self):
        dedup = RequestDeduplicator(max_entries=2)
        h1 = RequestDeduplicator.compute_hash("model-a", "a")
        h2 = RequestDeduplicator.compute_hash("model-a", "b")
        h3 = RequestDeduplicator.compute_hash("model-a", "c")
        dedup.register("r1", h1)
        dedup.register("r2", h2)
        assert len(dedup._entries) == 2
        # Complete h1 so it becomes evictable (in-flight entries are
        # protected from eviction to avoid orphaning shadow requests).
        dedup.complete(h1)
        dedup.register("r3", h3)  # should evict oldest completed entry
        assert len(dedup._entries) == 2
        assert dedup.get_entry(h1) is None  # evicted

    def test_ttl_expiry(self):
        dedup = RequestDeduplicator(ttl_seconds=0.01)
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        dedup.complete(h)
        time.sleep(0.02)
        # Pruning happens on next check
        h2 = RequestDeduplicator.compute_hash("model-a", "world")
        dedup.register("r2", h2)  # triggers prune
        assert dedup.get_entry(h) is None  # expired

    def test_get_entry(self):
        dedup = RequestDeduplicator()
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        entry = dedup.get_entry(h)
        assert entry is not None
        assert entry.primary_request_id == "r1"

    def test_stats(self):
        dedup = RequestDeduplicator(window_ms=10000.0)
        h = RequestDeduplicator.compute_hash("model-a", "hello")
        dedup.register("r1", h)
        dedup.check("r2", h)
        dedup.complete(h)
        stats = dedup.get_stats()
        assert stats["total_deduplicated"] == 1
        assert stats["total_saved_requests"] == 1
        assert stats["total_inferences"] == 1
        assert stats["dedup_rate"] == 1.0

    def test_stats_empty(self):
        dedup = RequestDeduplicator()
        stats = dedup.get_stats()
        assert stats["active_entries"] == 0
        assert stats["dedup_rate"] == 0.0

    def test_from_env(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {
            "YUNSHU_DEDUP_WINDOW_MS": "200.0",
            "YUNSHU_DEDUP_MAX_FANOUT": "16",
            "YUNSHU_DEDUP_MAX_ENTRIES": "500",
        }):
            dedup = RequestDeduplicator.from_env()
            assert dedup._window_ms == 200.0
            assert dedup._max_fan_out == 16
            assert dedup._max_entries == 500

    def test_multiple_requests_different_models(self):
        dedup = RequestDeduplicator(window_ms=10000.0)
        h1 = RequestDeduplicator.compute_hash("model-a", "hello")
        h2 = RequestDeduplicator.compute_hash("model-b", "hello")
        dedup.register("r1", h1, model="model-a")
        dedup.register("r2", h2, model="model-b")
        # r3 with same prompt but different model should not dedup
        result = dedup.check("r3", h2)  # same as r2
        assert result is not None
        assert result.fan_out == 2
