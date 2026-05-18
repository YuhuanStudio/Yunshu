"""Tests for inflight prefix sharing (SGLang cache_unfinished_req pattern)."""

import time
import pytest

from yunshu_engine.inflight_prefix_sharing import (
    InflightEntry,
    InflightPrefixTracker,
    get_inflight_tracker,
)


class TestInflightPrefixTracker:
    def setup_method(self):
        self.tracker = InflightPrefixTracker(max_entries=8, ttl_seconds=5.0)

    def test_register_and_find(self):
        self.tracker.register("req-1", [1, 2, 3, 4, 5], object(), "test-model")
        result = self.tracker.find_prefix([1, 2, 3, 4, 5, 6, 7], "test-model")
        assert result is not None
        assert result.request_id == "req-1"
        assert len(result.token_ids) == 5

    def test_find_no_match(self):
        self.tracker.register("req-1", [10, 20, 30, 40, 50], object(), "test-model")
        result = self.tracker.find_prefix([1, 2, 3, 4, 5], "test-model")
        assert result is None

    def test_find_model_isolation(self):
        self.tracker.register("req-1", [1, 2, 3, 4, 5], object(), "model-a")
        result = self.tracker.find_prefix([1, 2, 3, 4, 5, 6], "model-b")
        assert result is None

    def test_find_same_model_match(self):
        self.tracker.register("req-1", [1, 2, 3, 4, 5], object(), "model-a")
        result = self.tracker.find_prefix([1, 2, 3, 4, 5, 6], "model-a")
        assert result is not None

    def test_unregister(self):
        self.tracker.register("req-1", [1, 2, 3], object(), "model")
        self.tracker.unregister("req-1")
        result = self.tracker.find_prefix([1, 2, 3, 4], "model")
        assert result is None

    def test_unregister_nonexistent(self):
        self.tracker.unregister("nonexistent")

    def test_update_extends_prefix(self):
        self.tracker.register("req-1", [1, 2, 3], object(), "model")
        self.tracker.update("req-1", [1, 2, 3, 4, 5])
        result = self.tracker.find_prefix([1, 2, 3, 4, 5, 6], "model")
        assert result is not None
        assert len(result.token_ids) == 5

    def test_update_nonexistent(self):
        self.tracker.update("nonexistent", [1, 2, 3])

    def test_max_entries_eviction(self):
        tracker = InflightPrefixTracker(max_entries=3)
        for i in range(5):
            tracker.register(f"req-{i}", [i], object(), "model")
        assert len(tracker._entries) == 3
        stats = tracker.get_stats()
        assert stats["evictions"] == 2

    def test_ttl_expiry(self):
        tracker = InflightPrefixTracker(max_entries=8, ttl_seconds=100.0)
        tracker.register("req-1", [1, 2, 3], object(), "model")
        # Simulate time passage by manipulating both timestamps
        old_time = time.monotonic() - 200
        tracker._entries["req-1"].created_at = old_time
        tracker._entries["req-1"].last_updated_at = old_time
        tracker.register("req-2", [4, 5, 6], object(), "model")
        assert "req-1" not in tracker._entries
        stats = tracker.get_stats()
        assert stats["evictions"] == 1

    def test_empty_find(self):
        result = self.tracker.find_prefix([], "model")
        assert result is None

    def test_no_entries_find(self):
        result = self.tracker.find_prefix([1, 2, 3], "model")
        assert result is None

    def test_multiple_entries_best_match(self):
        self.tracker.register("req-1", [1, 2, 3], object(), "model")
        self.tracker.register("req-2", [1, 2, 3, 4, 5], object(), "model")
        result = self.tracker.find_prefix([1, 2, 3, 4, 5, 6, 7], "model")
        assert result is not None
        # Should match req-2 (longer prefix)
        assert len(result.token_ids) >= 5

    def test_stats(self):
        self.tracker.register("req-1", [1, 2, 3], object(), "model")
        self.tracker.find_prefix([1, 2, 3, 4], "model")
        self.tracker.find_prefix([9, 9, 9], "model")
        stats = self.tracker.get_stats()
        assert stats["registrations"] == 1
        assert stats["prefix_hits"] == 1
        assert stats["prefix_misses"] == 1
        assert stats["active_entries"] == 1

    def test_clear(self):
        self.tracker.register("req-1", [1, 2, 3], object(), "model")
        self.tracker.register("req-2", [4, 5, 6], object(), "model")
        self.tracker.clear()
        assert len(self.tracker._entries) == 0
        result = self.tracker.find_prefix([1, 2, 3], "model")
        assert result is None

    def test_long_prefix(self):
        tokens = list(range(1024))
        self.tracker.register("req-1", tokens, object(), "model")
        result = self.tracker.find_prefix(tokens + [9999], "model")
        assert result is not None
        assert len(result.token_ids) == 1024

    def test_single_token_prefix(self):
        self.tracker.register("req-1", [42], object(), "model")
        result = self.tracker.find_prefix([42, 100], "model")
        assert result is not None
        assert result.request_id == "req-1"


class TestGetInflightTracker:
    def test_singleton(self):
        t1 = get_inflight_tracker()
        t2 = get_inflight_tracker()
        assert t1 is t2

    def test_is_correct_type(self):
        assert isinstance(get_inflight_tracker(), InflightPrefixTracker)


class TestInflightEntry:
    def test_fields(self):
        entry = InflightEntry(
            request_id="test",
            token_ids=[1, 2, 3],
            kv_cache_ref="cache",
            model_name="model",
        )
        assert entry.request_id == "test"
        assert entry.token_ids == [1, 2, 3]
        assert entry.kv_cache_ref == "cache"
        assert entry.model_name == "model"
        assert entry.last_update_len == 0
        assert entry.created_at > 0
