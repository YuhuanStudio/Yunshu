"""Tests for SpecPrefillEngine — speculative prefill during decode idle time."""

import time

import pytest

from yunshu_engine.spec_prefill_engine import (
    PrefillEntry,
    SpecPrefillEngine,
    SpecPrefillStats,
)


# ── PrefillEntry ──


class TestPrefillEntry:
    def test_defaults(self):
        entry = PrefillEntry(request_id="r1", tokens=[1, 2, 3])
        assert entry.request_id == "r1"
        assert entry.tokens == [1, 2, 3]
        assert entry.priority == 0.0
        assert entry.tokens_prefilled == 0
        assert entry.kv_state is None
        assert entry.status == "pending"
        assert entry.started_at is None
        assert entry.completed_at is None

    def test_custom_values(self):
        entry = PrefillEntry(
            request_id="r2",
            tokens=[4, 5, 6],
            priority=5.0,
            tokens_prefilled=2,
            status="in_progress",
        )
        assert entry.priority == 5.0
        assert entry.tokens_prefilled == 2
        assert entry.status == "in_progress"


# ── SpecPrefillEngine — enqueue ──


class TestEnqueue:
    def test_enqueue_single(self):
        engine = SpecPrefillEngine()
        result = engine.enqueue("req-1", tokens=[1, 2, 3], priority=1.0)
        assert result is True
        assert engine.queue_size() == 1

    def test_enqueue_multiple(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1], priority=1.0)
        engine.enqueue("req-2", tokens=[2], priority=2.0)
        assert engine.queue_size() == 2

    def test_enqueue_duplicate_rejected(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        result = engine.enqueue("req-1", tokens=[4, 5, 6])
        assert result is False
        assert engine.queue_size() == 1

    def test_enqueue_queue_full(self):
        engine = SpecPrefillEngine(max_queue_size=2)
        engine.enqueue("req-1", tokens=[1])
        engine.enqueue("req-2", tokens=[2])
        result = engine.enqueue("req-3", tokens=[3])
        assert result is False

    def test_enqueue_priority_ordering(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-low", tokens=[1], priority=1.0)
        engine.enqueue("req-high", tokens=[2], priority=10.0)
        engine.enqueue("req-mid", tokens=[3], priority=5.0)

        # Peek should return highest priority
        next_entry = engine.peek_next()
        assert next_entry is not None
        assert next_entry.request_id == "req-high"

    def test_enqueue_fifo_for_same_priority(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1], priority=5.0)
        engine.enqueue("req-2", tokens=[2], priority=5.0)

        next_entry = engine.peek_next()
        assert next_entry is not None
        assert next_entry.request_id == "req-1"  # FIFO

    def test_has_capacity(self):
        engine = SpecPrefillEngine(max_queue_size=2)
        assert engine.has_capacity() is True
        engine.enqueue("req-1", tokens=[1])
        engine.enqueue("req-2", tokens=[2])
        assert engine.has_capacity() is False


# ── SpecPrefillEngine — try_prefill ──


class TestTryPrefill:
    def test_no_queue_returns_none(self):
        engine = SpecPrefillEngine()
        result = engine.try_prefill(available_budget=1024)
        assert result is None

    def test_zero_budget_returns_none(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        result = engine.try_prefill(available_budget=0)
        assert result is None

    def test_negative_budget_returns_none(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        result = engine.try_prefill(available_budget=-1)
        assert result is None

    def test_completes_small_request_in_one_call(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3], priority=1.0)
        result = engine.try_prefill(available_budget=4096)
        assert result is not None
        assert result.status == "completed"
        assert result.tokens_prefilled == 3
        assert result.request_id == "req-1"

    def test_chunked_prefill(self):
        engine = SpecPrefillEngine(prefill_chunk_size=2)
        tokens = list(range(10))
        engine.enqueue("req-1", tokens=tokens, priority=1.0)

        # First call: chunk_size=2, budget=2
        result = engine.try_prefill(available_budget=2)
        assert result is None  # not yet complete
        entry = engine.get_entry("req-1")
        assert entry.status == "in_progress"
        assert entry.tokens_prefilled == 2

        # Continue
        result = engine.try_prefill(available_budget=2)
        assert result is None
        assert entry.tokens_prefilled == 4

    def test_resumes_in_progress_entry(self):
        engine = SpecPrefillEngine(prefill_chunk_size=2)
        tokens = list(range(5))
        engine.enqueue("req-1", tokens=tokens)

        engine.try_prefill(available_budget=2)  # start: 2/5
        engine.try_prefill(available_budget=2)  # continue: 4/5
        result = engine.try_prefill(available_budget=2)  # finish: 5/5
        assert result is not None
        assert result.status == "completed"
        assert result.tokens_prefilled == 5

    def test_clears_current_on_completion(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2])
        engine.try_prefill(available_budget=1024)
        assert engine.current is None

    def test_priority_ordering_respected(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-low", tokens=[1], priority=1.0)
        engine.enqueue("req-high", tokens=[2], priority=10.0)
        result = engine.try_prefill(available_budget=1024)
        assert result is not None
        assert result.request_id == "req-high"

    def test_second_prefill_picks_next(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1], priority=1.0)
        engine.enqueue("req-2", tokens=[2], priority=2.0)

        r1 = engine.try_prefill(available_budget=1024)
        assert r1.request_id == "req-2"  # higher priority first

        r2 = engine.try_prefill(available_budget=1024)
        assert r2.request_id == "req-1"  # next highest


# ── SpecPrefillEngine — cancel ──


class TestCancelPrefill:
    def test_cancel_pending(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        assert engine.queue_size() == 1
        result = engine.cancel_prefill("req-1")
        assert result is True
        # After cancel, entry is removed from the engine (allows re-enqueue)
        assert engine.get_entry("req-1") is None
        assert engine.queue_size() == 0

    def test_cancel_in_progress(self):
        engine = SpecPrefillEngine(prefill_chunk_size=2)
        engine.enqueue("req-1", tokens=list(range(10)))
        engine.try_prefill(available_budget=2)
        assert engine.current is not None

        result = engine.cancel_prefill("req-1")
        assert result is True
        assert engine.current is None

    def test_cancel_nonexistent(self):
        engine = SpecPrefillEngine()
        result = engine.cancel_prefill("no-such-req")
        assert result is False

    def test_cancel_already_completed(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2])
        engine.try_prefill(available_budget=1024)  # completes
        result = engine.cancel_prefill("req-1")
        assert result is False

    def test_cancel_twice(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        assert engine.cancel_prefill("req-1") is True
        assert engine.cancel_prefill("req-1") is False

    def test_cancel_removes_from_queue(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1])
        engine.enqueue("req-2", tokens=[2])
        engine.cancel_prefill("req-1")
        assert engine.queue_size() == 1

    def test_cancel_allows_new_enqueue(self):
        engine = SpecPrefillEngine(max_queue_size=1)
        engine.enqueue("req-1", tokens=[1])
        engine.cancel_prefill("req-1")
        result = engine.enqueue("req-1", tokens=[1])
        assert result is True  # re-enqueue allowed after cancel


# ── SpecPrefillEngine — remove_entry ──


class TestRemoveEntry:
    def test_remove_existing(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        entry = engine.remove_entry("req-1")
        assert entry is not None
        assert entry.request_id == "req-1"
        assert engine.get_entry("req-1") is None

    def test_remove_nonexistent(self):
        engine = SpecPrefillEngine()
        entry = engine.remove_entry("no-such")
        assert entry is None


# ── SpecPrefillEngine — stats ──


class TestGetStats:
    def test_initial_stats(self):
        engine = SpecPrefillEngine()
        stats = engine.get_stats()
        assert stats["prefills_completed"] == 0
        assert stats["prefills_cancelled"] == 0
        assert stats["tokens_prefilled"] == 0
        assert stats["budget_utilization"] == 0.0
        assert stats["queue_size"] == 0
        assert stats["current_request"] is None

    def test_stats_after_completion(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        engine.try_prefill(available_budget=1024)
        stats = engine.get_stats()
        assert stats["prefills_completed"] == 1
        assert stats["tokens_prefilled"] == 3
        assert stats["queue_size"] == 0

    def test_stats_after_cancellation(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        engine.cancel_prefill("req-1")
        stats = engine.get_stats()
        assert stats["prefills_cancelled"] == 1

    def test_budget_utilization(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        engine.try_prefill(available_budget=1024)
        stats = engine.get_stats()
        # 3 tokens consumed out of 1024 budget requested
        assert stats["budget_consumed"] == 3
        assert stats["budget_requested"] == 1024
        assert stats["budget_utilization"] == pytest.approx(3.0 / 1024, abs=0.001)

    def test_stats_tracks_current(self):
        engine = SpecPrefillEngine(prefill_chunk_size=2)
        engine.enqueue("req-1", tokens=list(range(10)))
        engine.try_prefill(available_budget=2)
        stats = engine.get_stats()
        assert stats["current_request"] == "req-1"

    def test_avg_prefill_time(self):
        engine = SpecPrefillEngine()
        engine.enqueue("req-1", tokens=[1, 2, 3])
        engine.try_prefill(available_budget=1024)
        stats = engine.get_stats()
        assert stats["avg_prefill_time_ms"] >= 0.0
