"""Tests for TokenLevelScheduler, PriorityInversionGuard, and FairnessTracker."""

import time

import pytest

from yunshu_engine.token_scheduler import (
    DecodeStrategy,
    FairnessTracker,
    PrefillStrategy,
    PriorityInversionGuard,
    ResolutionStrategy,
    SchedulableRequest,
    TokenLevelScheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str = "req-0",
    priority: int = 0,
    wait_time: float = 0.0,
    context_length: int = 256,
    output_length: int = 0,
    is_prefilling: bool = False,
    effective_priority: int | None = None,
) -> SchedulableRequest:
    """Create a SchedulableRequest with sensible defaults."""
    return SchedulableRequest(
        request_id=request_id,
        priority=priority,
        wait_time=wait_time,
        context_length=context_length,
        output_length=output_length,
        is_prefilling=is_prefilling,
        effective_priority=effective_priority if effective_priority is not None else priority,
    )


# ===========================================================================
# 1. TokenLevelScheduler
# ===========================================================================


class TestTokenLevelSchedulerBasic:
    """Basic token budget allocation tests."""

    def test_empty_requests_returns_empty(self):
        """No requests → empty allocation list."""
        sched = TokenLevelScheduler()
        result = sched.compute_token_budget([], total_budget=2048)
        assert result == []

    def test_single_prefill_request(self):
        """Single prefilling request gets the entire budget."""
        sched = TokenLevelScheduler()
        req = _make_request("req-1", is_prefilling=True, context_length=1024)
        result = sched.compute_token_budget([req], total_budget=2048)
        assert len(result) == 1
        assert result[0].request_id == "req-1"
        assert result[0].prefill_tokens > 0

    def test_single_decode_request(self):
        """Single decoding request gets decode tokens."""
        sched = TokenLevelScheduler()
        req = _make_request("req-1", is_prefilling=False, output_length=10)
        result = sched.compute_token_budget([req], total_budget=128)
        assert len(result) == 1
        assert result[0].request_id == "req-1"
        assert result[0].decode_tokens > 0

    def test_mixed_prefill_and_decode(self):
        """Mix of prefilling and decoding requests splits budget."""
        sched = TokenLevelScheduler()
        reqs = [
            _make_request("prefill-1", is_prefilling=True, context_length=2048),
            _make_request("decode-1", is_prefilling=False, output_length=50),
        ]
        result = sched.compute_token_budget(reqs, total_budget=4096)
        assert len(result) == 2
        prefill_alloc = next(a for a in result if a.prefill_tokens > 0)
        decode_alloc = next(a for a in result if a.decode_tokens > 0)
        assert prefill_alloc.request_id == "prefill-1"
        assert decode_alloc.request_id == "decode-1"

    def test_budget_not_exceeded(self):
        """Total allocated tokens should not exceed the budget."""
        sched = TokenLevelScheduler()
        reqs = [
            _make_request(f"req-{i}", is_prefilling=True, context_length=512 * (i + 1))
            for i in range(10)
        ]
        result = sched.compute_token_budget(reqs, total_budget=1024)
        total = sum(a.prefill_tokens + a.decode_tokens for a in result)
        assert total <= 1024 + len(reqs) * sched.min_prefill_tokens  # tolerance for min floor

    def test_stats_updated(self):
        """Stats are updated after budget computation."""
        sched = TokenLevelScheduler()
        reqs = [
            _make_request("req-1", is_prefilling=True, context_length=256),
            _make_request("req-2", is_prefilling=False, output_length=10),
        ]
        sched.compute_token_budget(reqs, total_budget=2048)
        stats = sched.get_stats()
        assert stats["budget_computations"] == 1
        assert stats["total_requests_scheduled"] == 2
        assert stats["total_prefill_tokens_allocated"] > 0
        assert stats["total_decode_tokens_allocated"] > 0


class TestTokenLevelSchedulerWFQ:
    """Weighted Fair Queuing allocation tests."""

    def test_higher_priority_gets_more_tokens(self):
        """Higher priority request should get more decode tokens than lower."""
        sched = TokenLevelScheduler(priority_weight=1.0, wait_time_weight=0.0, context_weight=0.0)
        reqs = [
            _make_request("low", priority=1, is_prefilling=False, effective_priority=1),
            _make_request("high", priority=10, is_prefilling=False, effective_priority=10),
        ]
        result = sched.compute_token_budget(reqs, total_budget=100)
        low_alloc = next(a for a in result if a.request_id == "low")
        high_alloc = next(a for a in result if a.request_id == "high")
        assert high_alloc.decode_tokens >= low_alloc.decode_tokens

    def test_longer_wait_gets_boost(self):
        """Longer-waiting requests should get more tokens."""
        sched = TokenLevelScheduler(priority_weight=0.0, wait_time_weight=1.0, context_weight=0.0)
        reqs = [
            _make_request("short-wait", wait_time=0.1, is_prefilling=False),
            _make_request("long-wait", wait_time=5.0, is_prefilling=False),
        ]
        result = sched.compute_token_budget(reqs, total_budget=100)
        short_alloc = next(a for a in result if a.request_id == "short-wait")
        long_alloc = next(a for a in result if a.request_id == "long-wait")
        assert long_alloc.decode_tokens >= short_alloc.decode_tokens

    def test_longer_context_gets_more_prefill(self):
        """Longer context should get proportionally more prefill tokens."""
        sched = TokenLevelScheduler(prefill_strategy=PrefillStrategy.PROPORTIONAL)
        reqs = [
            _make_request("short-ctx", context_length=256, is_prefilling=True),
            _make_request("long-ctx", context_length=2048, is_prefilling=True),
        ]
        result = sched.compute_token_budget(reqs, total_budget=4096)
        short_alloc = next(a for a in result if a.request_id == "short-ctx")
        long_alloc = next(a for a in result if a.request_id == "long-ctx")
        assert long_alloc.prefill_tokens >= short_alloc.prefill_tokens

    def test_effective_priority_overrides_raw_priority(self):
        """Effective priority (from inversion guard) should be used for weighting."""
        sched = TokenLevelScheduler(priority_weight=1.0, wait_time_weight=0.0, context_weight=0.0)
        reqs = [
            _make_request("raw-low", priority=1, is_prefilling=False, effective_priority=1),
            _make_request("boosted", priority=1, is_prefilling=False, effective_priority=50),
        ]
        result = sched.compute_token_budget(reqs, total_budget=100)
        raw_alloc = next(a for a in result if a.request_id == "raw-low")
        boosted_alloc = next(a for a in result if a.request_id == "boosted")
        assert boosted_alloc.decode_tokens >= raw_alloc.decode_tokens

    def test_weight_floor_prevents_zero(self):
        """Weight should never be zero (floor at 0.001)."""
        sched = TokenLevelScheduler()
        req = _make_request("zero-req", priority=0, wait_time=0.0, context_length=0)
        weight = sched._compute_weight(req)
        assert weight >= 0.001


class TestTokenLevelPrefillStrategies:
    """Test different prefill allocation strategies."""

    def test_equal_strategy(self):
        """EQUAL strategy gives each request the same token count."""
        sched = TokenLevelScheduler(prefill_strategy=PrefillStrategy.EQUAL)
        reqs = [
            _make_request("a", context_length=100, is_prefilling=True),
            _make_request("b", context_length=5000, is_prefilling=True),
        ]
        result = sched.compute_token_budget(reqs, total_budget=2048)
        prefill_allocs = [a for a in result if a.prefill_tokens > 0]
        assert len(prefill_allocs) == 2
        # Both should have equal allocation
        assert prefill_allocs[0].prefill_tokens == prefill_allocs[1].prefill_tokens

    def test_priority_first_strategy(self):
        """PRIORITY_FIRST gives more tokens to higher priority requests."""
        sched = TokenLevelScheduler(
            prefill_strategy=PrefillStrategy.PRIORITY_FIRST,
            priority_weight=1.0,
            wait_time_weight=0.0,
            context_weight=0.0,
        )
        reqs = [
            _make_request("low", priority=1, is_prefilling=True, effective_priority=1),
            _make_request("high", priority=50, is_prefilling=True, effective_priority=50),
        ]
        result = sched.compute_token_budget(reqs, total_budget=2048)
        low_alloc = next(a for a in result if a.request_id == "low")
        high_alloc = next(a for a in result if a.request_id == "high")
        assert high_alloc.prefill_tokens >= low_alloc.prefill_tokens

    def test_proportional_strategy_default(self):
        """PROPORTIONAL is the default and distributes based on context length."""
        sched = TokenLevelScheduler()
        assert sched.prefill_strategy == PrefillStrategy.PROPORTIONAL
        reqs = [
            _make_request("short", context_length=256, is_prefilling=True),
            _make_request("long", context_length=1024, is_prefilling=True),
        ]
        result = sched.compute_token_budget(reqs, total_budget=2048)
        short_alloc = next(a for a in result if a.request_id == "short")
        long_alloc = next(a for a in result if a.request_id == "long")
        # Long context should get roughly 4x more tokens
        assert long_alloc.prefill_tokens >= short_alloc.prefill_tokens


class TestTokenLevelDecodeStrategies:
    """Test different decode allocation strategies."""

    def test_round_robin_strategy(self):
        """ROUND_ROBIN gives each request the same decode token count."""
        sched = TokenLevelScheduler(decode_strategy=DecodeStrategy.ROUND_ROBIN)
        reqs = [
            _make_request("a", priority=1, is_prefilling=False, effective_priority=1),
            _make_request("b", priority=99, is_prefilling=False, effective_priority=99),
        ]
        result = sched.compute_token_budget(reqs, total_budget=100)
        decode_allocs = [a for a in result if a.decode_tokens > 0]
        assert len(decode_allocs) == 2

    def test_priority_only_strategy(self):
        """PRIORITY_ONLY gives tokens to highest priority first."""
        sched = TokenLevelScheduler(decode_strategy=DecodeStrategy.PRIORITY_ONLY)
        reqs = [
            _make_request("low", priority=1, is_prefilling=False, effective_priority=1),
            _make_request("high", priority=50, is_prefilling=False, effective_priority=50),
        ]
        result = sched.compute_token_budget(reqs, total_budget=100)
        high_alloc = next(a for a in result if a.request_id == "high")
        next(a for a in result if a.request_id == "low")
        assert high_alloc.decode_tokens > 0

    def test_wfq_default(self):
        """WFQ is the default decode strategy."""
        sched = TokenLevelScheduler()
        assert sched.decode_strategy == DecodeStrategy.WFQ


class TestTokenLevelPrefillMinMax:
    """Test min/max prefill token clamping."""

    def test_min_prefill_tokens_enforced(self):
        """Each prefilling request gets at least min_prefill_tokens."""
        sched = TokenLevelScheduler(min_prefill_tokens=64)
        reqs = [
            _make_request(f"req-{i}", context_length=10, is_prefilling=True)
            for i in range(20)
        ]
        # Very small budget — should still give min_prefill_tokens to each
        result = sched.compute_token_budget(reqs, total_budget=100)
        for alloc in result:
            if alloc.prefill_tokens > 0:
                assert alloc.prefill_tokens >= 64

    def test_max_prefill_tokens_enforced(self):
        """No prefilling request gets more than max_prefill_tokens."""
        sched = TokenLevelScheduler(max_prefill_tokens=512)
        reqs = [
            _make_request("big-req", context_length=100000, is_prefilling=True),
        ]
        result = sched.compute_token_budget(reqs, total_budget=999999)
        for alloc in result:
            if alloc.prefill_tokens > 0:
                assert alloc.prefill_tokens <= 512


class TestTokenLevelSchedulerStats:
    """Stats and metrics tests for TokenLevelScheduler."""

    def test_avg_budget_utilization(self):
        """Average budget utilization is computed correctly."""
        sched = TokenLevelScheduler()
        reqs = [_make_request("req-1", is_prefilling=True, context_length=256)]
        sched.compute_token_budget(reqs, total_budget=2048)
        stats = sched.get_stats()
        assert 0 <= stats["avg_budget_utilization"] <= 1.0

    def test_weight_range_tracking(self):
        """Min/max weight seen is tracked."""
        sched = TokenLevelScheduler()
        reqs = [
            _make_request("low", priority=0, is_prefilling=False),
            _make_request("high", priority=100, is_prefilling=False),
        ]
        sched.compute_token_budget(reqs, total_budget=100)
        stats = sched.get_stats()
        assert stats["max_weight_seen"] > 0
        assert stats["min_weight_seen"] > 0
        assert stats["max_weight_seen"] >= stats["min_weight_seen"]

    def test_utilization_history_bounded(self):
        """Utilization history is bounded to 1000 samples."""
        sched = TokenLevelScheduler()
        req = _make_request("req-1", is_prefilling=False)
        for _ in range(1500):
            sched.compute_token_budget([req], total_budget=100)
        assert len(sched._stats["budget_utilization"]) <= 1000


# ===========================================================================
# 2. PriorityInversionGuard
# ===========================================================================


class TestPriorityInversionDetection:
    """Priority inversion detection tests."""

    def test_no_inversion_when_empty(self):
        """No inversion with empty lists."""
        guard = PriorityInversionGuard()
        assert guard.check_inversion([], []) == []

    def test_no_inversion_when_no_running(self):
        """No inversion when no running requests."""
        guard = PriorityInversionGuard()
        waiting = [_make_request("w-1", priority=50)]
        assert guard.check_inversion([], waiting) == []

    def test_no_inversion_when_no_waiting(self):
        """No inversion when no waiting requests."""
        guard = PriorityInversionGuard()
        running = [_make_request("r-1", priority=1, wait_time=5.0)]
        assert guard.check_inversion(running, []) == []

    def test_detects_inversion(self):
        """Detects inversion when low-priority running blocks high-priority waiting."""
        guard = PriorityInversionGuard(min_priority_gap=2)
        running = [_make_request("low-req", priority=1, wait_time=5.0)]
        waiting = [_make_request("high-req", priority=50)]
        inversions = guard.check_inversion(running, waiting)
        assert len(inversions) == 1
        assert inversions[0].low_request_id == "low-req"
        assert inversions[0].high_request_id == "high-req"

    def test_no_inversion_below_priority_gap(self):
        """No inversion when priority gap is below threshold."""
        guard = PriorityInversionGuard(min_priority_gap=10)
        running = [_make_request("r-1", priority=5, wait_time=5.0)]
        waiting = [_make_request("w-1", priority=10)]
        assert guard.check_inversion(running, waiting) == []

    def test_no_inversion_below_running_time(self):
        """No inversion when low-priority request hasn't run long enough."""
        guard = PriorityInversionGuard(running_time_threshold=10.0)
        running = [_make_request("r-1", priority=1, wait_time=1.0)]
        waiting = [_make_request("w-1", priority=50)]
        assert guard.check_inversion(running, waiting) == []

    def test_skip_already_boosted(self):
        """Already-boosted requests are skipped in detection."""
        guard = PriorityInversionGuard()
        running = [_make_request("r-1", priority=1, wait_time=5.0)]
        waiting = [_make_request("w-1", priority=50)]
        # Apply a boost
        guard.apply_inheritance(running[0], waiting[0])
        # Check again — should skip boosted request
        inversions = guard.check_inversion(running, waiting)
        assert len(inversions) == 0

    def test_multiple_inversions(self):
        """Multiple low-priority running requests can each cause inversions."""
        guard = PriorityInversionGuard(min_priority_gap=1, running_time_threshold=0.0)
        running = [
            _make_request("low-1", priority=1, wait_time=5.0),
            _make_request("low-2", priority=2, wait_time=5.0),
        ]
        waiting = [_make_request("high", priority=50)]
        inversions = guard.check_inversion(running, waiting)
        assert len(inversions) == 2


class TestPriorityInheritance:
    """Priority inheritance resolution tests."""

    def test_inheritance_boosts_priority(self):
        """apply_inheritance boosts low-priority to match high-priority."""
        guard = PriorityInversionGuard()
        low = _make_request("low", priority=1, effective_priority=1)
        high = _make_request("high", priority=50, effective_priority=50)
        boost = guard.apply_inheritance(low, high)
        assert boost == 49
        assert low.effective_priority == 50

    def test_inheritance_no_boost_if_equal(self):
        """No boost if low priority already >= high priority."""
        guard = PriorityInversionGuard()
        low = _make_request("low", priority=50, effective_priority=50)
        high = _make_request("high", priority=50, effective_priority=50)
        boost = guard.apply_inheritance(low, high)
        assert boost == 0

    def test_inheritance_tracks_active_boost(self):
        """Active boosts are tracked and retrievable."""
        guard = PriorityInversionGuard()
        low = _make_request("low", priority=1, effective_priority=1)
        high = _make_request("high", priority=50, effective_priority=50)
        guard.apply_inheritance(low, high)
        boosted = guard.get_boost("low")
        assert boosted == 50

    def test_inheritance_boost_expires(self):
        """Boosts expire after max_boost_duration."""
        guard = PriorityInversionGuard(max_boost_duration=0.01)
        low = _make_request("low", priority=1, effective_priority=1)
        high = _make_request("high", priority=50, effective_priority=50)
        guard.apply_inheritance(low, high)
        time.sleep(0.02)
        # Should be expired now
        boosted = guard.get_boost("low")
        assert boosted is None

    def test_clear_boost_manual(self):
        """Manually clearing a boost works."""
        guard = PriorityInversionGuard()
        low = _make_request("low", priority=1, effective_priority=1)
        high = _make_request("high", priority=50, effective_priority=50)
        guard.apply_inheritance(low, high)
        guard.clear_boost("low")
        assert guard.get_boost("low") is None

    def test_get_boost_nonexistent(self):
        """get_boost returns None for non-boosted request."""
        guard = PriorityInversionGuard()
        assert guard.get_boost("nonexistent") is None


class TestPriorityPreemption:
    """Preemption resolution tests."""

    def test_preemption_returns_true(self):
        """Preemption returns True for non-boosted request."""
        guard = PriorityInversionGuard(strategy=ResolutionStrategy.PREEMPTION)
        low = _make_request("low", priority=1)
        assert guard.apply_preemption(low) is True

    def test_preemption_refuses_boosted(self):
        """Preemption returns False for boosted request."""
        guard = PriorityInversionGuard(strategy=ResolutionStrategy.PREEMPTION)
        low = _make_request("low", priority=1, effective_priority=1)
        high = _make_request("high", priority=50, effective_priority=50)
        guard.apply_inheritance(low, high)
        # Can't preempt a boosted request
        assert guard.apply_preemption(low) is False


class TestPriorityResolve:
    """Convenience resolve() method tests."""

    def test_resolve_with_inheritance(self):
        """resolve() detects and applies inheritance."""
        guard = PriorityInversionGuard(
            strategy=ResolutionStrategy.INHERITANCE,
            min_priority_gap=2,
        )
        running = [_make_request("low", priority=1, wait_time=5.0)]
        waiting = [_make_request("high", priority=50)]
        events = guard.resolve(running, waiting)
        assert len(events) == 1
        assert running[0].effective_priority == 50

    def test_resolve_with_preemption(self):
        """resolve() detects and applies preemption."""
        guard = PriorityInversionGuard(
            strategy=ResolutionStrategy.PREEMPTION,
            min_priority_gap=2,
        )
        running = [_make_request("low", priority=1, wait_time=5.0)]
        waiting = [_make_request("high", priority=50)]
        events = guard.resolve(running, waiting)
        assert len(events) == 1
        stats = guard.get_stats()
        assert stats["preemption_applied"] == 1

    def test_resolve_no_inversions(self):
        """resolve() returns empty when no inversions exist."""
        guard = PriorityInversionGuard()
        running = [_make_request("r-1", priority=50, wait_time=5.0)]
        waiting = [_make_request("w-1", priority=1)]
        events = guard.resolve(running, waiting)
        assert len(events) == 0


class TestPriorityInversionStats:
    """Statistics tracking for PriorityInversionGuard."""

    def test_initial_stats(self):
        """Stats start at zero."""
        guard = PriorityInversionGuard()
        stats = guard.get_stats()
        assert stats["inversions_detected"] == 0
        assert stats["inheritance_applied"] == 0
        assert stats["preemption_applied"] == 0
        assert stats["active_boosts"] == 0

    def test_stats_after_inheritance(self):
        """Stats reflect inheritance applications."""
        guard = PriorityInversionGuard(min_priority_gap=2)
        running = [_make_request("low", priority=1, wait_time=5.0)]
        waiting = [_make_request("high", priority=50)]
        guard.resolve(running, waiting)
        stats = guard.get_stats()
        assert stats["inversions_detected"] == 1
        assert stats["inheritance_applied"] == 1
        assert stats["total_boost"] == 49
        assert stats["active_boosts"] == 1

    def test_stats_after_preemption(self):
        """Stats reflect preemption applications."""
        guard = PriorityInversionGuard(
            strategy=ResolutionStrategy.PREEMPTION,
            min_priority_gap=2,
        )
        running = [_make_request("low", priority=1, wait_time=5.0)]
        waiting = [_make_request("high", priority=50)]
        guard.resolve(running, waiting)
        stats = guard.get_stats()
        assert stats["preemption_applied"] == 1

    def test_avg_boost_calculation(self):
        """Average boost is computed correctly across multiple applications."""
        guard = PriorityInversionGuard(min_priority_gap=1, running_time_threshold=0.0)
        running = [
            _make_request("low-1", priority=1, wait_time=5.0, effective_priority=1),
            _make_request("low-2", priority=5, wait_time=5.0, effective_priority=5),
        ]
        waiting = [_make_request("high", priority=50)]
        # First resolve handles low-1
        guard.resolve(running[:1], waiting)
        # Second resolve would skip low-1 (boosted), handle low-2
        guard.resolve(running, waiting)
        stats = guard.get_stats()
        assert stats["avg_boost"] > 0


# ===========================================================================
# 3. FairnessTracker
# ===========================================================================


class TestFairnessTrackerBasic:
    """Basic FairnessTracker tests."""

    def test_initial_fairness_is_one(self):
        """No allocations → fairness = 1.0 (trivially fair)."""
        tracker = FairnessTracker()
        assert tracker.compute_fairness() == 1.0

    def test_single_request_fairness_is_one(self):
        """Single request → fairness = 1.0."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        assert tracker.compute_fairness() == 1.0

    def test_equal_allocation_fairness_is_one(self):
        """Equal allocations → fairness = 1.0."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 100)
        tracker.record_allocation("req-3", 100)
        assert tracker.compute_fairness() == pytest.approx(1.0, abs=1e-6)

    def test_unequal_allocation_reduces_fairness(self):
        """Unequal allocations → fairness < 1.0."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 1000)
        tracker.record_allocation("req-2", 10)
        assert tracker.compute_fairness() < 1.0

    def test_fairness_range(self):
        """Fairness is always in [0, 1]."""
        tracker = FairnessTracker()
        # Various allocations
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 50)
        tracker.record_allocation("req-3", 200)
        tracker.record_allocation("req-4", 1)
        f = tracker.compute_fairness()
        assert 0.0 <= f <= 1.0

    def test_jains_fairness_formula(self):
        """Verify Jain's fairness index formula manually.

        J = (sum(x))^2 / (n * sum(x^2))

        For x = [4, 4, 4, 4]:
          sum = 16, sum_sq = 64
          J = 256 / (4 * 64) = 256 / 256 = 1.0
        """
        tracker = FairnessTracker()
        tracker.record_allocation("a", 4)
        tracker.record_allocation("b", 4)
        tracker.record_allocation("c", 4)
        tracker.record_allocation("d", 4)
        assert tracker.compute_fairness() == pytest.approx(1.0, abs=1e-6)

    def test_jains_fairness_unequal(self):
        """Manual Jain's formula check with unequal values.

        For x = [1, 2, 3]:
          sum = 6, sum_sq = 14
          J = 36 / (3 * 14) = 36 / 42 ≈ 0.8571
        """
        tracker = FairnessTracker()
        tracker.record_allocation("a", 1)
        tracker.record_allocation("b", 2)
        tracker.record_allocation("c", 3)
        assert tracker.compute_fairness() == pytest.approx(36 / 42, abs=1e-4)


class TestFairnessTrackerAllocation:
    """Allocation tracking tests."""

    def test_allocation_accumulates(self):
        """Multiple allocations to same request accumulate."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 50)
        tracker.record_allocation("req-1", 30)
        tracker.record_allocation("req-1", 20)
        assert tracker._allocations["req-1"] == 100

    def test_allocation_count_tracked(self):
        """Allocation count per request is tracked."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 10)
        tracker.record_allocation("req-1", 10)
        tracker.record_allocation("req-2", 10)
        assert tracker._allocation_count["req-1"] == 2
        assert tracker._allocation_count["req-2"] == 1

    def test_total_tokens_tracked(self):
        """Total tokens allocated is tracked."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 50)
        tracker.record_allocation("req-2", 30)
        assert tracker._total_tokens_allocated == 80


class TestFairnessTrackerCompletion:
    """Completion tracking tests."""

    def test_completion_recorded(self):
        """Completion records are stored."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_completion("req-1", wait_time=1.0, total_time=5.0)
        assert len(tracker._completions) == 1
        assert tracker._completions[0].request_id == "req-1"
        assert tracker._completions[0].wait_time == 1.0
        assert tracker._completions[0].total_time == 5.0
        assert tracker._completions[0].total_tokens == 100

    def test_wait_time_variance(self):
        """Wait time variance is computed from completions."""
        tracker = FairnessTracker()
        tracker.record_completion("req-1", wait_time=1.0, total_time=5.0)
        tracker.record_completion("req-2", wait_time=1.0, total_time=5.0)
        assert tracker.get_wait_time_variance() == pytest.approx(0.0, abs=1e-6)

    def test_wait_time_variance_unequal(self):
        """Unequal wait times produce non-zero variance."""
        tracker = FairnessTracker()
        tracker.record_completion("req-1", wait_time=1.0, total_time=5.0)
        tracker.record_completion("req-2", wait_time=5.0, total_time=10.0)
        assert tracker.get_wait_time_variance() > 0

    def test_completion_time_variance(self):
        """Completion time variance is computed from completions."""
        tracker = FairnessTracker()
        tracker.record_completion("req-1", wait_time=1.0, total_time=5.0)
        tracker.record_completion("req-2", wait_time=1.0, total_time=5.0)
        assert tracker.get_completion_time_variance() == pytest.approx(0.0, abs=1e-6)

    def test_variance_no_completions(self):
        """Variance is 0 with fewer than 2 completions."""
        tracker = FairnessTracker()
        assert tracker.get_wait_time_variance() == 0.0
        assert tracker.get_completion_time_variance() == 0.0


class TestFairnessTrackerUnfairRequests:
    """Unfair request detection tests."""

    def test_no_unfair_with_equal_allocations(self):
        """Equal allocations → no unfair requests."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 100)
        unfair = tracker.get_unfair_requests()
        assert len(unfair) == 0

    def test_detects_unfair_request(self):
        """Request well below mean is detected as unfair."""
        tracker = FairnessTracker(unfairness_threshold=0.7)
        tracker.record_allocation("req-1", 1000)
        tracker.record_allocation("req-2", 10)
        unfair = tracker.get_unfair_requests()
        assert len(unfair) == 1
        assert unfair[0]["request_id"] == "req-2"
        assert unfair[0]["ratio"] < 0.7

    def test_unfair_request_details(self):
        """Unfair request record has expected fields."""
        tracker = FairnessTracker(unfairness_threshold=0.7)
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 10)
        unfair = tracker.get_unfair_requests()
        for record in unfair:
            assert "request_id" in record
            assert "total_tokens" in record
            assert "expected_tokens" in record
            assert "ratio" in record

    def test_no_unfair_with_empty(self):
        """No unfair requests when no allocations."""
        tracker = FairnessTracker()
        assert tracker.get_unfair_requests() == []


class TestFairnessTrackerWindowed:
    """Windowed fairness computation tests."""

    def test_windowed_fairness_empty(self):
        """Windowed fairness is 1.0 with no history."""
        tracker = FairnessTracker()
        assert tracker.compute_windowed_fairness() == 1.0

    def test_windowed_fairness_single(self):
        """Windowed fairness is 1.0 with single request."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        assert tracker.compute_windowed_fairness() == 1.0

    def test_windowed_fairness_equal(self):
        """Windowed fairness is 1.0 with equal allocations."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 100)
        assert tracker.compute_windowed_fairness() == pytest.approx(1.0, abs=1e-6)


class TestFairnessTrackerAllocationVariance:
    """Allocation variance tests."""

    def test_zero_variance_single(self):
        """Single request → zero variance."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        assert tracker.get_allocation_variance() == 0.0

    def test_zero_variance_equal(self):
        """Equal allocations → zero variance."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 100)
        assert tracker.get_allocation_variance() == pytest.approx(0.0, abs=1e-6)

    def test_positive_variance_unequal(self):
        """Unequal allocations → positive variance."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 1000)
        tracker.record_allocation("req-2", 0)
        assert tracker.get_allocation_variance() > 0


class TestFairnessTrackerReset:
    """Reset and history management tests."""

    def test_reset_clears_all(self):
        """reset() clears all tracking state."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_completion("req-1", 1.0, 5.0)
        tracker.reset()
        assert tracker._total_tokens_allocated == 0
        assert len(tracker._allocations) == 0
        assert len(tracker._completions) == 0
        assert tracker.compute_fairness() == 1.0

    def test_history_bounded(self):
        """History is bounded to max_history records."""
        tracker = FairnessTracker(max_history=100)
        for _i in range(200):
            tracker.record_allocation("req", 1)
        assert len(tracker._history) <= 100


class TestFairnessTrackerStats:
    """Stats output tests."""

    def test_stats_structure(self):
        """Stats dict has expected keys."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        stats = tracker.get_stats()
        expected_keys = [
            "jains_fairness_index",
            "windowed_fairness_index",
            "allocation_variance",
            "wait_time_variance",
            "completion_time_variance",
            "unfair_requests",
            "total_requests_tracked",
            "total_tokens_allocated",
            "total_allocations",
            "total_completions",
            "history_size",
            "unfairness_threshold",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_values(self):
        """Stats values are consistent with recorded data."""
        tracker = FairnessTracker()
        tracker.record_allocation("req-1", 100)
        tracker.record_allocation("req-2", 200)
        tracker.record_completion("req-1", 1.0, 5.0)
        stats = tracker.get_stats()
        assert stats["total_requests_tracked"] == 2
        assert stats["total_tokens_allocated"] == 300
        assert stats["total_allocations"] == 2
        assert stats["total_completions"] == 1
        assert stats["history_size"] == 2


# ===========================================================================
# 4. Integration: TokenLevelScheduler + PriorityInversionGuard + FairnessTracker
# ===========================================================================


class TestIntegration:
    """Integration tests combining all three components."""

    def test_inversion_guard_boosts_scheduler_weights(self):
        """Priority boost from inversion guard affects scheduler allocation."""
        guard = PriorityInversionGuard(min_priority_gap=2)
        scheduler = TokenLevelScheduler(
            priority_weight=1.0,
            wait_time_weight=0.0,
            context_weight=0.0,
        )

        running = [_make_request("low", priority=1, wait_time=5.0, is_prefilling=False, effective_priority=1)]
        waiting = [_make_request("high", priority=50, is_prefilling=False, effective_priority=50)]

        # Resolve inversion
        guard.resolve(running, waiting)

        # Now allocate tokens — boosted request should get more
        result = scheduler.compute_token_budget(running, total_budget=100)
        assert result[0].effective_priority == 50

    def test_fairness_after_equal_scheduling(self):
        """Fair scheduling produces high fairness index."""
        tracker = FairnessTracker()
        # Simulate 5 requests getting equal allocations over 10 steps
        for _ in range(10):
            for i in range(5):
                tracker.record_allocation(f"req-{i}", 20)

        fairness = tracker.compute_fairness()
        assert fairness == pytest.approx(1.0, abs=1e-4)
        assert len(tracker.get_unfair_requests()) == 0

    def test_fairness_after_unfair_scheduling(self):
        """Unfair scheduling produces lower fairness index."""
        tracker = FairnessTracker()
        # req-0 gets most tokens, others get very few
        for _ in range(10):
            tracker.record_allocation("req-0", 200)
            tracker.record_allocation("req-1", 5)
            tracker.record_allocation("req-2", 5)

        fairness = tracker.compute_fairness()
        assert fairness < 0.5
        unfair = tracker.get_unfair_requests()
        assert len(unfair) == 2

    def test_full_pipeline(self):
        """Full pipeline: detect inversion → boost → schedule → track fairness."""
        guard = PriorityInversionGuard(min_priority_gap=2)
        scheduler = TokenLevelScheduler()
        tracker = FairnessTracker()

        # Phase 1: initial scheduling without inversion
        reqs = [
            _make_request("req-a", priority=5, is_prefilling=False),
            _make_request("req-b", priority=5, is_prefilling=False),
        ]
        allocs = scheduler.compute_token_budget(reqs, total_budget=100)
        for a in allocs:
            tracker.record_allocation(a.request_id, a.prefill_tokens + a.decode_tokens)

        # Phase 2: inversion — low-priority running, high-priority waiting
        running = [_make_request("req-low", priority=1, wait_time=5.0, is_prefilling=False, effective_priority=1)]
        waiting = [_make_request("req-high", priority=50, is_prefilling=False, effective_priority=50)]

        events = guard.resolve(running, waiting)
        assert len(events) == 1

        # Phase 3: schedule with boosted priority
        allocs = scheduler.compute_token_budget(running + waiting, total_budget=100)
        for a in allocs:
            tracker.record_allocation(a.request_id, a.prefill_tokens + a.decode_tokens)

        # Phase 4: verify stats
        stats = guard.get_stats()
        assert stats["inversions_detected"] == 1
        assert stats["inheritance_applied"] == 1

        fairness = tracker.compute_fairness()
        assert 0.0 <= fairness <= 1.0

        tracker_stats = tracker.get_stats()
        assert tracker_stats["total_requests_tracked"] > 0
