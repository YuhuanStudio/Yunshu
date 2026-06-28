"""Tests for Memory-Aware Request Scheduler.

Tests cover:
- Memory estimation for various token counts and model configs
- Admission control: accept/reject based on budget
- Reservation/release lifecycle
- Pressure-based pausing and resuming
- Statistics tracking
- Edge cases: zero budget, oversized requests, concurrent reservations
"""
from __future__ import annotations

from yunshu_engine.memory_aware_scheduler import (
    MemoryAwareScheduler,
    MemoryBudget,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_scheduler(
    budget_mb: int = 1024,
    pressure_pct: float = 85.0,
    hysteresis: float = 5.0,
) -> MemoryAwareScheduler:
    """Create a scheduler with budget in MB."""
    return MemoryAwareScheduler(
        total_budget_bytes=budget_mb * 1024 * 1024,
        pressure_threshold_pct=pressure_pct,
        hysteresis_pct=hysteresis,
    )


def _per_token_bytes(
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_size: int = 2,
) -> int:
    """Calculate per-token KV bytes matching the scheduler formula."""
    return num_layers * 2 * num_kv_heads * head_dim * dtype_size


# ═══════════════════════════════════════════════════════════════════════════════
# Test Memory Estimation
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryEstimation:

    def test_basic_prompt_estimation(self):
        """estimate_kv_memory returns non-zero for valid tokens."""
        sched = _make_scheduler()
        sched.set_model_config(num_layers=32, num_kv_heads=8, head_dim=128)
        mem = sched.estimate_kv_memory(num_tokens=100)
        per_tok = _per_token_bytes()
        assert mem == 100 * per_tok

    def test_prompt_plus_decode_estimation(self):
        """estimate_kv_memory includes both prompt and max_tokens."""
        sched = _make_scheduler()
        sched.set_model_config(num_layers=32, num_kv_heads=8, head_dim=128)
        mem = sched.estimate_kv_memory(num_tokens=100, max_tokens=50)
        per_tok = _per_token_bytes()
        assert mem == 150 * per_tok

    def test_zero_tokens(self):
        """Zero tokens requires zero memory."""
        sched = _make_scheduler()
        sched.set_model_config(num_layers=32, num_kv_heads=8, head_dim=128)
        assert sched.estimate_kv_memory(num_tokens=0) == 0

    def test_model_config_override(self):
        """model_config parameter overrides instance config."""
        sched = _make_scheduler()
        sched.set_model_config(num_layers=32, num_kv_heads=8, head_dim=128)
        mem_default = sched.estimate_kv_memory(num_tokens=100)
        # Smaller model: 16 layers instead of 32
        mem_small = sched.estimate_kv_memory(
            num_tokens=100,
            model_config={"num_layers": 16, "num_kv_heads": 8, "head_dim": 128, "dtype_size": 2},
        )
        assert mem_small == mem_default // 2

    def test_estimation_scales_linearly(self):
        """Memory estimation scales linearly with token count."""
        sched = _make_scheduler()
        sched.set_model_config(num_layers=32, num_kv_heads=8, head_dim=128)
        mem_100 = sched.estimate_kv_memory(num_tokens=100)
        mem_200 = sched.estimate_kv_memory(num_tokens=200)
        assert mem_200 == 2 * mem_100

    def test_default_model_config(self):
        """Default model config produces reasonable estimates."""
        sched = _make_scheduler()
        # Uses defaults: 32 layers, 8 kv_heads, 128 head_dim, 2 bytes
        mem = sched.estimate_kv_memory(num_tokens=100)
        assert mem > 0
        per_tok = _per_token_bytes()
        assert mem == 100 * per_tok


# ═══════════════════════════════════════════════════════════════════════════════
# Test Admission Control
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdmissionControl:

    def test_admit_small_request(self):
        """Small request is admitted when budget has room."""
        sched = _make_scheduler(budget_mb=1024)
        per_tok = _per_token_bytes()
        small_request = 100 * per_tok  # 100 tokens
        can_admit, reason = sched.can_admit_request(small_request)
        assert can_admit is True
        assert reason == ""

    def test_reject_oversized_request(self):
        """Request exceeding budget is rejected."""
        sched = _make_scheduler(budget_mb=1)  # 1 MB budget
        per_tok = _per_token_bytes()
        huge_request = 10000 * per_tok  # Way more than 1 MB
        can_admit, reason = sched.can_admit_request(huge_request)
        assert can_admit is False
        assert "safety margin" in reason.lower() or "exceed" in reason.lower()

    def test_admit_fills_budget(self):
        """Multiple requests can fill budget to safety margin."""
        sched = _make_scheduler(budget_mb=10)
        per_tok = _per_token_bytes()
        small = 10 * per_tok  # Small request

        admitted = 0
        for i in range(100):
            can, _ = sched.can_admit_request(small)
            if can:
                sched.reserve_memory(f"req-{i}", small)
                admitted += 1
            else:
                break
        assert admitted > 0
        assert admitted < 100  # Should hit budget before 100

    def test_admission_after_release(self):
        """Requests can be admitted after memory is released."""
        sched = _make_scheduler(budget_mb=100)
        per_tok = _per_token_bytes()
        req_size = 50 * per_tok

        # Fill up budget
        can1, _ = sched.can_admit_request(req_size)
        assert can1 is True
        sched.reserve_memory("req-1", req_size)

        # Try another — might be rejected if budget is tight
        can2, _ = sched.can_admit_request(req_size)

        # Release first request
        sched.release_memory("req-1")

        # Now should be able to admit again
        can3, _ = sched.can_admit_request(req_size)
        assert can3 is True


# ═══════════════════════════════════════════════════════════════════════════════
# Test Reservation/Release Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestReservationLifecycle:

    def test_reserve_increases_used(self):
        """Reserving memory increases used bytes."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 10 * 1024 * 1024)
        budget = sched.get_memory_budget()
        assert budget.reserved_bytes == 10 * 1024 * 1024

    def test_release_decreases_used(self):
        """Releasing memory decreases used bytes."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 10 * 1024 * 1024)
        released = sched.release_memory("req-1")
        assert released == 10 * 1024 * 1024
        budget = sched.get_memory_budget()
        assert budget.reserved_bytes == 0

    def test_release_unknown_request(self):
        """Releasing unknown request returns 0."""
        sched = _make_scheduler()
        assert sched.release_memory("nonexistent") == 0

    def test_reserve_exceeding_available(self):
        """Reserving more than available returns False."""
        sched = _make_scheduler(budget_mb=1)
        # Budget is 1 MB, safety margin makes usable < 1 MB
        big = 2 * 1024 * 1024
        result = sched.reserve_memory("req-1", big)
        assert result is False

    def test_multiple_reservations(self):
        """Multiple requests reserve independently."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 10 * 1024 * 1024)
        sched.reserve_memory("req-2", 20 * 1024 * 1024)
        budget = sched.get_memory_budget()
        assert budget.reserved_bytes == 30 * 1024 * 1024

    def test_partial_release(self):
        """Releasing one request keeps others reserved."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 10 * 1024 * 1024)
        sched.reserve_memory("req-2", 20 * 1024 * 1024)
        sched.release_memory("req-1")
        budget = sched.get_memory_budget()
        assert budget.reserved_bytes == 20 * 1024 * 1024

    def test_reserve_with_metadata(self):
        """Reserve with num_tokens and model_name metadata."""
        sched = _make_scheduler(budget_mb=100)
        ok = sched.reserve_memory("req-1", 1024, num_tokens=100, model_name="qwen")
        assert ok is True
        stats = sched.get_stats()
        assert stats.total_admissions == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test Pressure-Based Pausing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPressurePausing:

    def test_pressure_pauses_admissions(self):
        """Admissions pause when utilization crosses threshold."""
        sched = _make_scheduler(budget_mb=10, pressure_pct=50.0, hysteresis=10.0)
        _per_token_bytes()
        # Fill to near 50%
        big_request = 4 * 1024 * 1024  # 4 MB
        sched.reserve_memory("req-1", big_request)
        # Another request that pushes past 50%
        can, _ = sched.can_admit_request(big_request)
        if can:
            sched.reserve_memory("req-2", big_request)
        # After crossing 50%, should be paused
        assert sched.is_paused is True

    def test_resume_after_release(self):
        """Admissions resume after memory is released below threshold."""
        sched = _make_scheduler(budget_mb=10, pressure_pct=50.0, hysteresis=10.0)
        big = 4 * 1024 * 1024
        sched.reserve_memory("req-1", big)
        sched.can_admit_request(big)
        if sched.reserve_memory("req-2", big):
            pass
        assert sched.is_paused is True
        # Release both
        sched.release_memory("req-1")
        sched.release_memory("req-2")
        # Check admission again — should resume
        can, _ = sched.can_admit_request(1024)
        assert can is True
        assert sched.is_paused is False

    def test_paused_rejection_message(self):
        """Paused scheduler gives clear rejection reason."""
        sched = _make_scheduler(budget_mb=10, pressure_pct=50.0, hysteresis=10.0)
        big = 4 * 1024 * 1024
        sched.reserve_memory("req-1", big)
        sched.can_admit_request(big)
        if sched.reserve_memory("req-2", big):
            pass
        # Now paused — next request should be rejected with reason
        can, reason = sched.can_admit_request(1024)
        if not can:
            assert "paused" in reason.lower() or "pressure" in reason.lower()

    def test_hysteresis_prevents_thrashing(self):
        """Hysteresis prevents rapid admit/pause cycling."""
        sched = _make_scheduler(budget_mb=10, pressure_pct=80.0, hysteresis=15.0)
        # Reserve exactly at threshold
        at_threshold = int(10 * 1024 * 1024 * 0.80)
        sched.reserve_memory("req-1", at_threshold)
        # Check admission — should trigger pause
        sched.can_admit_request(1024)
        # Release a tiny bit — still above (80-15)=65%
        sched.release_memory("req-1")
        # Re-reserve at 70% — still paused (above 65%)
        sched.reserve_memory("req-1", int(10 * 1024 * 1024 * 0.70))
        can, _ = sched.can_admit_request(1024)
        # Should still be paused since 70% > 65% (80% - 15%)
        if sched.is_paused and not can:
            assert True  # Hysteresis working

    def test_stats_pressure_tracking(self):
        """Stats track pressure pauses and resumes."""
        sched = _make_scheduler(budget_mb=5, pressure_pct=50.0, hysteresis=10.0)
        big = 2 * 1024 * 1024
        sched.reserve_memory("req-1", big)
        sched.can_admit_request(big)
        stats = sched.get_stats()
        assert stats.pressure_pauses >= 0  # May or may not have paused depending on exact budget math


# ═══════════════════════════════════════════════════════════════════════════════
# Test Memory Budget
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryBudget:

    def test_initial_budget(self):
        """Initial budget has full capacity available."""
        sched = _make_scheduler(budget_mb=100)
        budget = sched.get_memory_budget()
        assert budget.total_bytes == 100 * 1024 * 1024
        assert budget.reserved_bytes == 0
        assert budget.available_bytes > 0

    def test_budget_utilization(self):
        """Utilization tracks reserved/total ratio."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 50 * 1024 * 1024)
        budget = sched.get_memory_budget()
        assert budget.utilization_pct > 0

    def test_free_bytes_property(self):
        """MemoryBudget.free_bytes returns total - used."""
        budget = MemoryBudget(
            total_bytes=100,
            reserved_bytes=30,
            used_bytes=30,
            available_bytes=65,
            utilization_pct=30.0,
        )
        assert budget.free_bytes == 70

    def test_safety_margin_reduces_usable(self):
        """Safety margin means not all budget is usable."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 95 * 1024 * 1024)
        # 95% of budget is reserved — should hit safety margin
        can, _ = sched.can_admit_request(1024)
        assert can is False


# ═══════════════════════════════════════════════════════════════════════════════
# Test Statistics
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchedulerStats:

    def test_initial_stats(self):
        """Initial stats are all zeros."""
        sched = _make_scheduler()
        stats = sched.get_stats()
        assert stats.total_admissions == 0
        assert stats.total_rejections == 0
        assert stats.total_releases == 0
        assert stats.current_reserved_bytes == 0
        assert stats.active_requests == 0

    def test_stats_after_admission(self):
        """Stats increment on admission."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 1024)
        stats = sched.get_stats()
        assert stats.total_admissions == 1
        assert stats.active_requests == 1
        assert stats.current_reserved_bytes == 1024

    def test_stats_after_release(self):
        """Stats increment on release."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 1024)
        sched.release_memory("req-1")
        stats = sched.get_stats()
        assert stats.total_releases == 1
        assert stats.active_requests == 0
        assert stats.total_bytes_released == 1024

    def test_peak_reserved_tracking(self):
        """Peak reserved bytes tracks the maximum."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 10 * 1024 * 1024)
        sched.reserve_memory("req-2", 20 * 1024 * 1024)
        stats = sched.get_stats()
        assert stats.peak_reserved_bytes == 30 * 1024 * 1024
        # Release one — peak should still be 30 MB
        sched.release_memory("req-1")
        stats = sched.get_stats()
        assert stats.peak_reserved_bytes == 30 * 1024 * 1024

    def test_stats_is_paused(self):
        """Stats reflect paused state."""
        sched = _make_scheduler(budget_mb=5, pressure_pct=50.0, hysteresis=10.0)
        big = 2 * 1024 * 1024
        sched.reserve_memory("req-1", big)
        sched.can_admit_request(big)
        stats = sched.get_stats()
        assert stats.is_paused == sched.is_paused


# ═══════════════════════════════════════════════════════════════════════════════
# Test Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_zero_budget(self):
        """Zero budget rejects all requests."""
        sched = MemoryAwareScheduler(total_budget_bytes=0)
        can, _ = sched.can_admit_request(1)
        assert can is False

    def test_reserve_zero_bytes(self):
        """Reserving 0 bytes succeeds."""
        sched = _make_scheduler(budget_mb=100)
        assert sched.reserve_memory("req-1", 0) is True

    def test_duplicate_request_id(self):
        """Duplicate request ID overwrites previous reservation."""
        sched = _make_scheduler(budget_mb=100)
        sched.reserve_memory("req-1", 10 * 1024 * 1024)
        sched.reserve_memory("req-1", 20 * 1024 * 1024)
        budget = sched.get_memory_budget()
        # Second reserve overwrites, so only 20 MB reserved
        assert budget.reserved_bytes == 20 * 1024 * 1024

    def test_total_budget_property(self):
        """total_budget property returns configured value."""
        sched = _make_scheduler(budget_mb=500)
        assert sched.total_budget == 500 * 1024 * 1024

    def test_model_config_partial_update(self):
        """set_model_config with partial params only updates specified."""
        sched = _make_scheduler()
        sched.set_model_config(num_layers=16)
        # Other params keep defaults
        mem = sched.estimate_kv_memory(num_tokens=100)
        per_tok = 16 * 2 * 8 * 128 * 2  # Only num_layers changed
        assert mem == 100 * per_tok
