"""Tests for Two-Batch Overlap (TBO) Scheduler — §14.1 gap from SGLang comparison.

Covers:
  - TBOConfig initialization, defaults, from_env
  - BatchState lifecycle
  - TwoBatchOverlapScheduler initialization, step, batch swapping
  - Auto-detection via should_enable_tbo()
  - get_stats() metrics
  - wrap_step() double-buffering integration
  - Sequential fallback logic
  - TBOMetrics recording and efficiency calculation
"""

import os
import time

import pytest

from yunshu_engine.two_batch_overlap import (
    BatchSlot,
    BatchState,
    TBOConfig,
    TBOMetrics,
    TwoBatchOverlapScheduler,
)


# ── Helpers ──────────────────────────────────────────────────


class FakeSchedulerOutput:
    """Minimal scheduler output for testing."""

    def __init__(self, outputs=None):
        self.outputs = outputs or []


class FakeRequestOutput:
    """Minimal request output for testing."""

    def __init__(self, request_id="req-1", finished=False, text="hello"):
        self.request_id = request_id
        self.finished = finished
        self.text = text
        self.completion_tokens = 1


class FakeScheduler:
    """Fake scheduler for testing TBO without MLX/BatchGenerator."""

    def __init__(self, num_running=4):
        self.running = {f"req-{i}": True for i in range(num_running)}
        self.step_count = 0

    def step(self):
        self.step_count += 1
        return FakeSchedulerOutput(
            outputs=[FakeRequestOutput(request_id=f"req-{i}") for i in range(len(self.running))]
        )

    def has_requests(self):
        return bool(self.running)


# ── TBOConfig ────────────────────────────────────────────────


class TestTBOConfig:
    def test_defaults(self):
        cfg = TBOConfig()
        assert cfg.enabled is False
        assert cfg.min_batch_size == 2
        assert cfg.low_util_threshold == 0.1
        assert cfg.fallback_window == 50
        assert cfg.metrics_window == 100

    def test_custom_config(self):
        cfg = TBOConfig(
            enabled=True,
            min_batch_size=4,
            low_util_threshold=0.2,
            fallback_window=30,
            metrics_window=200,
        )
        assert cfg.enabled is True
        assert cfg.min_batch_size == 4
        assert cfg.low_util_threshold == 0.2
        assert cfg.fallback_window == 30
        assert cfg.metrics_window == 200

    def test_from_env_disabled(self, monkeypatch):
        monkeypatch.delenv("YUNSHU_TBO", raising=False)
        cfg = TBOConfig.from_env()
        assert cfg.enabled is False

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_TBO", "1")
        cfg = TBOConfig.from_env()
        assert cfg.enabled is True

    def test_from_env_custom_min_batch(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_TBO", "1")
        monkeypatch.setenv("YUNSHU_TBO_MIN_BATCH", "8")
        cfg = TBOConfig.from_env()
        assert cfg.min_batch_size == 8

    def test_from_env_custom_threshold(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_TBO_LOW_UTIL", "0.25")
        cfg = TBOConfig.from_env()
        assert cfg.low_util_threshold == 0.25

    def test_to_dict(self):
        cfg = TBOConfig(enabled=True, min_batch_size=3)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["min_batch_size"] == 3
        assert "low_util_threshold" in d
        assert "fallback_window" in d
        assert "metrics_window" in d


# ── BatchState ───────────────────────────────────────────────


class TestBatchState:
    def test_default_state(self):
        bs = BatchState()
        assert bs.slot == BatchSlot.A
        assert bs.is_active is False
        assert bs.is_pending is False
        assert bs.outputs is None
        assert bs.gpu_time_ms == 0.0
        assert bs.cpu_prep_time_ms == 0.0

    def test_slot_b(self):
        bs = BatchState(slot=BatchSlot.B)
        assert bs.slot == BatchSlot.B

    def test_reset(self):
        bs = BatchState(
            slot=BatchSlot.A,
            is_active=True,
            is_pending=False,
            outputs=FakeSchedulerOutput(),
            gpu_time_ms=5.0,
            cpu_prep_time_ms=2.0,
        )
        bs.reset()
        assert bs.is_active is False
        assert bs.is_pending is False
        assert bs.outputs is None
        assert bs.gpu_time_ms == 0.0
        assert bs.cpu_prep_time_ms == 0.0


# ── BatchSlot ────────────────────────────────────────────────


class TestBatchSlot:
    def test_slots_are_distinct(self):
        assert BatchSlot.A != BatchSlot.B

    def test_slot_names(self):
        assert BatchSlot.A.name == "A"
        assert BatchSlot.B.name == "B"


# ── TBOMetrics ───────────────────────────────────────────────


class TestTBOMetrics:
    def test_initial_state(self):
        m = TBOMetrics()
        assert m.total_steps == 0
        assert m.overlapped_steps == 0
        assert m.sequential_fallback_steps == 0
        assert m.overlap_efficiency == 0.0
        assert m.overlap_rate == 0.0
        assert m.avg_gpu_time_ms == 0.0

    def test_record_step(self):
        m = TBOMetrics()
        m.record_step(
            gpu_time_ms=10.0,
            cpu_overlap_time_ms=3.0,
            idle_time_ms=7.0,
            was_overlapped=True,
        )
        assert m.total_steps == 1
        assert m.overlapped_steps == 1
        assert m.sequential_fallback_steps == 0

    def test_record_sequential_fallback(self):
        m = TBOMetrics()
        m.record_step(
            gpu_time_ms=10.0,
            cpu_overlap_time_ms=0.0,
            idle_time_ms=0.0,
            was_overlapped=False,
            was_sequential_fallback=True,
        )
        assert m.total_steps == 1
        assert m.overlapped_steps == 0
        assert m.sequential_fallback_steps == 1

    def test_overlap_rate(self):
        m = TBOMetrics()
        m.record_step(10, 3, 7, True)
        m.record_step(10, 0, 10, False)
        m.record_step(10, 3, 7, True)
        assert m.overlap_rate == pytest.approx(2 / 3, abs=0.01)

    def test_overlap_efficiency_with_data(self):
        m = TBOMetrics(_window=10)
        m.record_step(gpu_time_ms=10.0, cpu_overlap_time_ms=5.0, idle_time_ms=5.0, was_overlapped=True)
        # efficiency = avg_overlap / avg_gpu = 5.0 / 10.0 = 0.5
        assert m.overlap_efficiency == pytest.approx(0.5, abs=0.01)

    def test_overlap_efficiency_capped_at_1(self):
        m = TBOMetrics(_window=10)
        # CPU overlap > GPU time should be capped at 1.0
        m.record_step(gpu_time_ms=5.0, cpu_overlap_time_ms=10.0, idle_time_ms=0.0, was_overlapped=True)
        assert m.overlap_efficiency == 1.0

    def test_avg_times(self):
        m = TBOMetrics()
        m.record_step(10.0, 3.0, 7.0, True)
        m.record_step(20.0, 5.0, 15.0, True)
        assert m.avg_gpu_time_ms == pytest.approx(15.0)
        assert m.avg_cpu_overlap_ms == pytest.approx(4.0)
        assert m.avg_idle_time_ms == pytest.approx(11.0)

    def test_get_stats(self):
        m = TBOMetrics()
        m.record_step(10.0, 3.0, 7.0, True)
        stats = m.get_stats()
        assert "total_steps" in stats
        assert "overlapped_steps" in stats
        assert "overlap_rate" in stats
        assert "overlap_efficiency" in stats
        assert "avg_gpu_time_ms" in stats
        assert "avg_cpu_overlap_ms" in stats
        assert "avg_idle_time_ms" in stats
        assert stats["total_steps"] == 1

    def test_reset(self):
        m = TBOMetrics()
        m.record_step(10.0, 3.0, 7.0, True)
        m.reset()
        assert m.total_steps == 0
        assert m.overlapped_steps == 0
        assert m.avg_gpu_time_ms == 0.0

    def test_efficiency_streak(self):
        m = TBOMetrics()
        m.update_efficiency_streak(0.05, 0.1)
        assert m.low_efficiency_streak == 1
        m.update_efficiency_streak(0.05, 0.1)
        assert m.low_efficiency_streak == 2
        # Good efficiency resets streak
        m.update_efficiency_streak(0.5, 0.1)
        assert m.low_efficiency_streak == 0

    def test_rolling_window_truncation(self):
        m = TBOMetrics(_window=3)
        for i in range(10):
            m.record_step(10.0, 5.0, 5.0, True)
        assert len(m._recent_gpu_ms) == 3
        assert len(m._recent_cpu_overlap_ms) == 3


# ── TwoBatchOverlapScheduler ─────────────────────────────────


class TestTwoBatchOverlapSchedulerInit:
    def test_default_config(self):
        tbo = TwoBatchOverlapScheduler()
        assert tbo.config.enabled is False
        assert tbo.active_slot == BatchSlot.A
        assert tbo.is_overlapped is True  # not in fallback initially

    def test_custom_config(self):
        cfg = TBOConfig(enabled=True, min_batch_size=4)
        tbo = TwoBatchOverlapScheduler(cfg)
        assert tbo.config.enabled is True
        assert tbo.config.min_batch_size == 4

    def test_initial_batch_states(self):
        tbo = TwoBatchOverlapScheduler()
        assert tbo.batch_a.slot == BatchSlot.A
        assert tbo.batch_b.slot == BatchSlot.B
        assert tbo.batch_a.is_active is False
        assert tbo.batch_b.is_active is False


class TestTwoBatchOverlapSchedulerStep:
    def test_step_disabled_passes_through(self):
        """When TBO is disabled, step() passes through to scheduler directly."""
        cfg = TBOConfig(enabled=False)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        output = tbo.step(scheduler)
        assert scheduler.step_count == 1
        assert isinstance(output, FakeSchedulerOutput)

    def test_step_enabled_returns_output(self):
        """When TBO is enabled with sufficient batch, returns output."""
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        output = tbo.step(scheduler)
        assert isinstance(output, FakeSchedulerOutput)
        assert len(output.outputs) == 4

    def test_step_records_metrics(self):
        """Step records timing metrics."""
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        tbo.step(scheduler)
        stats = tbo.metrics.get_stats()
        assert stats["total_steps"] == 1

    def test_multiple_steps_record_metrics(self):
        """Multiple steps accumulate metrics."""
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        for _ in range(5):
            tbo.step(scheduler)
        stats = tbo.metrics.get_stats()
        assert stats["total_steps"] == 5


class TestBatchSwapLogic:
    def test_swap_a_to_b(self):
        tbo = TwoBatchOverlapScheduler()
        assert tbo.active_slot == BatchSlot.A
        tbo._swap_batches()
        assert tbo.active_slot == BatchSlot.B

    def test_swap_b_to_a(self):
        tbo = TwoBatchOverlapScheduler()
        tbo._swap_batches()  # A -> B
        tbo._swap_batches()  # B -> A
        assert tbo.active_slot == BatchSlot.A

    def test_swap_cycles_abab(self):
        """Verify A→B→A→A→B cycling over multiple swaps."""
        tbo = TwoBatchOverlapScheduler()
        expected = [BatchSlot.B, BatchSlot.A, BatchSlot.B, BatchSlot.A]
        actual = []
        for _ in range(4):
            tbo._swap_batches()
            actual.append(tbo.active_slot)
        assert actual == expected

    def test_swap_updates_batch_state(self):
        tbo = TwoBatchOverlapScheduler()
        # Initially A is neither active nor pending
        tbo._batch_a.is_active = True
        tbo._swap_batches()
        assert tbo._batch_a.is_active is False
        assert tbo._batch_a.is_pending is True
        assert tbo._batch_b.is_active is True
        assert tbo._batch_b.is_pending is False

    def test_step_with_tbo_swaps_batch(self):
        """Each TBO step swaps the active batch."""
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)

        tbo.step(scheduler)
        # After first step, should have swapped at least once
        assert tbo.metrics.total_steps == 1


class TestSequentialFallback:
    def test_fallback_when_batch_too_small(self):
        """TBO falls back to sequential when batch size < min_batch_size."""
        cfg = TBOConfig(enabled=True, min_batch_size=4)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=1)  # Only 1 running request
        tbo.step(scheduler)
        assert tbo._in_fallback is True
        assert tbo.metrics.get_stats()["sequential_fallback_steps"] == 1

    def test_fallback_records_zero_overlap(self):
        """Sequential fallback records no overlap time."""
        cfg = TBOConfig(enabled=True, min_batch_size=4)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=1)
        tbo.step(scheduler)
        stats = tbo.metrics.get_stats()
        assert stats["avg_cpu_overlap_ms"] == 0.0

    def test_no_fallback_with_sufficient_batch(self):
        """TBO engages when batch size >= min_batch_size."""
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        tbo.step(scheduler)
        assert tbo._in_fallback is False

    def test_mixed_steps_track_both(self):
        """Mix of overlapped and fallback steps both tracked."""
        cfg = TBOConfig(enabled=True, min_batch_size=3)
        tbo = TwoBatchOverlapScheduler(cfg)

        # Big batch — should overlap
        big_scheduler = FakeScheduler(num_running=5)
        tbo.step(big_scheduler)

        # Small batch — should fallback
        small_scheduler = FakeScheduler(num_running=1)
        tbo.step(small_scheduler)

        stats = tbo.metrics.get_stats()
        assert stats["total_steps"] == 2
        assert stats["overlapped_steps"] == 1
        assert stats["sequential_fallback_steps"] == 1


class TestShouldEnableTBO:
    def test_returns_false_with_no_history(self):
        tbo = TwoBatchOverlapScheduler()
        assert tbo.should_enable_tbo() is False

    def test_returns_false_for_small_batches(self):
        cfg = TBOConfig(min_batch_size=4)
        tbo = TwoBatchOverlapScheduler(cfg)
        # All batches are size 1 — below threshold
        for _ in range(20):
            tbo._record_batch_size(1)
        assert tbo.should_enable_tbo() is False

    def test_returns_true_for_large_batches(self):
        cfg = TBOConfig(min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        # All batches are size 8 — above threshold
        for _ in range(20):
            tbo._record_batch_size(8)
        assert tbo.should_enable_tbo() is True

    def test_requires_minimum_fraction_of_multi_batches(self):
        """Even with high average, if <30% of steps are multi-batch, returns False."""
        cfg = TBOConfig(min_batch_size=4)
        tbo = TwoBatchOverlapScheduler(cfg)
        # 10 steps with batch=8, 40 steps with batch=1 — average ~2.4, but mostly single
        for _ in range(10):
            tbo._record_batch_size(8)
        for _ in range(40):
            tbo._record_batch_size(1)
        assert tbo.should_enable_tbo() is False

    def test_mixed_workload_with_sufficient_multi_batches(self):
        """With >=30% multi-batch steps and good average, returns True."""
        cfg = TBOConfig(min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        # 20 steps batch=8, 10 steps batch=1 — avg ~5.7, 67% multi
        for _ in range(20):
            tbo._record_batch_size(8)
        for _ in range(10):
            tbo._record_batch_size(1)
        assert tbo.should_enable_tbo() is True


class TestGetStats:
    def test_stats_structure(self):
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        stats = tbo.get_stats()
        assert "config" in stats
        assert "metrics" in stats
        assert "active_slot" in stats
        assert "in_fallback" in stats
        assert "recent_avg_batch_size" in stats
        assert "should_enable_tbo" in stats

    def test_stats_after_steps(self):
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        for _ in range(3):
            tbo.step(scheduler)
        stats = tbo.get_stats()
        assert stats["metrics"]["total_steps"] == 3
        assert stats["active_slot"] in ("A", "B")

    def test_stats_config_matches_input(self):
        cfg = TBOConfig(enabled=True, min_batch_size=5, low_util_threshold=0.2)
        tbo = TwoBatchOverlapScheduler(cfg)
        stats = tbo.get_stats()
        assert stats["config"]["enabled"] is True
        assert stats["config"]["min_batch_size"] == 5
        assert stats["config"]["low_util_threshold"] == 0.2

    def test_stats_avg_batch_size(self):
        tbo = TwoBatchOverlapScheduler()
        for bs in [4, 8, 2, 6]:
            tbo._record_batch_size(bs)
        stats = tbo.get_stats()
        assert stats["recent_avg_batch_size"] == pytest.approx(5.0)


class TestReset:
    def test_reset_clears_all_state(self):
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        tbo.step(scheduler)
        tbo.step(scheduler)
        tbo.step(scheduler)

        tbo.reset()
        assert tbo.metrics.total_steps == 0
        assert tbo.active_slot == BatchSlot.A
        assert tbo._in_fallback is False
        assert tbo._batch_a.is_active is False
        assert tbo._batch_b.is_active is False
        assert len(tbo._batch_size_history) == 0


class TestWrapStep:
    def test_wrap_step_returns_callable(self):
        tbo = TwoBatchOverlapScheduler(TBOConfig(enabled=True, min_batch_size=2))
        scheduler = FakeScheduler(num_running=4)
        wrapped = tbo.wrap_step(scheduler, scheduler.step)
        assert callable(wrapped)

    def test_wrap_step_produces_output(self):
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        wrapped = tbo.wrap_step(scheduler, scheduler.step)
        output = wrapped()
        assert isinstance(output, FakeSchedulerOutput)
        assert len(output.outputs) == 4

    def test_wrap_step_tracks_metrics(self):
        cfg = TBOConfig(enabled=True, min_batch_size=2)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        wrapped = tbo.wrap_step(scheduler, scheduler.step)
        for _ in range(3):
            wrapped()
        assert tbo.metrics.total_steps == 3

    def test_wrap_step_disabled_still_works(self):
        """wrap_step works even when TBO is disabled (passthrough)."""
        cfg = TBOConfig(enabled=False)
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)
        wrapped = tbo.wrap_step(scheduler, scheduler.step)
        output = wrapped()
        assert isinstance(output, FakeSchedulerOutput)
        assert scheduler.step_count == 1


class TestCountRunningRequests:
    def test_dict_running(self):
        tbo = TwoBatchOverlapScheduler()
        scheduler = FakeScheduler(num_running=5)
        assert tbo._count_running_requests(scheduler) == 5

    def test_list_running(self):
        tbo = TwoBatchOverlapScheduler()

        class ListScheduler:
            running = ["r1", "r2", "r3"]

        assert tbo._count_running_requests(ListScheduler()) == 3

    def test_no_running_attr(self):
        tbo = TwoBatchOverlapScheduler()

        class EmptyScheduler:
            pass

        assert tbo._count_running_requests(EmptyScheduler()) == 0

    def test_set_running(self):
        tbo = TwoBatchOverlapScheduler()

        class SetScheduler:
            running = {"r1", "r2"}

        assert tbo._count_running_requests(SetScheduler()) == 2


class TestLowEfficiencyFallback:
    def test_sustained_low_efficiency_triggers_fallback(self):
        """When efficiency stays below threshold for fallback_window steps,
        sequential fallback activates."""
        cfg = TBOConfig(
            enabled=True,
            min_batch_size=2,
            low_util_threshold=0.1,
            fallback_window=5,
        )
        tbo = TwoBatchOverlapScheduler(cfg)
        scheduler = FakeScheduler(num_running=4)

        # Simulate low efficiency by directly manipulating streak
        for _ in range(6):
            tbo.metrics.update_efficiency_streak(0.01, 0.1)

        assert tbo.metrics.low_efficiency_streak >= 5
        # Now step should use fallback
        tbo.step(scheduler)
        assert tbo._in_fallback is True
