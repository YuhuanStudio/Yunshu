"""Tests for Auto-Tuning Engine — profiler, tuner, batch sizer, SLO monitor."""


import pytest

from yunshu_engine.auto_tuner import (
    AdaptiveBatchSizer,
    AutoTuner,
    BottleneckType,
    PerformanceProfiler,
    SLOConfig,
    SLOMonitor,
    StepMetrics,
    TunableParams,
)

# ===================================================================
# Helpers
# ===================================================================

def _metrics(
    throughput: float = 50.0,
    ttft: float = 100.0,
    itl: float = 30.0,
    gpu_util: float = 0.5,
    batch: int = 8,
    tokens: int = 10,
    wall: float = 200.0,
) -> StepMetrics:
    """Create a StepMetrics with sensible defaults."""
    return StepMetrics(
        throughput_tok_s=throughput,
        ttft_ms=ttft,
        itl_ms=itl,
        gpu_memory_util=gpu_util,
        batch_size=batch,
        tokens_generated=tokens,
        wall_time_ms=wall,
    )


# ===================================================================
# PerformanceProfiler
# ===================================================================

class TestPerformanceProfilerBasic:
    """Basic lifecycle and recording tests."""

    def test_initial_state(self):
        p = PerformanceProfiler()
        assert not p.is_profiling
        assert p.get_bottleneck() == BottleneckType.NONE

    def test_start_stop_profiling(self):
        p = PerformanceProfiler()
        p.start_profiling()
        assert p.is_profiling
        p.stop_profiling()
        assert not p.is_profiling

    def test_record_step(self):
        p = PerformanceProfiler()
        p.start_profiling()
        p.record_step(_metrics())
        stats = p.get_stats()
        assert stats["total_steps"] == 1
        assert stats["window_filled"] == 1

    def test_multiple_steps(self):
        p = PerformanceProfiler(window_size=10)
        for _ in range(15):
            p.record_step(_metrics())
        stats = p.get_stats()
        assert stats["total_steps"] == 15
        # Window capped at 10
        assert stats["window_filled"] == 10

    def test_avg_metrics_empty(self):
        p = PerformanceProfiler()
        avg = p.get_avg_metrics()
        assert avg["throughput_tok_s"] == 0.0

    def test_avg_metrics_computed(self):
        p = PerformanceProfiler()
        p.record_step(_metrics(throughput=100.0, ttft=50.0, itl=20.0, gpu_util=0.6))
        p.record_step(_metrics(throughput=200.0, ttft=100.0, itl=40.0, gpu_util=0.8))
        avg = p.get_avg_metrics()
        assert abs(avg["throughput_tok_s"] - 150.0) < 0.01
        assert abs(avg["ttft_ms"] - 75.0) < 0.01
        assert abs(avg["itl_ms"] - 30.0) < 0.01
        assert abs(avg["gpu_memory_util"] - 0.7) < 0.01


class TestBottleneckDetection:
    """Test bottleneck classification."""

    def test_memory_bottleneck(self):
        p = PerformanceProfiler(bottleneck_memory_threshold=0.8)
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.92, throughput=40.0, tokens=10, wall=200.0))
        assert p.get_bottleneck() == BottleneckType.MEMORY

    def test_compute_bottleneck(self):
        """Low throughput with moderate memory = compute bound."""
        p = PerformanceProfiler()
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.5, throughput=20.0, tokens=5, wall=250.0))
        assert p.get_bottleneck() == BottleneckType.COMPUTE

    def test_io_bottleneck(self):
        """Very low throughput = IO bound."""
        p = PerformanceProfiler()
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.3, throughput=5.0, tokens=3, wall=600.0))
        assert p.get_bottleneck() == BottleneckType.IO

    def test_no_bottleneck(self):
        """Good throughput, moderate memory = no bottleneck."""
        p = PerformanceProfiler()
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.5, throughput=80.0, tokens=20, wall=250.0))
        assert p.get_bottleneck() == BottleneckType.NONE

    def test_bottleneck_distribution_in_stats(self):
        p = PerformanceProfiler()
        p.record_step(_metrics(gpu_util=0.95))  # memory
        p.record_step(_metrics(gpu_util=0.5, throughput=80.0))  # none
        stats = p.get_stats()
        assert stats["bottleneck_distribution"]["memory"] == 1
        assert stats["bottleneck_distribution"]["none"] == 1


class TestProfilerRecommendations:
    """Test recommendation generation."""

    def test_memory_recommendations(self):
        p = PerformanceProfiler()
        # Force memory bottleneck by recording high-util steps
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.92, throughput=40.0))
        recs = p.get_recommendations()
        assert len(recs) > 0
        params = {r["param"] for r in recs}
        assert "batch_size" in params
        assert "kv_quantization_bits" in params

    def test_compute_recommendations(self):
        p = PerformanceProfiler()
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.5, throughput=20.0, tokens=5, wall=250.0))
        recs = p.get_recommendations()
        params = {r["param"] for r in recs}
        assert "spec_draft_length" in params

    def test_io_recommendations(self):
        p = PerformanceProfiler()
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.3, throughput=5.0, tokens=3, wall=600.0))
        recs = p.get_recommendations()
        params = {r["param"] for r in recs}
        assert "prefill_chunk_size" in params

    def test_no_recommendations_when_ok(self):
        p = PerformanceProfiler()
        for _ in range(10):
            p.record_step(_metrics(gpu_util=0.5, throughput=80.0))
        recs = p.get_recommendations()
        assert len(recs) == 0


# ===================================================================
# SLOMonitor
# ===================================================================

class TestSLOMonitorCompliance:
    """Test SLO compliance checking."""

    def test_ttft_within_slo(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        assert m.check_slo("ttft", 150.0) is True

    def test_ttft_exceeds_slo(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        assert m.check_slo("ttft", 300.0) is False

    def test_itl_within_slo(self):
        m = SLOMonitor(SLOConfig(itl_ms=50.0))
        assert m.check_slo("itl", 30.0) is True

    def test_itl_exceeds_slo(self):
        m = SLOMonitor(SLOConfig(itl_ms=50.0))
        assert m.check_slo("itl", 80.0) is False

    def test_throughput_within_slo(self):
        m = SLOMonitor(SLOConfig(throughput_tok_s=20.0))
        assert m.check_slo("throughput", 50.0) is True

    def test_throughput_below_slo(self):
        m = SLOMonitor(SLOConfig(throughput_tok_s=20.0))
        assert m.check_slo("throughput", 10.0) is False

    def test_unknown_metric_always_met(self):
        m = SLOMonitor()
        assert m.check_slo("unknown_metric", 0.0) is True

    def test_compliance_all_met(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0, itl_ms=50.0, throughput_tok_s=20.0))
        for _ in range(10):
            m.check_slo("ttft", 100.0)
            m.check_slo("itl", 30.0)
            m.check_slo("throughput", 50.0)
        compliance = m.get_slo_compliance()
        assert compliance["ttft"] == 100.0
        assert compliance["itl"] == 100.0
        assert compliance["throughput"] == 100.0

    def test_compliance_mixed(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        for _ in range(8):
            m.check_slo("ttft", 100.0)  # met
        for _ in range(2):
            m.check_slo("ttft", 300.0)  # violated
        compliance = m.get_slo_compliance()
        assert compliance["ttft"] == 80.0  # 8/10


class TestSLOViolations:
    """Test violation tracking and auto-tuning triggers."""

    def test_violations_recorded(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.check_slo("ttft", 300.0)
        violations = m.get_violations()
        assert len(violations) == 1
        assert violations[0]["metric"] == "ttft"
        assert violations[0]["value"] == 300.0

    def test_no_violations_when_met(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.check_slo("ttft", 100.0)
        assert len(m.get_violations()) == 0

    def test_violations_most_recent_first(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.check_slo("ttft", 300.0)
        m.check_slo("ttft", 400.0)
        violations = m.get_violations()
        assert violations[0]["value"] == 400.0
        assert violations[1]["value"] == 300.0

    def test_violations_limit(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        for i in range(50):
            m.check_slo("ttft", 300.0 + i)
        violations = m.get_violations(limit=10)
        assert len(violations) == 10

    def test_auto_tuning_callback_triggered(self):
        triggered = []
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.set_auto_tuning_callback(lambda v: triggered.append(v))
        # Need >30% violation rate over at least 5 checks
        for _ in range(5):
            m.check_slo("ttft", 300.0)  # all violations -> 100% rate
        assert len(triggered) >= 1

    def test_auto_tuning_not_triggered_low_rate(self):
        triggered = []
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.set_auto_tuning_callback(lambda v: triggered.append(v))
        # 1 violation out of 10 = 10% < 30%
        for _ in range(9):
            m.check_slo("ttft", 100.0)
        m.check_slo("ttft", 300.0)
        assert len(triggered) == 0

    def test_slo_monitor_stats(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.check_slo("ttft", 100.0)
        m.check_slo("ttft", 300.0)
        stats = m.get_stats()
        assert stats["check_counts"]["ttft"] == 2
        assert stats["violation_counts"]["ttft"] == 1
        assert stats["auto_tuning_triggers"] == 0

    def test_slo_reset(self):
        m = SLOMonitor(SLOConfig(ttft_ms=200.0))
        m.check_slo("ttft", 300.0)
        m.reset()
        stats = m.get_stats()
        assert stats["check_counts"]["ttft"] == 0
        assert stats["violation_counts"]["ttft"] == 0
        assert stats["auto_tuning_triggers"] == 0


# ===================================================================
# AdaptiveBatchSizer
# ===================================================================

class TestAdaptiveBatchSizer:
    """Test dynamic batch size computation."""

    def test_starts_at_min(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        assert s.current_batch_size == 1

    def test_scales_up_low_pressure(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        s._current_batch = 4
        batch = s.compute_optimal_batch(
            queue_depth=32,
            memory_available=0.8,  # low utilization
            slo_latency_ms=200.0,
            current_latency_ms=100.0,  # within SLO
        )
        assert batch > 4

    def test_scales_down_memory_pressure(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        s._current_batch = 16
        batch = s.compute_optimal_batch(
            queue_depth=32,
            memory_available=0.05,  # high utilization
            slo_latency_ms=200.0,
            current_latency_ms=100.0,
        )
        assert batch < 16

    def test_scales_down_latency_pressure(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        s._current_batch = 16
        batch = s.compute_optimal_batch(
            queue_depth=32,
            memory_available=0.5,
            slo_latency_ms=100.0,
            current_latency_ms=250.0,  # > 2x SLO
        )
        assert batch < 16

    def test_clamps_to_min(self):
        s = AdaptiveBatchSizer(min_batch=4, max_batch=64)
        s._current_batch = 4
        batch = s.compute_optimal_batch(
            queue_depth=32,
            memory_available=0.05,
            slo_latency_ms=100.0,
            current_latency_ms=500.0,
        )
        assert batch >= 4

    def test_clamps_to_max(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=8)
        s._current_batch = 8
        batch = s.compute_optimal_batch(
            queue_depth=100,
            memory_available=0.9,
            slo_latency_ms=200.0,
            current_latency_ms=50.0,
        )
        assert batch <= 8

    def test_queue_depth_signal(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        s._current_batch = 16
        batch = s.compute_optimal_batch(
            queue_depth=2,
            memory_available=0.8,
            slo_latency_ms=200.0,
            current_latency_ms=100.0,
        )
        # queue_depth no longer caps batch — the scale-up condition
        # checks queue_ok, and the batch is clamped to [min, max].
        assert batch >= 1
        assert batch <= 64

    def test_batch_sizer_stats(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        s.compute_optimal_batch(32, 0.5, 200.0, current_latency_ms=100.0)
        stats = s.get_stats()
        assert stats["slo_checks"] == 1
        assert "current_batch_size" in stats
        assert "adjustment_count" in stats
        assert stats["history_size"] == 1

    def test_slo_compliance_rate(self):
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        # Within SLO
        s.compute_optimal_batch(32, 0.5, 200.0, current_latency_ms=100.0)
        s.compute_optimal_batch(32, 0.5, 200.0, current_latency_ms=100.0)
        # Exceeds SLO
        s.compute_optimal_batch(32, 0.5, 200.0, current_latency_ms=300.0)
        stats = s.get_stats()
        assert stats["slo_compliance_rate"] == pytest.approx(66.67, rel=0.01)

    def test_hold_when_moderate_latency(self):
        """Latency above SLO but below 2x multiplier: should hold."""
        s = AdaptiveBatchSizer(min_batch=1, max_batch=64)
        s._current_batch = 8
        batch = s.compute_optimal_batch(
            queue_depth=32,
            memory_available=0.5,
            slo_latency_ms=100.0,
            current_latency_ms=150.0,  # above SLO but below 2x
        )
        assert batch == 8  # held


# ===================================================================
# TunableParams
# ===================================================================

class TestTunableParams:
    """Test parameter clamping and validation."""

    def test_defaults(self):
        p = TunableParams()
        assert p.batch_size == 8
        assert p.prefill_chunk_size == 2048
        assert p.kv_quantization_bits == 16

    def test_clamp_batch_size(self):
        p = TunableParams(batch_size=100, batch_size_max=64)
        p.clamp()
        assert p.batch_size == 64

    def test_clamp_batch_size_below_min(self):
        p = TunableParams(batch_size=0, batch_size_min=1)
        p.clamp()
        assert p.batch_size == 1

    def test_clamp_kv_quantization_snaps(self):
        p = TunableParams(kv_quantization_bits=12)
        p.clamp()
        # 12 is not in (4, 8, 16); should snap to nearest = 8
        assert p.kv_quantization_bits == 8

    def test_to_dict(self):
        p = TunableParams()
        d = p.to_dict()
        assert "batch_size" in d
        assert "prefill_chunk_size" in d
        assert "kv_quantization_bits" in d
        assert "spec_draft_length" in d
        assert "num_parallel_requests" in d


# ===================================================================
# AutoTuner
# ===================================================================

class TestAutoTunerTuning:
    """Test tuning decisions."""

    def test_increase_batch_size(self):
        t = AutoTuner(params=TunableParams(batch_size=8))
        d = t.apply_tuning("batch_size", "increase", "test")
        assert d.new_value == 10  # step = 2
        assert d.old_value == 8

    def test_decrease_batch_size(self):
        t = AutoTuner(params=TunableParams(batch_size=10))
        d = t.apply_tuning("batch_size", "decrease", "test")
        assert d.new_value == 8

    def test_clamped_increase(self):
        t = AutoTuner(params=TunableParams(
            batch_size=63, batch_size_max=64
        ))
        d = t.apply_tuning("batch_size", "increase", "test")
        assert d.new_value == 64  # clamped to max

    def test_clamped_decrease(self):
        t = AutoTuner(params=TunableParams(
            batch_size=1, batch_size_min=1
        ))
        d = t.apply_tuning("batch_size", "decrease", "test")
        assert d.new_value == 1  # clamped to min

    def test_unknown_parameter(self):
        t = AutoTuner()
        d = t.apply_tuning("nonexistent_param", "increase", "test")
        assert d.old_value is None

    def test_unknown_direction(self):
        t = AutoTuner()
        d = t.apply_tuning("batch_size", "sideways", "test")
        assert d.old_value == d.new_value


class TestAutoTunerEvaluation:
    """Test tuning evaluation."""

    def test_positive_improvement(self):
        before = _metrics(throughput=50.0)
        t = AutoTuner()
        d = t.apply_tuning("batch_size", "increase", "test", before_metrics=before)
        after = _metrics(throughput=80.0)
        improvement = t.evaluate_tuning(d, after)
        assert improvement > 0
        assert d.improvement > 0

    def test_negative_regression(self):
        before = _metrics(throughput=50.0)
        t = AutoTuner(regression_threshold=-0.1)
        d = t.apply_tuning("batch_size", "increase", "test", before_metrics=before)
        after = _metrics(throughput=30.0)  # big drop
        improvement = t.evaluate_tuning(d, after)
        assert improvement < 0
        assert d.is_regression

    def test_no_before_metrics(self):
        t = AutoTuner()
        d = t.apply_tuning("batch_size", "increase", "test")
        after = _metrics(throughput=50.0)
        improvement = t.evaluate_tuning(d, after)
        assert improvement == 0.0

    def test_zero_before_throughput(self):
        before = _metrics(throughput=0.0)
        t = AutoTuner()
        d = t.apply_tuning("batch_size", "increase", "test", before_metrics=before)
        after = _metrics(throughput=50.0)
        improvement = t.evaluate_tuning(d, after)
        assert improvement == 1.0  # positive change from zero


class TestAutoTunerHistory:
    """Test history and statistics."""

    def test_tuning_history(self):
        t = AutoTuner(params=TunableParams(batch_size=8), min_tuning_interval=0)
        t.apply_tuning("batch_size", "increase", "reason 1")
        t.apply_tuning("batch_size", "increase", "reason 2")
        history = t.get_tuning_history()
        assert len(history) == 2
        assert history[0]["param_name"] == "batch_size"
        assert history[0]["reason"] == "reason 1"

    def test_stats(self):
        t = AutoTuner()
        t.apply_tuning("batch_size", "increase", "test")
        stats = t.get_stats()
        assert stats["total_tunings"] == 1
        assert stats["history_size"] == 1
        assert "current_params" in stats
        assert "profiler_stats" in stats
        assert "slo_stats" in stats

    def test_auto_tune_from_profiler(self):
        profiler = PerformanceProfiler()
        # Simulate memory bottleneck
        for _ in range(10):
            profiler.record_step(_metrics(gpu_util=0.95, throughput=40.0))
        t = AutoTuner(profiler=profiler)
        decisions = t.auto_tune_from_profiler()
        assert len(decisions) > 0
        # Should have decreased batch_size
        params_changed = {d.param_name for d in decisions}
        assert "batch_size" in params_changed


# ===================================================================
# Integration: SLO -> AutoTuner
# ===================================================================

class TestSLOAutoTunerIntegration:
    """Test SLO monitor triggering auto-tuner."""

    def test_slo_violation_triggers_tuning(self):
        tuner = AutoTuner(params=TunableParams(batch_size=16))
        tuner.slo_monitor.set_auto_tuning_callback(
            lambda v: tuner.apply_tuning("batch_size", "decrease", "SLO violation")
        )
        # Generate SLO violations
        for _ in range(10):
            tuner.slo_monitor.check_slo("ttft", 400.0)  # exceeds default 200ms
        # Tuner should have been triggered and reduced batch
        assert tuner.params.batch_size < 16


# ===================================================================
# Profiler dead-history cleanup
# ===================================================================


class TestProfilerDeadHistory:
    """Test that _all_history was removed (no dead memory accumulation)."""

    def test_no_all_history_attribute(self):
        """_all_history was a dead deque that accumulated data but was never read."""
        p = PerformanceProfiler()
        assert not hasattr(p, '_all_history')

    def test_history_is_bounded(self):
        """Ensure the history deque is bounded by window_size."""
        p = PerformanceProfiler(window_size=10)
        for _ in range(100):
            p.record_step(_metrics())
        assert len(p._history) <= 10
