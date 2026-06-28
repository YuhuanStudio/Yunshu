"""Tests for C18: CPU/GPU Overlap scheduling."""
import pytest

from yunshu_engine.cpu_gpu_overlap import (
    OverlapConfig,
    OverlapMetrics,
    OverlapScheduler,
)


class TestOverlapConfig:
    def test_defaults(self):
        cfg = OverlapConfig()
        assert not cfg.enabled
        assert cfg.async_eval
        assert cfg.overlap_detokenize
        assert cfg.overlap_grammar
        assert not cfg.overlap_response_dist
        assert cfg.sync_timeout_ms == 100.0
        assert cfg.metrics_window == 100

    def test_from_env_disabled(self):
        cfg = OverlapConfig.from_env()
        assert not cfg.enabled

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_CPU_GPU_OVERLAP", "1")
        cfg = OverlapConfig.from_env()
        assert cfg.enabled

    def test_from_env_all_flags(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_CPU_GPU_OVERLAP", "1")
        monkeypatch.setenv("YUNSHU_ASYNC_EVAL", "0")
        monkeypatch.setenv("YUNSHU_OVERLAP_DETOKENIZE", "0")
        monkeypatch.setenv("YUNSHU_OVERLAP_GRAMMAR", "0")
        monkeypatch.setenv("YUNSHU_OVERLAP_RESPONSE_DIST", "1")
        cfg = OverlapConfig.from_env()
        assert cfg.enabled
        assert not cfg.async_eval
        assert not cfg.overlap_detokenize
        assert not cfg.overlap_grammar
        assert cfg.overlap_response_dist

    def test_to_dict(self):
        cfg = OverlapConfig(enabled=True)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert "async_eval" in d
        assert "overlap_detokenize" in d
        assert "overlap_grammar" in d
        assert "metrics_window" in d


class TestOverlapMetrics:
    def test_initial_state(self):
        m = OverlapMetrics()
        assert m.total_steps == 0
        assert m.steps_with_overlap == 0
        assert m.overlap_rate == 0.0
        assert m.overlap_efficiency == 0.0

    def test_record_step(self):
        m = OverlapMetrics()
        m.record_step(gpu_time_ms=10.0, cpu_overlap_time_ms=5.0, cpu_only_time_ms=2.0, had_overlap=True)
        assert m.total_steps == 1
        assert m.steps_with_overlap == 1
        assert m.overlap_rate == 1.0

    def test_overlap_rate(self):
        m = OverlapMetrics()
        m.record_step(10.0, 5.0, 2.0, True)
        m.record_step(10.0, 5.0, 2.0, False)
        assert m.total_steps == 2
        assert m.overlap_rate == 0.5

    def test_overlap_efficiency(self):
        m = OverlapMetrics()
        # 5ms CPU overlap during 10ms GPU = 50% efficiency
        m.record_step(10.0, 5.0, 2.0, True)
        assert m.overlap_efficiency == pytest.approx(0.5, abs=0.01)

    def test_efficiency_capped_at_1(self):
        m = OverlapMetrics()
        # CPU overlap > GPU time should be capped at 1.0
        m.record_step(5.0, 10.0, 2.0, True)
        assert m.overlap_efficiency == 1.0

    def test_rolling_window(self):
        m = OverlapMetrics(_window=3)
        m.record_step(10.0, 2.0, 1.0, True)
        m.record_step(10.0, 4.0, 1.0, True)
        m.record_step(10.0, 6.0, 1.0, True)
        # Window is full, avg overlap = 4.0, avg gpu = 10.0, efficiency = 0.4
        assert m.overlap_efficiency == pytest.approx(0.4, abs=0.01)

    def test_rolling_window_eviction(self):
        m = OverlapMetrics(_window=2)
        m.record_step(10.0, 2.0, 1.0, True)
        m.record_step(10.0, 4.0, 1.0, True)
        m.record_step(10.0, 8.0, 1.0, True)
        # Window evicts first entry: avg overlap = (4+8)/2 = 6.0, efficiency = 0.6
        assert m.overlap_efficiency == pytest.approx(0.6, abs=0.01)

    def test_get_stats(self):
        m = OverlapMetrics()
        m.record_step(10.0, 5.0, 2.0, True)
        stats = m.get_stats()
        assert stats["total_steps"] == 1
        assert stats["steps_with_overlap"] == 1
        assert stats["overlap_rate"] == 1.0
        assert "avg_gpu_time_ms" in stats
        assert "avg_cpu_overlap_ms" in stats

    def test_reset(self):
        m = OverlapMetrics()
        m.record_step(10.0, 5.0, 2.0, True)
        m.reset()
        assert m.total_steps == 0
        assert m.steps_with_overlap == 0
        assert m.gpu_time_ms == 0.0


class TestOverlapScheduler:
    def test_creation_default(self):
        scheduler = OverlapScheduler()
        assert not scheduler.config.enabled
        assert not scheduler.is_gpu_pending

    def test_creation_enabled(self):
        cfg = OverlapConfig(enabled=True)
        scheduler = OverlapScheduler(cfg)
        assert scheduler.config.enabled

    def test_step_disabled(self):
        """When disabled, step() delegates to scheduler directly."""
        scheduler = OverlapScheduler(OverlapConfig(enabled=False))
        call_count = 0

        class FakeScheduler:
            def step(self):
                nonlocal call_count
                call_count += 1
                return "result"

        result = scheduler.step(FakeScheduler())
        assert result == "result"
        assert call_count == 1

    def test_step_async_then_sync(self):
        """Async/sync pattern: launch GPU, then synchronize."""
        cfg = OverlapConfig(enabled=True, async_eval=False)
        scheduler = OverlapScheduler(cfg)

        class FakeScheduler:
            def step(self):
                return FakeOutput()

        class FakeOutput:
            outputs = []

        scheduler.step_async(FakeScheduler())
        assert scheduler.is_gpu_pending

        output = scheduler.step_sync()
        assert output is not None
        assert not scheduler.is_gpu_pending

    def test_get_stats(self):
        scheduler = OverlapScheduler(OverlapConfig(enabled=True))
        stats = scheduler.get_stats()
        assert "config" in stats
        assert "metrics" in stats
        assert "gpu_pending" in stats
        assert stats["metrics"]["total_steps"] == 0

    def test_reset(self):
        cfg = OverlapConfig(enabled=True, async_eval=False)
        scheduler = OverlapScheduler(cfg)

        class FakeScheduler:
            def step(self):
                return type("Out", (), {"outputs": []})()

        scheduler.step_async(FakeScheduler())
        scheduler.step_sync()

        scheduler.reset()
        assert not scheduler.is_gpu_pending
        assert scheduler.metrics.total_steps == 0

    def test_cpu_work_extraction_from_output(self):
        """CPU work items should be extracted from scheduler outputs."""
        scheduler = OverlapScheduler(OverlapConfig(enabled=True))

        class FakeDetokenizer:
            def add_token(self, tid):
                pass

        class FakeResponse:
            _detokenizer = None
            _last_token_id = None

        resp = FakeResponse()
        resp._detokenizer = FakeDetokenizer()
        resp._last_token_id = 42

        class FakeOutput:
            outputs = [resp]

        work = scheduler._extract_cpu_work(FakeOutput())
        assert len(work) == 1
        assert work[0]["kind"] == "detokenize"
        assert work[0]["token_id"] == 42

    def test_cpu_work_extraction_empty(self):
        scheduler = OverlapScheduler(OverlapConfig(enabled=True))
        work = scheduler._extract_cpu_work(None)
        assert work == []

    def test_detokenize_overlap_disabled(self):
        """When overlap_detokenize is disabled, no detokenize work is extracted."""
        cfg = OverlapConfig(enabled=True, overlap_detokenize=False)
        scheduler = OverlapScheduler(cfg)

        class FakeOutput:
            outputs = []
            def _detokenizer():
                return None
            _last_token_id = 42

        work = scheduler._extract_cpu_work(FakeOutput())
        assert len(work) == 0

    def test_run_cpu_postprocess_detokenize(self):
        tokens = []

        class FakeDetokenizer:
            def add_token(self, tid):
                tokens.append(tid)

        scheduler = OverlapScheduler(OverlapConfig(enabled=True))
        scheduler._run_cpu_postprocess([{
            "kind": "detokenize",
            "detokenizer": FakeDetokenizer(),
            "token_id": 99,
        }])
        assert tokens == [99]

    def test_run_cpu_postprocess_grammar(self):
        checks = []

        class FakeChecker:
            def validate(self, text):
                checks.append(text)

        scheduler = OverlapScheduler(OverlapConfig(enabled=True))
        scheduler._run_cpu_postprocess([{
            "kind": "grammar",
            "checker": FakeChecker(),
            "text": "hello",
        }])
        assert checks == ["hello"]

    def test_collect_arrays(self):
        """Collect mx.arrays from nested structures."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        a = mx.array([1.0, 2.0])
        b = mx.array([3.0])
        nested = {"data": [a, {"inner": b, "not_array": "text"}, "skip"]}
        arrays = OverlapScheduler._collect_arrays(nested)
        assert len(arrays) >= 2

    def test_collect_arrays_depth_limit(self):
        """Deep nesting should be truncated."""
        result = OverlapScheduler._collect_arrays({"a": {"b": {"c": {"d": {"e": "deep"}}}}}, depth=5)
        assert result == []

    def test_synchronize_gpu(self):
        """Synchronize should not raise."""
        OverlapScheduler._synchronize_gpu()


class TestOverlapSchedulerMetrics:
    def test_metrics_after_steps(self):
        cfg = OverlapConfig(enabled=True, async_eval=False)
        scheduler = OverlapScheduler(cfg)

        class FakeScheduler:
            def step(self):
                return type("Out", (), {"outputs": []})()

        for _ in range(5):
            scheduler.step_async(FakeScheduler())
            scheduler.step_sync()

        stats = scheduler.metrics.get_stats()
        assert stats["total_steps"] == 5
        assert stats["overlap_rate"] >= 0.0


class TestOverlapIntegration:
    def test_config_from_env_roundtrip(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_CPU_GPU_OVERLAP", "1")
        cfg = OverlapConfig.from_env()
        scheduler = OverlapScheduler(cfg)
        assert scheduler.config.enabled
        stats = scheduler.get_stats()
        assert stats["config"]["enabled"] is True

    def test_engine_core_config_has_overlap(self):
        from yunshu_engine.engine_core import EngineCoreConfig
        cfg = EngineCoreConfig()
        assert hasattr(cfg, "enable_cpu_gpu_overlap")
        assert not cfg.enable_cpu_gpu_overlap
