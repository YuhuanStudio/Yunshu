"""Tests for yunshu_engine.telemetry — TelemetryConfig and TelemetryCollector."""
from __future__ import annotations

import pytest

from yunshu_engine.telemetry import TelemetryCollector, TelemetryConfig

# ── TelemetryConfig ──


class TestTelemetryConfig:
    """Test TelemetryConfig construction and validation."""

    def test_defaults(self):
        cfg = TelemetryConfig()
        assert cfg.enabled is False
        assert cfg.endpoint == "http://localhost:4318/v1/metrics"
        assert cfg.sample_rate == 1.0
        assert cfg.batch_size == 100

    def test_custom_values(self):
        cfg = TelemetryConfig(
            enabled=True,
            endpoint="http://custom:9090/metrics",
            sample_rate=0.5,
            batch_size=50,
        )
        assert cfg.enabled is True
        assert cfg.endpoint == "http://custom:9090/metrics"
        assert cfg.sample_rate == 0.5
        assert cfg.batch_size == 50

    def test_sample_rate_zero(self):
        cfg = TelemetryConfig(sample_rate=0.0)
        assert cfg.sample_rate == 0.0

    def test_sample_rate_one(self):
        cfg = TelemetryConfig(sample_rate=1.0)
        assert cfg.sample_rate == 1.0

    def test_sample_rate_too_high(self):
        with pytest.raises(ValueError, match="sample_rate"):
            TelemetryConfig(sample_rate=1.5)

    def test_sample_rate_negative(self):
        with pytest.raises(ValueError, match="sample_rate"):
            TelemetryConfig(sample_rate=-0.1)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TelemetryConfig(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TelemetryConfig(batch_size=-1)


# ── TelemetryCollector — Disabled Mode ──


class TestDisabledMode:
    """When enabled=False, all methods should be no-ops."""

    def test_collect_returns_false(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=False))
        assert tc.collect("metric", 42.0) is False

    def test_collect_no_tags(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=False))
        assert tc.collect("metric", 1.0) is False

    def test_flush_returns_zero(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=False))
        assert tc.flush() == 0

    def test_pending_count_zero(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=False))
        assert tc.get_pending_count() == 0

    def test_default_config_is_disabled(self):
        """Default TelemetryConfig has enabled=False."""
        tc = TelemetryCollector()
        assert tc.collect("x", 1.0) is False
        assert tc.get_pending_count() == 0
        assert tc.flush() == 0


# ── TelemetryCollector — Enabled Mode ──


class TestEnabledCollect:
    """Test metric collection when enabled=True."""

    def test_collect_returns_true(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        assert tc.collect("test_metric", 42.0) is True

    def test_collect_with_tags(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        assert tc.collect("latency_ms", 100.0, tags={"model": "qwen"}) is True

    def test_collect_increments_pending(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("m1", 1.0)
        tc.collect("m2", 2.0)
        tc.collect("m3", 3.0)
        assert tc.get_pending_count() == 3

    def test_collect_integer_value(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("tokens", 128)
        assert tc.get_pending_count() == 1

    def test_collect_none_tags(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("metric", 1.0, tags=None)
        assert tc.get_pending_count() == 1

    def test_collect_empty_tags(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("metric", 1.0, tags={})
        assert tc.get_pending_count() == 1


# ── Sampling ──


class TestSampling:
    """Test sample_rate behavior."""

    def test_sample_rate_one_collects_all(self):
        tc = TelemetryCollector(
            TelemetryConfig(enabled=True, sample_rate=1.0)
        )
        for i in range(50):
            tc.collect("metric", float(i))
        assert tc.get_pending_count() == 50

    def test_sample_rate_zero_collects_none(self):
        tc = TelemetryCollector(
            TelemetryConfig(enabled=True, sample_rate=0.0)
        )
        for i in range(50):
            tc.collect("metric", float(i))
        assert tc.get_pending_count() == 0

    def test_sample_rate_partial(self):
        """With sample_rate=0.5, roughly half should be collected."""
        tc = TelemetryCollector(
            TelemetryConfig(enabled=True, sample_rate=0.5, batch_size=2000)
        )
        for i in range(1000):
            tc.collect("metric", float(i))
        count = tc.get_pending_count()
        # Should be roughly 500, but allow wide statistical margin
        assert 200 < count < 800


# ── Flushing ──


class TestFlush:
    """Test flush behavior."""

    def test_flush_clears_batch(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("m1", 1.0)
        tc.collect("m2", 2.0)
        assert tc.get_pending_count() == 2
        flushed = tc.flush()
        assert flushed == 2
        assert tc.get_pending_count() == 0

    def test_flush_empty_returns_zero(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        assert tc.flush() == 0

    def test_double_flush(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("m", 1.0)
        assert tc.flush() == 1
        assert tc.flush() == 0

    def test_collect_after_flush(self):
        tc = TelemetryCollector(TelemetryConfig(enabled=True))
        tc.collect("m1", 1.0)
        tc.flush()
        tc.collect("m2", 2.0)
        assert tc.get_pending_count() == 1


# ── Auto-flush on batch_size ──


class TestAutoFlush:
    """Test that auto-flush triggers when batch_size is reached."""

    def test_auto_flush_on_batch_size(self):
        tc = TelemetryCollector(
            TelemetryConfig(enabled=True, sample_rate=1.0, batch_size=5)
        )
        # Collect 4 — should not auto-flush
        for i in range(4):
            tc.collect("m", float(i))
        assert tc.get_pending_count() == 4

        # 5th metric triggers auto-flush
        tc.collect("m", 4.0)
        # After auto-flush, batch should be empty
        assert tc.get_pending_count() == 0

    def test_auto_flush_with_small_batch(self):
        tc = TelemetryCollector(
            TelemetryConfig(enabled=True, sample_rate=1.0, batch_size=2)
        )
        tc.collect("m1", 1.0)
        assert tc.get_pending_count() == 1
        tc.collect("m2", 2.0)  # triggers auto-flush
        assert tc.get_pending_count() == 0
        tc.collect("m3", 3.0)
        assert tc.get_pending_count() == 1


# ── Thread Safety ──


class TestThreadSafety:
    """Test that concurrent collect calls don't corrupt state."""

    def test_concurrent_collects(self):
        import threading

        tc = TelemetryCollector(
            TelemetryConfig(enabled=True, sample_rate=1.0, batch_size=10000)
        )
        errors = []

        def worker(start: int) -> None:
            try:
                for i in range(100):
                    tc.collect("metric", float(start + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 10 threads * 100 metrics each = 1000 total
        assert tc.get_pending_count() == 1000


# ── Config Property ──


class TestConfigProperty:
    def test_config_accessible(self):
        cfg = TelemetryConfig(enabled=True, sample_rate=0.75)
        tc = TelemetryCollector(cfg)
        assert tc.config is cfg
        assert tc.config.sample_rate == 0.75

    def test_none_config_uses_defaults(self):
        tc = TelemetryCollector(None)
        assert tc.config.enabled is False
