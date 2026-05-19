"""Tests for AdaptiveBatchScheduler — batch size and prefill chunk computation."""

import pytest

from yunshu_engine.adaptive_batch import (
    AdaptiveBatchConfig,
    AdaptiveBatchScheduler,
)


class TestAdaptiveBatchConfig:
    def test_defaults(self):
        cfg = AdaptiveBatchConfig()
        assert cfg.min_batch == 1
        assert cfg.max_batch == 32
        assert cfg.target_latency_ms == 100.0
        assert cfg.memory_threshold == 0.85
        assert cfg.ewma_alpha == 0.3

    def test_custom_config(self):
        cfg = AdaptiveBatchConfig(
            min_batch=2,
            max_batch=64,
            target_latency_ms=50.0,
            memory_threshold=0.90,
        )
        assert cfg.min_batch == 2
        assert cfg.max_batch == 64
        assert cfg.target_latency_ms == 50.0
        assert cfg.memory_threshold == 0.90


class TestComputeBatchSize:
    """Test batch size computation under various load conditions."""

    def test_defaults_start_at_min(self):
        scheduler = AdaptiveBatchScheduler()
        batch = scheduler.compute_batch_size(
            current_memory_usage=0.5,
            avg_latency_ms=50.0,
            pending_count=16,
        )
        # No memory or latency pressure; should try to scale up
        # but current batch is 1, scaled up by 1.5 -> 1, clamped to min
        assert batch >= 1

    def test_memory_pressure_scales_down(self):
        cfg = AdaptiveBatchConfig(min_batch=1, max_batch=32)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 16

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.95,  # above 0.85 threshold
            avg_latency_ms=50.0,
            pending_count=32,
        )
        assert batch < 16
        assert batch >= cfg.min_batch

    def test_high_memory_scales_down_aggressively(self):
        cfg = AdaptiveBatchConfig(min_batch=1, max_batch=32)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 16

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.99,  # way above threshold
            avg_latency_ms=50.0,
            pending_count=32,
        )
        assert batch < 16
        assert batch >= cfg.min_batch

    def test_latency_pressure_scales_down(self):
        cfg = AdaptiveBatchConfig(min_batch=1, max_batch=32, target_latency_ms=100.0)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 16

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.5,
            avg_latency_ms=300.0,  # 3x target
            pending_count=32,
        )
        assert batch < 16

    def test_moderate_latency_holds(self):
        """Latency above target but below 2x: should hold current."""
        cfg = AdaptiveBatchConfig(min_batch=4, max_batch=32, target_latency_ms=100.0)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 8

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.5,
            avg_latency_ms=150.0,  # between target and 2x
            pending_count=32,
        )
        # Should stay at 8 (or be clamped by pending_count)
        assert batch == 8

    def test_low_latency_low_memory_scales_up(self):
        cfg = AdaptiveBatchConfig(min_batch=1, max_batch=32, target_latency_ms=100.0)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 4

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.3,  # well below threshold * 0.75
            avg_latency_ms=50.0,  # below target
            pending_count=32,
        )
        # Should scale up: 4 * 1.5 = 6
        assert batch > 4
        assert batch <= 32

    def test_clamps_to_pending_count(self):
        cfg = AdaptiveBatchConfig(min_batch=1, max_batch=32)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 16

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.3,
            avg_latency_ms=50.0,
            pending_count=2,  # only 2 pending
        )
        # Should not exceed pending_count
        assert batch <= 2

    def test_clamps_to_min_batch(self):
        cfg = AdaptiveBatchConfig(min_batch=4, max_batch=32)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 4

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.99,
            avg_latency_ms=500.0,
            pending_count=32,
        )
        # Should never go below min_batch
        assert batch >= 4

    def test_clamps_to_max_batch(self):
        cfg = AdaptiveBatchConfig(min_batch=1, max_batch=8)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 8

        batch = scheduler.compute_batch_size(
            current_memory_usage=0.3,
            avg_latency_ms=10.0,
            pending_count=100,
        )
        # Should not exceed max_batch
        assert batch <= 8

    def test_zero_pending_returns_zero(self):
        scheduler = AdaptiveBatchScheduler()
        batch = scheduler.compute_batch_size(
            current_memory_usage=0.3,
            avg_latency_ms=50.0,
            pending_count=0,
        )
        # No pending requests → batch size 0 (don't schedule empty batches)
        assert batch == 0

    def test_batch_size_is_integer(self):
        scheduler = AdaptiveBatchScheduler()
        for memory in [0.1, 0.5, 0.9]:
            for latency in [10.0, 100.0, 500.0]:
                for pending in [0, 1, 10, 100]:
                    batch = scheduler.compute_batch_size(memory, latency, pending)
                    assert isinstance(batch, int)


class TestComputePrefillChunkSize:
    """Test prefill chunk size computation."""

    def test_short_sequence_returns_full_length(self):
        scheduler = AdaptiveBatchScheduler()
        chunk = scheduler.compute_prefill_chunk_size(
            seq_length=50,
            available_memory=0.5,
        )
        assert chunk == 50  # shorter than MIN_CHUNK, return as-is

    def test_high_memory_gives_max_chunk(self):
        scheduler = AdaptiveBatchScheduler()
        chunk = scheduler.compute_prefill_chunk_size(
            seq_length=10000,
            available_memory=0.8,
        )
        assert chunk == 4096  # MAX_CHUNK

    def test_medium_memory_proportional_chunk(self):
        scheduler = AdaptiveBatchScheduler()
        chunk = scheduler.compute_prefill_chunk_size(
            seq_length=10000,
            available_memory=0.35,  # between 0.25 and 0.5
        )
        # Should be proportional: (0.35 / 0.5) * 4096 = 2867
        assert 128 <= chunk <= 4096
        assert chunk < 4096

    def test_low_memory_gives_min_chunk(self):
        scheduler = AdaptiveBatchScheduler()
        chunk = scheduler.compute_prefill_chunk_size(
            seq_length=10000,
            available_memory=0.05,  # below 0.1
        )
        assert chunk == 128  # MIN_CHUNK

    def test_chunk_does_not_exceed_seq_length(self):
        scheduler = AdaptiveBatchScheduler()
        chunk = scheduler.compute_prefill_chunk_size(
            seq_length=500,
            available_memory=0.8,
        )
        assert chunk <= 500

    def test_exact_min_chunk_boundary(self):
        scheduler = AdaptiveBatchScheduler()
        chunk = scheduler.compute_prefill_chunk_size(
            seq_length=128,
            available_memory=0.8,
        )
        assert chunk == 128  # exactly at boundary

    def test_chunk_is_always_positive(self):
        scheduler = AdaptiveBatchScheduler()
        for seq in [1, 100, 1000, 100000]:
            for mem in [0.01, 0.1, 0.5, 0.9]:
                chunk = scheduler.compute_prefill_chunk_size(seq, mem)
                assert chunk >= 1
                assert isinstance(chunk, int)


class TestUpdateMetrics:
    """Test EWMA metric tracking."""

    def test_first_update_initializes(self):
        scheduler = AdaptiveBatchScheduler()
        scheduler.update_metrics(latency_ms=80.0, memory_usage=0.6, batch_size=8)
        stats = scheduler.get_stats()
        assert stats["avg_latency_ms"] == 80.0
        assert stats["avg_memory_usage"] == 0.6
        assert stats["total_updates"] == 1

    def test_ewma_smoothing(self):
        cfg = AdaptiveBatchConfig(ewma_alpha=0.5)
        scheduler = AdaptiveBatchScheduler(cfg)

        scheduler.update_metrics(latency_ms=100.0, memory_usage=0.5, batch_size=8)
        scheduler.update_metrics(latency_ms=200.0, memory_usage=0.7, batch_size=8)

        stats = scheduler.get_stats()
        # EWMA: 0.5*200 + 0.5*100 = 150
        assert stats["avg_latency_ms"] == 150.0
        # EWMA: 0.5*0.7 + 0.5*0.5 = 0.6
        assert stats["avg_memory_usage"] == 0.6
        assert stats["total_updates"] == 2

    def test_ewma_low_alpha_smooths(self):
        cfg = AdaptiveBatchConfig(ewma_alpha=0.1)
        scheduler = AdaptiveBatchScheduler(cfg)

        scheduler.update_metrics(latency_ms=100.0, memory_usage=0.5, batch_size=4)
        scheduler.update_metrics(latency_ms=200.0, memory_usage=0.9, batch_size=4)

        stats = scheduler.get_stats()
        # EWMA: 0.1*200 + 0.9*100 = 110
        assert stats["avg_latency_ms"] == 110.0
        # EWMA: 0.1*0.9 + 0.9*0.5 = 0.54
        assert abs(stats["avg_memory_usage"] - 0.54) < 0.001

    def test_multiple_updates(self):
        scheduler = AdaptiveBatchScheduler()
        for i in range(100):
            scheduler.update_metrics(
                latency_ms=50.0 + i,
                memory_usage=0.5,
                batch_size=8,
            )
        stats = scheduler.get_stats()
        assert stats["total_updates"] == 100
        # Should converge toward recent values
        assert stats["avg_latency_ms"] > 50.0


class TestGetStats:
    """Test statistics reporting."""

    def test_initial_stats(self):
        scheduler = AdaptiveBatchScheduler()
        stats = scheduler.get_stats()
        assert stats["current_batch_size"] == 1
        assert stats["avg_latency_ms"] == 0.0
        assert stats["avg_memory_usage"] == 0.0
        assert stats["total_updates"] == 0
        assert "config" in stats
        assert stats["config"]["min_batch"] == 1
        assert stats["config"]["max_batch"] == 32

    def test_stats_after_updates(self):
        scheduler = AdaptiveBatchScheduler()
        scheduler.update_metrics(100.0, 0.7, 16)
        scheduler.compute_batch_size(0.7, 100.0, 32)

        stats = scheduler.get_stats()
        assert stats["total_updates"] == 1
        assert stats["avg_latency_ms"] == 100.0

    def test_stats_values_are_rounded(self):
        scheduler = AdaptiveBatchScheduler()
        scheduler.update_metrics(123.456, 0.789123, 8)
        stats = scheduler.get_stats()
        # avg_latency_ms rounded to 2 decimal places
        assert stats["avg_latency_ms"] == 123.46
        # avg_memory_usage rounded to 4 decimal places
        assert stats["avg_memory_usage"] == 0.7891


class TestReset:
    """Test scheduler state reset."""

    def test_reset_clears_state(self):
        scheduler = AdaptiveBatchScheduler()
        scheduler.update_metrics(100.0, 0.8, 16)
        scheduler.compute_batch_size(0.8, 100.0, 32)

        scheduler.reset()
        stats = scheduler.get_stats()
        assert stats["current_batch_size"] == 1
        assert stats["avg_latency_ms"] == 0.0
        assert stats["avg_memory_usage"] == 0.0
        assert stats["total_updates"] == 0

    def test_reset_preserves_config(self):
        cfg = AdaptiveBatchConfig(min_batch=4, max_batch=64)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler.update_metrics(100.0, 0.8, 16)
        scheduler.reset()

        assert scheduler.config.min_batch == 4
        assert scheduler.config.max_batch == 64


class TestIntegration:
    """Integration scenarios: multiple compute + update cycles."""

    def test_adapts_to_increasing_load(self):
        scheduler = AdaptiveBatchScheduler()
        pending = 32

        # Start: low load
        batch = scheduler.compute_batch_size(0.3, 30.0, pending)
        scheduler.update_metrics(30.0, 0.3, batch)

        # Load increases
        batch = scheduler.compute_batch_size(0.7, 80.0, pending)
        scheduler.update_metrics(80.0, 0.7, batch)

        # Load spikes high
        batch = scheduler.compute_batch_size(0.92, 250.0, pending)
        assert batch < scheduler.config.max_batch

    def test_adapts_to_decreasing_load(self):
        cfg = AdaptiveBatchConfig(min_batch=2, max_batch=32)
        scheduler = AdaptiveBatchScheduler(cfg)
        scheduler._state.current_batch_size = 4

        # High load initially
        batch = scheduler.compute_batch_size(0.9, 200.0, 32)

        # Load drops
        batch = scheduler.compute_batch_size(0.2, 30.0, 32)
        scheduler.update_metrics(30.0, 0.2, batch)

        # Should be scaling up
        batch2 = scheduler.compute_batch_size(0.2, 30.0, 32)
        assert batch2 >= batch

    def test_prefill_chunk_scales_with_memory(self):
        scheduler = AdaptiveBatchScheduler()
        seq = 8192

        # High memory available
        chunk_high = scheduler.compute_prefill_chunk_size(seq, 0.8)

        # Low memory available
        chunk_low = scheduler.compute_prefill_chunk_size(seq, 0.1)

        assert chunk_high >= chunk_low
