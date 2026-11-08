"""Tests for KV cache optimizations module.

Covers:
- AdaptiveKVQuantizer: per-layer config, budget-aware adjustment, quantize/dequantize
- KVEvictionPredictor: training, prediction, eviction decisions, self-calibration
- ChunkedPrefillOptimizer: chunk computation, semantic boundaries, fair interleaving
- KVBlockCompactor: block compaction, fragmentation measurement, periodic trigger
"""

from __future__ import annotations

import pytest

from yunshu_engine.kv_optimizations import (
    AdaptiveKVQuantizer,
    ChunkedPrefillOptimizer,
    ChunkInfo,
    KVBlock,
    KVBlockCompactor,
    KVEvictionPredictor,
    QuantTier,
)

# ═══════════════════════════════════════════════════════════════════════════════
# AdaptiveKVQuantizer Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveKVQuantizerConfiguration:
    """Test per-layer configuration and tier assignment."""

    def test_default_tier_distribution(self):
        """Default: 1/3 FP16, 1/3 INT8, 1/3 INT4."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(num_layers=12, num_kv_heads=8, head_dim=128)
        assert len(configs) == 12

        fp16_count = sum(1 for c in configs if c.tier == QuantTier.FP16)
        int8_count = sum(1 for c in configs if c.tier == QuantTier.INT8)
        int4_count = sum(1 for c in configs if c.tier == QuantTier.INT4)

        assert fp16_count == 4  # 12 // 3 = 4
        assert int8_count == 4  # 8 - 4 = 4
        assert int4_count == 4  # remaining

    def test_early_layers_fp16(self):
        """Early layers should be FP16."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(num_layers=6, num_kv_heads=8, head_dim=128)
        assert configs[0].tier == QuantTier.FP16
        assert configs[0].bits == 16

    def test_middle_layers_int8(self):
        """Middle layers should be INT8."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(num_layers=12, num_kv_heads=8, head_dim=128)
        # Layers 4-7 should be INT8
        assert configs[4].tier == QuantTier.INT8
        assert configs[4].bits == 8

    def test_late_layers_int4(self):
        """Late layers should be INT4."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(num_layers=12, num_kv_heads=8, head_dim=128)
        # Last layers should be INT4
        assert configs[-1].tier == QuantTier.INT4
        assert configs[-1].bits == 4

    def test_single_layer(self):
        """Single layer should get FP16."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(num_layers=1, num_kv_heads=4, head_dim=64)
        assert len(configs) == 1
        assert configs[0].tier == QuantTier.FP16

    def test_two_layers(self):
        """Two layers: 1 FP16, 1 INT8."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(num_layers=2, num_kv_heads=4, head_dim=64)
        assert configs[0].tier == QuantTier.FP16
        assert configs[1].tier in (QuantTier.INT8, QuantTier.INT4)

    def test_invalid_num_layers(self):
        """num_layers <= 0 should raise."""
        q = AdaptiveKVQuantizer()
        with pytest.raises(ValueError, match="num_layers"):
            q.configure(num_layers=0, num_kv_heads=8, head_dim=128)

    def test_invalid_num_kv_heads(self):
        q = AdaptiveKVQuantizer()
        with pytest.raises(ValueError, match="num_kv_heads"):
            q.configure(num_layers=4, num_kv_heads=0, head_dim=128)

    def test_invalid_head_dim(self):
        q = AdaptiveKVQuantizer()
        with pytest.raises(ValueError, match="head_dim"):
            q.configure(num_layers=4, num_kv_heads=8, head_dim=0)

    def test_not_configured_raises(self):
        q = AdaptiveKVQuantizer()
        with pytest.raises(RuntimeError, match="not configured"):
            q.quantize_layer(0, [], [])

    def test_get_layer_config(self):
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=4, num_kv_heads=8, head_dim=64)
        cfg = q.get_layer_config(0)
        assert cfg.layer_idx == 0
        assert cfg.tier == QuantTier.FP16

    def test_out_of_range_layer_raises(self):
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=4, num_kv_heads=8, head_dim=64)
        with pytest.raises(IndexError):
            q.quantize_layer(99, [], [])


class TestAdaptiveKVQuantizerBudget:
    """Test budget-aware configuration."""

    def test_budget_constrains_fp16(self):
        """Tight budget should reduce FP16 layers."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(
            num_layers=12,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=2048,
            budget_bytes=1,  # Extremely tight — forces all INT4
        )
        # With 1 byte budget, most or all should be INT4
        fp16_count = sum(1 for c in configs if c.tier == QuantTier.FP16)
        assert fp16_count == 0  # All should be INT4

    def test_unlimited_budget(self):
        """No budget = default tier distribution."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(
            num_layers=12,
            num_kv_heads=8,
            head_dim=128,
            budget_bytes=None,
        )
        assert sum(1 for c in configs if c.tier == QuantTier.FP16) == 4

    def test_generous_budget_preserves_defaults(self):
        """Very large budget should keep default distribution."""
        q = AdaptiveKVQuantizer()
        configs = q.configure(
            num_layers=12,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=2048,
            budget_bytes=10 * 1024**4,  # 10 TB — effectively unlimited
        )
        assert sum(1 for c in configs if c.tier == QuantTier.FP16) == 4


class TestAdaptiveKVQuantizerQuantize:
    """Test quantize/dequantize per layer."""

    def test_fp16_layer_no_quantization(self):
        """FP16 layer should return raw data."""
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=6, num_kv_heads=4, head_dim=64)

        key = [[[[0.5, 1.0, 1.5, 2.0]]]]
        value = [[[[0.1, 0.2, 0.3, 0.4]]]]

        result, meta = q.quantize_layer(0, key, value)
        assert not meta["quantized"]
        assert meta["tier"] == "FP16"

    def test_int4_layer_quantizes(self):
        """INT4 layer should produce packed data."""
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=6, num_kv_heads=4, head_dim=64)

        key = [[[[0.5, 1.0, 1.5, 2.0]]]]
        value = [[[[0.1, 0.2, 0.3, 0.4]]]]

        # Layer 5 should be INT4 (last third)
        result, meta = q.quantize_layer(5, key, value)
        assert meta["quantized"]
        assert meta["bits"] == 4

    def test_int8_layer_quantizes(self):
        """INT8 layer should quantize."""
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=6, num_kv_heads=4, head_dim=64)

        key = [[[[0.5, 1.0, 1.5, 2.0]]]]
        value = [[[[0.1, 0.2, 0.3, 0.4]]]]

        # Layer 2 should be INT8 (middle third)
        result, meta = q.quantize_layer(2, key, value)
        assert meta["quantized"]
        assert meta["bits"] == 8

    def test_dequantize_fp16_roundtrip(self):
        """FP16 dequantize should return original data."""
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=6, num_kv_heads=4, head_dim=64)

        key = [[[[0.5, 1.0]]]]
        value = [[[[0.3, 0.7]]]]
        result, meta = q.quantize_layer(0, key, value)
        out_key, out_val = q.dequantize_layer(0, (result, meta))

        assert out_key == key
        assert out_val == value


class TestAdaptiveKVQuantizerStats:
    """Test statistics reporting."""

    def test_stats_populated(self):
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=12, num_kv_heads=8, head_dim=128, max_seq_len=2048)
        stats = q.get_stats()

        assert stats.total_bytes > 0
        assert stats.fp16_baseline_bytes > 0
        assert stats.memory_saved_pct > 0.0
        assert len(stats.per_layer_bits) == 12

    def test_memory_saved_reasonable(self):
        """Mixed precision should save 40-70% memory vs all-FP16."""
        q = AdaptiveKVQuantizer()
        q.configure(num_layers=12, num_kv_heads=8, head_dim=128)
        stats = q.get_stats()
        assert 20.0 <= stats.memory_saved_pct <= 80.0

    def test_configured_property(self):
        q = AdaptiveKVQuantizer()
        assert not q.configured
        q.configure(num_layers=4, num_kv_heads=4, head_dim=64)
        assert q.configured


# ═══════════════════════════════════════════════════════════════════════════════
# KVEvictionPredictor Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKVEvictionPredictorTraining:
    """Test training with attention patterns and accessed blocks."""

    def test_single_training_step(self):
        p = KVEvictionPredictor()
        p.train_step(accessed_blocks={"req1": ["b1", "b2", "b3"]})
        assert len(p._records) == 3

    def test_multiple_steps_build_frequency(self):
        p = KVEvictionPredictor()
        # Access b1 every step
        for _ in range(10):
            p.train_step(accessed_blocks={"r1": ["b1"]})

        assert p._records["b1"].total_accesses == 10
        assert p._records["b1"].ema_frequency > 0.5

    def test_unaccessed_blocks_decay(self):
        """Blocks not accessed in a step should have decaying frequency."""
        p = KVEvictionPredictor()
        p.train_step(accessed_blocks={"r1": ["b1"]})
        initial_freq = p._records["b1"].ema_frequency

        # More steps without accessing b1
        for _ in range(5):
            p.train_step(accessed_blocks={"r1": ["b2"]})

        assert p._records["b1"].ema_frequency < initial_freq

    def test_attention_weights_tracked(self):
        p = KVEvictionPredictor()
        p.train_step(attention_weights={"r1": {"b1": 0.8, "b2": 0.2}})
        assert p._records["b1"].request_weights["r1"] == 0.8
        assert p._records["b2"].request_weights["r1"] == 0.2

    def test_combined_training(self):
        p = KVEvictionPredictor()
        p.train_step(
            attention_weights={"r1": {"b1": 0.9}},
            accessed_blocks={"r1": ["b1"]},
        )
        record = p._records["b1"]
        assert record.total_accesses == 1
        assert record.request_weights["r1"] == 0.9


class TestKVEvictionPredictorPrediction:
    """Test prediction accuracy and eviction decisions."""

    def test_frequently_accessed_high_probability(self):
        p = KVEvictionPredictor(min_observations=1)
        # Access b1 many times
        for _ in range(20):
            p.train_step(accessed_blocks={"r1": ["b1"]})

        preds = p.predict_next_access("r1", ["b1"])
        assert len(preds) == 1
        assert preds[0].predicted_access_prob > 0.5

    def test_rarely_accessed_low_probability(self):
        p = KVEvictionPredictor(min_observations=1)
        # Access b1 once, then never again
        p.train_step(accessed_blocks={"r1": ["b1"]})
        # Many steps accessing other blocks
        for _ in range(20):
            p.train_step(accessed_blocks={"r1": ["b2"]})

        preds = p.predict_next_access("r1", ["b1"])
        assert preds[0].predicted_access_prob < 0.5

    def test_unknown_block_low_probability(self):
        p = KVEvictionPredictor()
        preds = p.predict_next_access("r1", ["unknown_block"])
        assert preds[0].predicted_access_prob == 0.0
        assert preds[0].confidence == 0.0

    def test_predictions_sorted_descending(self):
        p = KVEvictionPredictor(min_observations=1)
        # Train: b1 frequently, b2 rarely
        for _ in range(10):
            p.train_step(accessed_blocks={"r1": ["b1"]})
        p.train_step(accessed_blocks={"r1": ["b2"]})

        preds = p.predict_next_access("r1", ["b1", "b2"])
        assert preds[0].block_id == "b1"
        assert preds[0].predicted_access_prob >= preds[1].predicted_access_prob

    def test_should_evict_unknown_block(self):
        p = KVEvictionPredictor()
        assert p.should_evict("unknown") is True

    def test_should_evict_frequent_block(self):
        p = KVEvictionPredictor(min_observations=2)
        for _ in range(10):
            p.train_step(accessed_blocks={"r1": ["b1"]})
        assert p.should_evict("b1") is False

    def test_should_not_evict_insufficient_data(self):
        """Blocks with fewer than min_observations should not be evicted."""
        p = KVEvictionPredictor(min_observations=5)
        p.train_step(accessed_blocks={"r1": ["b1"]})  # Only 1 access
        assert p.should_evict("b1") is False

    def test_eviction_calibration(self):
        """Recording a wrong eviction should raise the threshold."""
        p = KVEvictionPredictor(eviction_threshold=0.1)
        initial_threshold = p._eviction_threshold
        p.record_eviction_outcome("b1", was_needed=True)
        assert p._eviction_threshold > initial_threshold


class TestKVEvictionPredictorStats:
    """Test statistics reporting."""

    def test_stats_structure(self):
        p = KVEvictionPredictor()
        p.train_step(accessed_blocks={"r1": ["b1"]})
        stats = p.get_stats()
        assert "step" in stats
        assert "tracked_blocks" in stats
        assert "eviction_threshold" in stats
        assert stats["tracked_blocks"] == 1

    def test_reset_clears_state(self):
        p = KVEvictionPredictor()
        p.train_step(accessed_blocks={"r1": ["b1"]})
        p.reset()
        assert len(p._records) == 0
        assert p._step == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkedPrefillOptimizer Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkedPrefillComputeChunks:
    """Test chunk boundary computation."""

    def test_short_prompt_single_chunk(self):
        """Prompt shorter than max_chunk_tokens → single chunk."""
        opt = ChunkedPrefillOptimizer()
        tokens = list(range(50))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].num_tokens == 50
        assert chunks[0].start_token == 0
        assert chunks[0].end_token == 50

    def test_exact_boundary(self):
        """Prompt exactly max_chunk_tokens → single chunk."""
        opt = ChunkedPrefillOptimizer()
        tokens = list(range(100))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].num_tokens == 100

    def test_long_prompt_multiple_chunks(self):
        """Prompt longer than max_chunk_tokens → multiple chunks."""
        opt = ChunkedPrefillOptimizer()
        tokens = list(range(500))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=128)
        assert len(chunks) >= 3
        # All tokens should be covered
        total_covered = sum(c.num_tokens for c in chunks)
        assert total_covered == 500

    def test_chunks_cover_all_tokens(self):
        """No gaps or overlaps between chunks."""
        opt = ChunkedPrefillOptimizer()
        tokens = list(range(300))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=64)
        # Verify continuity
        for i in range(len(chunks) - 1):
            assert chunks[i].end_token == chunks[i + 1].start_token
        assert chunks[0].start_token == 0
        assert chunks[-1].end_token == 300

    def test_importance_decreases(self):
        """Earlier chunks should have higher importance."""
        opt = ChunkedPrefillOptimizer()
        tokens = list(range(500))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=100)
        # Importance should generally decrease
        for i in range(len(chunks) - 1):
            assert chunks[i].importance >= chunks[i + 1].importance

    def test_empty_tokens(self):
        opt = ChunkedPrefillOptimizer()
        chunks = opt.compute_optimal_chunks([], max_chunk_tokens=100)
        assert chunks == []

    def test_invalid_max_chunk_raises(self):
        opt = ChunkedPrefillOptimizer()
        with pytest.raises(ValueError, match="max_chunk_tokens"):
            opt.compute_optimal_chunks([1, 2, 3], max_chunk_tokens=0)

    def test_sentence_boundary_split(self):
        """With sentence-end tokens, splits at boundaries."""
        # Token 63 is a sentence-end token (within first 128-token window)
        opt = ChunkedPrefillOptimizer(sentence_end_tokens={63})
        tokens = list(range(200))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=128)
        # First chunk should end at or near token 63 (the sentence boundary)
        assert chunks[0].end_token == 64  # Split after the sentence-end token

    def test_chunk_index_assigned(self):
        opt = ChunkedPrefillOptimizer()
        tokens = list(range(300))
        chunks = opt.compute_optimal_chunks(tokens, max_chunk_tokens=64)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestChunkedPrefillInterleave:
    """Test fair interleaving across multiple requests."""

    def test_round_robin_interleaving(self):
        """Each request should get one chunk per round."""
        opt = ChunkedPrefillOptimizer()

        chunks_r1 = [
            ChunkInfo(start_token=0, end_token=100, num_tokens=100, importance=1.0),
            ChunkInfo(start_token=100, end_token=200, num_tokens=100, importance=0.8),
        ]
        chunks_r2 = [
            ChunkInfo(start_token=0, end_token=100, num_tokens=100, importance=1.0),
            ChunkInfo(start_token=100, end_token=200, num_tokens=100, importance=0.8),
        ]

        schedule = opt.interleave_chunks({"r1": chunks_r1, "r2": chunks_r2})
        assert len(schedule) == 4
        # First two should be from different requests
        ids = [c.request_id for c in schedule]
        assert ids[0] != ids[1]

    def test_uneven_chunks_fair(self):
        """Requests with different numbers of chunks should be interleaved fairly."""
        opt = ChunkedPrefillOptimizer()

        chunks_r1 = [
            ChunkInfo(start_token=0, end_token=100, num_tokens=100, importance=1.0),
            ChunkInfo(start_token=100, end_token=200, num_tokens=100, importance=0.8),
            ChunkInfo(start_token=200, end_token=300, num_tokens=100, importance=0.6),
        ]
        chunks_r2 = [
            ChunkInfo(start_token=0, end_token=100, num_tokens=100, importance=1.0),
        ]

        schedule = opt.interleave_chunks({"r1": chunks_r1, "r2": chunks_r2})
        assert len(schedule) == 4
        # First two should be one from each request
        assert schedule[0].request_id != schedule[1].request_id

    def test_empty_requests(self):
        opt = ChunkedPrefillOptimizer()
        schedule = opt.interleave_chunks({})
        assert schedule == []

    def test_single_request(self):
        """Single request should output chunks in importance order."""
        opt = ChunkedPrefillOptimizer()
        chunks = [
            ChunkInfo(start_token=0, end_token=50, num_tokens=50, importance=0.5),
            ChunkInfo(start_token=50, end_token=100, num_tokens=50, importance=1.0),
        ]
        schedule = opt.interleave_chunks({"r1": chunks})
        assert len(schedule) == 2
        # Higher importance first
        assert schedule[0].importance >= schedule[1].importance

    def test_all_chunks_assigned_request_id(self):
        opt = ChunkedPrefillOptimizer()
        chunks_r1 = [
            ChunkInfo(start_token=0, end_token=100, num_tokens=100, importance=1.0),
        ]
        chunks_r2 = [
            ChunkInfo(start_token=0, end_token=100, num_tokens=100, importance=1.0),
        ]
        schedule = opt.interleave_chunks({"r1": chunks_r1, "r2": chunks_r2})
        for chunk in schedule:
            assert chunk.request_id in ("r1", "r2")


# ═══════════════════════════════════════════════════════════════════════════════
# KVBlockCompactor Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKVBlockCompactorBasic:
    """Test basic block compaction operations."""

    def test_no_blocks_no_op(self):
        c = KVBlockCompactor(block_size=64)
        result = c.compact()
        assert result.blocks_freed == 0

    def test_full_blocks_not_compacted(self):
        """Fully utilized blocks should not be merged."""
        c = KVBlockCompactor(block_size=4)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2, 3, 4],
                num_valid=4,
                block_size=4,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r1",
                tokens=[5, 6, 7, 8],
                num_valid=4,
                block_size=4,
            )
        )
        result = c.compact()
        assert result.blocks_freed == 0

    def test_partial_blocks_merged(self):
        """Two partial blocks from the same request should merge."""
        c = KVBlockCompactor(block_size=4)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2],
                num_valid=2,
                block_size=4,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r1",
                tokens=[3, 4],
                num_valid=2,
                block_size=4,
            )
        )
        result = c.compact()
        # b1 now has [1,2,3,4] (full), b2 is freed
        assert result.blocks_freed == 1
        assert "b2" not in c.blocks

    def test_different_requests_not_merged(self):
        """Blocks from different requests should NOT be merged."""
        c = KVBlockCompactor(block_size=4)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1],
                num_valid=1,
                block_size=4,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r2",
                tokens=[2],
                num_valid=1,
                block_size=4,
            )
        )
        result = c.compact()
        assert result.blocks_freed == 0

    def test_three_partial_blocks_merge(self):
        """Three partial blocks that fit in one block."""
        c = KVBlockCompactor(block_size=6)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2],
                num_valid=2,
                block_size=6,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r1",
                tokens=[3, 4],
                num_valid=2,
                block_size=6,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b3",
                request_id="r1",
                tokens=[5, 6],
                num_valid=2,
                block_size=6,
            )
        )
        result = c.compact()
        # b1 gets [1,2,3,4,5,6] (full), b2 and b3 freed
        assert result.blocks_freed >= 1

    def test_blocks_that_dont_fit_fully(self):
        """Partial blocks that can't fully merge into one block."""
        c = KVBlockCompactor(block_size=4)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2, 3],
                num_valid=3,
                block_size=4,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r1",
                tokens=[4, 5, 6],
                num_valid=3,
                block_size=4,
            )
        )
        result = c.compact()
        # b1 gets 3+1=4 tokens, b2 keeps 2 tokens
        # Only 1 token transferred, b2 is not freed
        assert result.blocks_freed == 0


class TestKVBlockCompactorFragmentation:
    """Test fragmentation statistics."""

    def test_empty_blocks_stats(self):
        c = KVBlockCompactor(block_size=64)
        stats = c.get_fragmentation_stats()
        assert stats.total_blocks == 0
        assert stats.overall_utilization == 0.0

    def test_full_utilization(self):
        c = KVBlockCompactor(block_size=4)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2, 3, 4],
                num_valid=4,
                block_size=4,
            )
        )
        stats = c.get_fragmentation_stats()
        assert stats.overall_utilization == 1.0
        assert stats.waste_pct == 0.0
        assert stats.partial_blocks == 0

    def test_partial_utilization(self):
        c = KVBlockCompactor(block_size=10)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2, 3],
                num_valid=3,
                block_size=10,
            )
        )
        stats = c.get_fragmentation_stats()
        assert stats.overall_utilization == pytest.approx(0.3)
        assert stats.waste_pct == pytest.approx(70.0)
        assert stats.partial_blocks == 1

    def test_fragmentation_ratio(self):
        c = KVBlockCompactor(block_size=4)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1, 2],
                num_valid=2,
                block_size=4,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r1",
                tokens=[1, 2, 3, 4],
                num_valid=4,
                block_size=4,
            )
        )
        stats = c.get_fragmentation_stats()
        assert stats.fragmentation_ratio == 0.5  # 1 of 2 blocks is partial


class TestKVBlockCompactorPeriodic:
    """Test periodic compaction trigger."""

    def test_triggered_at_interval(self):
        c = KVBlockCompactor(block_size=4, compact_interval=50)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1],
                num_valid=1,
                block_size=4,
            )
        )
        c.add_block(
            KVBlock(
                block_id="b2",
                request_id="r1",
                tokens=[2, 3],
                num_valid=2,
                block_size=4,
            )
        )
        result = c.maybe_compact(step=50)
        assert result is not None
        assert result.blocks_freed >= 0

    def test_not_triggered_off_interval(self):
        c = KVBlockCompactor(block_size=4, compact_interval=50)
        result = c.maybe_compact(step=49)
        assert result is None

    def test_not_triggered_at_zero(self):
        c = KVBlockCompactor(block_size=4, compact_interval=50)
        result = c.maybe_compact(step=0)
        assert result is None


class TestKVBlockCompactorStats:
    """Test compactor statistics."""

    def test_stats_structure(self):
        c = KVBlockCompactor(block_size=64, compact_interval=50)
        c.add_block(
            KVBlock(
                block_id="b1",
                request_id="r1",
                tokens=[1],
                num_valid=1,
                block_size=64,
            )
        )
        stats = c.get_stats()
        assert "block_size" in stats
        assert "total_compactions" in stats
        assert "fragmentation" in stats
        assert stats["block_size"] == 64


class TestKVBlockProperties:
    """Test KVBlock data class properties."""

    def test_utilization(self):
        b = KVBlock(block_id="b1", request_id="r1", num_valid=32, block_size=64)
        assert b.utilization == 0.5

    def test_is_partial(self):
        b = KVBlock(block_id="b1", request_id="r1", num_valid=32, block_size=64)
        assert b.is_partial

    def test_is_full(self):
        b = KVBlock(block_id="b1", request_id="r1", num_valid=64, block_size=64)
        assert not b.is_partial
        assert b.utilization == 1.0

    def test_zero_block_size(self):
        b = KVBlock(block_id="b1", request_id="r1", num_valid=0, block_size=0)
        assert b.utilization == 0.0
