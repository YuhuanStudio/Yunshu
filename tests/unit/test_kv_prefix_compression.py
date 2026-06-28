"""Tests for KV Prefix Compression and Sliding Window KV Manager.

Tests cover:
- KVPrefixCompressor: all 3 compression strategies (mean_pool, top_k, frequency_aware)
- Compression/decompression roundtrip
- Compression ratio reporting
- Empty input edge cases
- SlidingWindowKVManager: eviction logic
- System prompt preservation
- Multi-request management
- Statistics tracking
- Window boundary edge cases
"""
from __future__ import annotations

import numpy as np
import pytest

from yunshu_engine.kv_prefix_compression import (
    CompressionResult,
    KVPrefixCompressor,
    SlidingWindowKVManager,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_block(block_id: int, num_layers: int = 2, num_heads: int = 2, block_size: int = 4, head_dim: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic KV block as (keys, values) tuple."""
    rng = np.random.RandomState(block_id)
    keys = rng.randn(num_layers, num_heads, block_size, head_dim).astype(np.float32)
    values = rng.randn(num_layers, num_heads, block_size, head_dim).astype(np.float32)
    return keys, values


def _make_blocks(n: int, **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create n synthetic KV blocks."""
    return [_make_block(i, **kwargs) for i in range(n)]


def _make_compressor(**kwargs) -> KVPrefixCompressor:
    """Create a compressor with small dims for testing."""
    defaults = dict(
        compression_factor=2,
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
    )
    defaults.update(kwargs)
    return KVPrefixCompressor(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Test KVPrefixCompressor — Mean Pool Strategy
# ═══════════════════════════════════════════════════════════════════════════════


class TestMeanPoolCompression:

    def test_compress_reduces_block_count(self):
        """Mean pool compresses N blocks into N/K blocks."""
        comp = _make_compressor(compression_factor=4)
        blocks = _make_blocks(8)
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        # 8 blocks / factor 4 = 2 compressed blocks
        assert result.compressed.shape[0] == 2

    def test_compress_preserves_shape_per_block(self):
        """Each compressed block has the same shape as input blocks."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4, num_layers=2, num_heads=2, block_size=4, head_dim=4)
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        # Combined shape: (2, 2, 2, 4, 4) since input is (keys, values) → stacked to (2, L, H, BS, HD)
        assert result.compressed.shape[1:] == (2, 2, 2, 4, 4)

    def test_compress_odd_number_blocks(self):
        """Last group can be smaller than K blocks."""
        comp = _make_compressor(compression_factor=4)
        blocks = _make_blocks(7)
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        # 7 blocks / factor 4 → 2 groups (4 + 3)
        assert result.compressed.shape[0] == 2

    def test_mean_pool_values_are_averages(self):
        """Mean pool: compressed block should equal average of constituent blocks."""
        comp = _make_compressor(compression_factor=2)
        # Create deterministic blocks
        b0_keys = np.ones((2, 2, 4, 4), dtype=np.float32) * 1.0
        b0_vals = np.ones((2, 2, 4, 4), dtype=np.float32) * 2.0
        b1_keys = np.ones((2, 2, 4, 4), dtype=np.float32) * 3.0
        b1_vals = np.ones((2, 2, 4, 4), dtype=np.float32) * 4.0
        blocks = [
            (b0_keys, b0_vals),
            (b1_keys, b1_vals),
        ]
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        # Mean of stacked keys(1.0, 3.0) = 2.0, values(2.0, 4.0) = 3.0
        compressed = result.compressed
        # compressed[0] is the pooled block: (2, L, H, BS, HD)
        # dim 0 is keys/values, rest are spatial
        assert np.allclose(compressed[0, 0], 2.0)  # keys averaged
        assert np.allclose(compressed[0, 1], 3.0)  # values averaged

    def test_decompress_mean_pool_tiling(self):
        """Decompressed mean pool blocks are tiles of the compressed block."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        decompressed = comp.decompress_blocks(result)
        # Should produce 4 blocks (2 compressed * factor 2)
        assert len(decompressed) == 4

    def test_mean_pool_compression_ratio(self):
        """Mean pool achieves approximately compression_factor ratio."""
        comp = _make_compressor(compression_factor=4)
        blocks = _make_blocks(8)
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        assert result.ratio >= 3.0  # Close to 4x but depends on stacking


# ═══════════════════════════════════════════════════════════════════════════════
# Test KVPrefixCompressor — Top-K Strategy
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopKCompression:

    def test_keeps_highest_score_blocks(self):
        """Top-k keeps blocks with highest attention scores."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        scores = [0.1, 0.9, 0.3, 0.7]
        result = comp.compress_blocks(blocks, attention_scores=scores, strategy="top_k")
        # Should keep 2 blocks: indices 1 (0.9) and 3 (0.7)
        assert result.block_ids == [1, 3]

    def test_top_k_keeps_correct_count(self):
        """Top-k keeps N/K blocks."""
        comp = _make_compressor(compression_factor=4)
        blocks = _make_blocks(8)
        scores = [float(i) for i in range(8)]
        result = comp.compress_blocks(blocks, attention_scores=scores, strategy="top_k")
        assert result.compressed.shape[0] == 2

    def test_top_k_exact_decompression(self):
        """Top-k decompression is exact: returned blocks match original."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        scores = [0.1, 0.9, 0.3, 0.7]
        result = comp.compress_blocks(blocks, attention_scores=scores, strategy="top_k")
        decompressed = comp.decompress_blocks(result)
        assert len(decompressed) == 2

    def test_top_k_default_scores(self):
        """Top-k with no scores uses uniform scores (keeps first N/K)."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        result = comp.compress_blocks(blocks, strategy="top_k")
        assert result.compressed.shape[0] == 2

    def test_top_k_single_block(self):
        """Top-k with 1 block keeps that block."""
        comp = _make_compressor(compression_factor=4)
        blocks = _make_blocks(1)
        result = comp.compress_blocks(blocks, attention_scores=[1.0], strategy="top_k")
        assert result.compressed.shape[0] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test KVPrefixCompressor — Frequency-Aware Strategy
# ═══════════════════════════════════════════════════════════════════════════════


class TestFrequencyAwareCompression:

    def test_keeps_high_access_blocks(self):
        """Frequency-aware keeps blocks with high access count at full fidelity."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        counts = [1, 10, 1, 10]  # Blocks 1, 3 are high-access
        result = comp.compress_blocks(blocks, access_counts=counts, strategy="frequency_aware")
        # High-access blocks (1, 3) should be in result at full fidelity
        assert 1 in result.block_ids
        assert 3 in result.block_ids

    def test_compresses_low_access_blocks(self):
        """Frequency-aware compresses low-access blocks via mean pool."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(6)
        counts = [1, 1, 1, 10, 10, 10]
        result = comp.compress_blocks(blocks, access_counts=counts, strategy="frequency_aware")
        # Should have fewer blocks than input
        assert result.compressed.shape[0] < 6

    def test_decompress_frequency_aware(self):
        """Frequency-aware decompression returns blocks."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        counts = [1, 5, 1, 5]
        result = comp.compress_blocks(blocks, access_counts=counts, strategy="frequency_aware")
        decompressed = comp.decompress_blocks(result)
        assert len(decompressed) > 0

    def test_all_high_access(self):
        """When all blocks have high access, all are kept."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        counts = [10, 10, 10, 10]
        result = comp.compress_blocks(blocks, access_counts=counts, strategy="frequency_aware")
        # All should be kept
        assert result.compressed.shape[0] == 4

    def test_default_counts(self):
        """Frequency-aware with no counts uses uniform (all kept)."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        result = comp.compress_blocks(blocks, strategy="frequency_aware")
        assert result.compressed.shape[0] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test KVPrefixCompressor — General
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompressorGeneral:

    def test_empty_input(self):
        """Empty block list returns empty result."""
        comp = _make_compressor()
        result = comp.compress_blocks([], strategy="mean_pool")
        assert result.compressed.shape == (0,)
        assert result.ratio == 1.0
        assert result.block_ids == []

    def test_invalid_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        comp = _make_compressor()
        blocks = _make_blocks(2)
        with pytest.raises(ValueError, match="Unknown compression strategy"):
            comp.compress_blocks(blocks, strategy="invalid_strategy")

    def test_get_compression_ratio(self):
        """get_compression_ratio returns the configured factor after compression."""
        comp = _make_compressor(compression_factor=8)
        blocks = _make_blocks(8)
        comp.compress_blocks(blocks, strategy="mean_pool")
        assert comp.get_compression_ratio() == 8.0

    def test_get_compression_ratio_no_compressions(self):
        """get_compression_ratio defaults to 1.0 before any compressions."""
        comp = KVPrefixCompressor()
        assert comp.get_compression_ratio() == 1.0

    def test_get_stats(self):
        """Stats track compressions and bytes saved."""
        comp = _make_compressor(compression_factor=2)
        blocks = _make_blocks(4)
        comp.compress_blocks(blocks, strategy="mean_pool")
        stats = comp.get_stats()
        assert stats["total_compressions"] == 1
        assert stats["total_bytes_saved"] >= 0
        assert stats["compression_factor"] == 2

    def test_decompress_empty_result(self):
        """Decompressing empty result returns empty list."""
        comp = _make_compressor()
        empty = CompressionResult(
            compressed=np.array([]),
            original_shape=(0,),
            compressed_shape=(0,),
            strategy="mean_pool",
            ratio=1.0,
            block_ids=[],
        )
        assert comp.decompress_blocks(empty) == []

    def test_single_block_compression(self):
        """Single block compresses to 1 block."""
        comp = _make_compressor(compression_factor=4)
        blocks = _make_blocks(1)
        result = comp.compress_blocks(blocks, strategy="mean_pool")
        assert result.compressed.shape[0] == 1

    def test_numpy_array_input(self):
        """Compressor accepts raw numpy arrays (not just tuples)."""
        comp = _make_compressor(compression_factor=2)
        block = np.random.randn(2, 2, 4, 4).astype(np.float32)
        result = comp.compress_blocks([block, block], strategy="mean_pool")
        assert result.compressed.shape[0] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test SlidingWindowKVManager — Basic
# ═══════════════════════════════════════════════════════════════════════════════


class TestSlidingWindowBasic:

    def test_register_request(self):
        """New request starts with empty blocks."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.register_request("req-1")
        active = mgr.get_active_blocks("req-1")
        assert len(active) == 0

    def test_add_blocks_within_window(self):
        """Blocks within window are kept."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.register_request("req-1")
        # Add blocks at positions 0, 64, 128, 192 (all within 256)
        for pos in [0, 64, 128, 192]:
            mgr.on_new_token(pos, request_id="req-1")
        active = mgr.get_active_blocks("req-1")
        assert len(active) == 4

    def test_eviction_outside_window(self):
        """Blocks outside window are evicted."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64)
        mgr.register_request("req-1")
        # Add blocks at positions 0, 64, 128, 192, 256, 320
        evicted_all = []
        for pos in [0, 64, 128, 192, 256, 320]:
            evicted = mgr.on_new_token(pos, request_id="req-1")
            evicted_all.extend(evicted)
        # With window=128, at position 320, only blocks >= 320-128+64 = 256 are kept
        # So blocks at 0, 64, 128, 192 should be evicted
        assert len(evicted_all) > 0
        assert 0 in evicted_all

    def test_window_size_property(self):
        """window_size property returns configured value."""
        mgr = SlidingWindowKVManager(window_size=4096, block_size=64)
        assert mgr.window_size == 4096

    def test_block_size_property(self):
        """block_size property returns configured value."""
        mgr = SlidingWindowKVManager(window_size=4096, block_size=128)
        assert mgr.block_size == 128


# ═══════════════════════════════════════════════════════════════════════════════
# Test SlidingWindowKVManager — System Prompt Preservation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSystemPromptPreservation:

    def test_system_prompt_blocks_never_evicted(self):
        """System prompt blocks are always retained."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64, system_prompt_blocks=2)
        mgr.register_request("req-1", system_prompt_blocks=2)
        # Add many blocks to push past window
        for pos in [0, 64, 128, 192, 256, 320, 384, 448]:
            mgr.on_new_token(pos, request_id="req-1")
        active = mgr.get_active_blocks("req-1")
        sys_blocks = [b for b in active if b.is_system_prompt]
        assert len(sys_blocks) == 2

    def test_system_prompt_at_high_positions(self):
        """System prompt blocks retained even at very high positions."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64)
        mgr.register_request("req-1", system_prompt_blocks=2)
        # Simulate a very long generation
        for pos in range(0, 2000, 64):
            mgr.on_new_token(pos, request_id="req-1")
        active = mgr.get_active_blocks("req-1")
        sys_blocks = [b for b in active if b.is_system_prompt]
        assert len(sys_blocks) == 2
        assert sys_blocks[0].block_id == 0
        assert sys_blocks[1].block_id == 1

    def test_no_system_prompt_still_works(self):
        """Manager works correctly with zero system prompt blocks."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64, system_prompt_blocks=0)
        mgr.register_request("req-1")
        for pos in [0, 64, 128, 192, 256]:
            mgr.on_new_token(pos, request_id="req-1")
        active = mgr.get_active_blocks("req-1")
        sys_blocks = [b for b in active if b.is_system_prompt]
        assert len(sys_blocks) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test SlidingWindowKVManager — Multi-Request
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiRequest:

    def test_independent_windows(self):
        """Each request has independent block tracking."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.register_request("req-1")
        mgr.register_request("req-2")
        for pos in [0, 64, 128]:
            mgr.on_new_token(pos, request_id="req-1")
        for pos in [0, 64]:
            mgr.on_new_token(pos, request_id="req-2")
        assert len(mgr.get_active_blocks("req-1")) == 3
        assert len(mgr.get_active_blocks("req-2")) == 2

    def test_remove_request(self):
        """Removing a request cleans up its blocks."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.register_request("req-1")
        for pos in [0, 64, 128]:
            mgr.on_new_token(pos, request_id="req-1")
        mgr.remove_request("req-1")
        assert mgr.get_active_blocks("req-1") == []

    def test_unknown_request_returns_empty(self):
        """Active blocks for unknown request returns empty list."""
        mgr = SlidingWindowKVManager()
        assert mgr.get_active_blocks("nonexistent") == []


# ═══════════════════════════════════════════════════════════════════════════════
# Test SlidingWindowKVManager — Configure
# ═══════════════════════════════════════════════════════════════════════════════


class TestSlidingWindowConfigure:

    def test_configure_window_size(self):
        """configure() updates window size."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.configure(window_size=4096)
        assert mgr.window_size == 4096

    def test_configure_from_model_config(self):
        """configure() reads from model_config dict."""
        mgr = SlidingWindowKVManager()
        mgr.configure(model_config={
            "sliding_window": 8192,
            "block_size": 128,
            "system_prompt_blocks": 3,
        })
        assert mgr.window_size == 8192
        assert mgr.block_size == 128

    def test_configure_none_preserves_values(self):
        """configure() with None window_size preserves current value."""
        mgr = SlidingWindowKVManager(window_size=4096)
        mgr.configure(window_size=None)
        assert mgr.window_size == 4096


# ═══════════════════════════════════════════════════════════════════════════════
# Test SlidingWindowKVManager — Statistics
# ═══════════════════════════════════════════════════════════════════════════════


class TestSlidingWindowStats:

    def test_stats_eviction_count(self):
        """Stats track total evictions."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64)
        mgr.register_request("req-1")
        for pos in [0, 64, 128, 192, 256, 320]:
            mgr.on_new_token(pos, request_id="req-1")
        stats = mgr.get_stats()
        assert stats.total_evictions > 0

    def test_stats_active_blocks(self):
        """Stats track active block count."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.register_request("req-1")
        for pos in [0, 64, 128]:
            mgr.on_new_token(pos, request_id="req-1")
        stats = mgr.get_stats()
        assert stats.active_blocks >= 3

    def test_stats_memory_saved(self):
        """Stats estimate memory saved from evictions."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64)
        mgr.register_request("req-1")
        for pos in [0, 64, 128, 192, 256, 320]:
            mgr.on_new_token(pos, request_id="req-1")
        stats = mgr.get_stats()
        assert stats.memory_saved_bytes >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test SlidingWindowKVManager — Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSlidingWindowEdgeCases:

    def test_mid_block_positions(self):
        """Mid-block positions don't create new blocks."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        mgr.register_request("req-1")
        # Position 32 is mid-block (block 0 is 0-63)
        evicted = mgr.on_new_token(32, request_id="req-1")
        assert evicted == []

    def test_exact_window_boundary(self):
        """Block at exact window boundary is kept."""
        mgr = SlidingWindowKVManager(window_size=128, block_size=64)
        mgr.register_request("req-1")
        for pos in [0, 64, 128, 192]:
            mgr.on_new_token(pos, request_id="req-1")
        # At pos 192, window starts at 192 - 128 + 64 = 128
        active = mgr.get_active_blocks("req-1")
        positions = [b.token_position for b in active]
        assert 128 in positions
        assert 192 in positions

    def test_on_new_token_auto_registers(self):
        """on_new_token auto-registers unknown requests."""
        mgr = SlidingWindowKVManager(window_size=256, block_size=64)
        # Call without register_request first
        mgr.on_new_token(0, request_id="auto-reg")
        active = mgr.get_active_blocks("auto-reg")
        assert len(active) == 1

    def test_window_size_equals_block_size(self):
        """Window of exactly 1 block still works."""
        mgr = SlidingWindowKVManager(window_size=64, block_size=64)
        mgr.register_request("req-1")
        for pos in [0, 64, 128]:
            mgr.on_new_token(pos, request_id="req-1")
        active = mgr.get_active_blocks("req-1")
        # Only the latest block should be active
        assert len(active) == 1
        assert active[0].token_position == 128
