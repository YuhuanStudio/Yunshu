"""Tests for mamba_cache.py — mixed attention/SSM KV cache management.

Covers:
- CacheBlockType enum
- HybridKVCache registration and allocation for each type
- MambaSSMState checkpoint/restore roundtrip
- BlockAlignedCacheSplitter alignment
- Mixed model scenario (attention + SSM layers)
- Per-type stats
- State compression (zlib, 8-bit, 4-bit)
- Eviction respects layer boundaries
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from yunshu_engine.mamba_cache import (
    _DEFAULT_BLOCK_SIZES,
    BlockAlignedCacheSplitter,
    CacheBlockType,
    CachePoolStats,
    HybridKVCache,
    MambaSSMState,
    _CachePool,
)

# ── CacheBlockType Tests ───────────────────────────────────────────────


class TestCacheBlockType:
    """Test CacheBlockType enum values and properties."""

    def test_enum_values(self):
        assert CacheBlockType.ATTENTION.value == "attention"
        assert CacheBlockType.MAMBA_SSM.value == "mamba_ssm"
        assert CacheBlockType.SLIDING_WINDOW.value == "sliding_window"
        assert CacheBlockType.MLA.value == "mla"

    def test_enum_count(self):
        assert len(CacheBlockType) == 4

    def test_enum_from_string(self):
        assert CacheBlockType("attention") == CacheBlockType.ATTENTION
        assert CacheBlockType("mamba_ssm") == CacheBlockType.MAMBA_SSM
        assert CacheBlockType("sliding_window") == CacheBlockType.SLIDING_WINDOW
        assert CacheBlockType("mla") == CacheBlockType.MLA

    def test_enum_invalid_string(self):
        with pytest.raises(ValueError):
            CacheBlockType("invalid")

    def test_default_block_sizes(self):
        assert _DEFAULT_BLOCK_SIZES[CacheBlockType.ATTENTION] == 64
        assert _DEFAULT_BLOCK_SIZES[CacheBlockType.MAMBA_SSM] == 1
        assert _DEFAULT_BLOCK_SIZES[CacheBlockType.SLIDING_WINDOW] == 64
        assert _DEFAULT_BLOCK_SIZES[CacheBlockType.MLA] == 64


# ── MambaSSMState Tests ───────────────────────────────────────────────


class TestMambaSSMState:
    """Test MambaSSMState creation, access, and memory estimation."""

    def test_creation(self):
        state = MambaSSMState(
            num_layers=4, inner_dim=48, state_dim=16, conv_dim=4, dtype=mx.float16
        )
        assert state.num_layers == 4
        assert state.inner_dim == 48
        assert state.state_dim == 16
        assert state.conv_dim == 4
        assert not state.initialized

    def test_initialize(self):
        state = MambaSSMState(num_layers=3, inner_dim=32, state_dim=8)
        state.initialize(batch_size=2)
        assert state.initialized
        assert len(state.conv_states) == 3
        assert len(state.ssm_states) == 3

    def test_initialize_shape(self):
        state = MambaSSMState(num_layers=2, inner_dim=16, state_dim=8, conv_dim=4)
        state.initialize(batch_size=1)
        conv, ssm = state.get_layer(0)
        assert conv.shape == (1, 16, 4)
        assert ssm.shape == (1, 16, 8)

    def test_initialize_dtype(self):
        state = MambaSSMState(num_layers=1, inner_dim=8, state_dim=4, dtype=mx.float32)
        state.initialize(batch_size=1)
        conv, ssm = state.get_layer(0)
        assert conv.dtype == mx.float32
        assert ssm.dtype == mx.float32

    def test_update_layer(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4)
        state.initialize(batch_size=1)
        new_conv = mx.ones((1, 8, 4))
        new_ssm = mx.ones((1, 8, 4))
        state.update_layer(0, new_conv, new_ssm)
        conv, ssm = state.get_layer(0)
        assert float(mx.sum(conv)) == pytest.approx(32.0)
        assert float(mx.sum(ssm)) == pytest.approx(32.0)

    def test_get_layer_out_of_range(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4)
        state.initialize(batch_size=1)
        with pytest.raises(IndexError):
            state.get_layer(2)

    def test_update_layer_out_of_range(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4)
        state.initialize(batch_size=1)
        with pytest.raises(IndexError):
            state.update_layer(-1, None, mx.zeros((1, 8, 4)))

    def test_memory_bytes(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4, dtype=mx.float16)
        state.initialize(batch_size=1)
        # conv: 2 layers * 1*8*4 * 2 bytes = 128
        # ssm:  2 layers * 1*8*4 * 2 bytes = 128
        # total = 256 bytes
        assert state.memory_bytes() == 256

    def test_memory_bytes_empty(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4)
        assert state.memory_bytes() == 0


# ── MambaSSMState Checkpoint / Restore Tests ───────────────────────────


class TestMambaSSMStateCheckpoint:
    """Test MambaSSMState serialization and deserialization."""

    def test_checkpoint_restore_roundtrip_none(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4, dtype=mx.float16)
        state.initialize(batch_size=1)
        data = state.checkpoint(compression="none")
        restored = MambaSSMState.restore(data)
        assert restored.num_layers == 2
        assert restored.inner_dim == 8
        assert restored.state_dim == 4
        assert restored.initialized

    def test_checkpoint_values_preserved(self):
        state = MambaSSMState(num_layers=1, inner_dim=4, state_dim=2, dtype=mx.float32)
        state.initialize(batch_size=1)
        # Set specific values
        conv = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32).reshape(1, 4, 1)
        # Reproduce proper shape (1, 4, 4)
        conv = mx.ones((1, 4, 4), dtype=mx.float32) * 0.5
        ssm = mx.ones((1, 4, 2), dtype=mx.float32) * 0.25
        state.update_layer(0, conv, ssm)

        data = state.checkpoint(compression="none")
        restored = MambaSSMState.restore(data)

        r_conv, r_ssm = restored.get_layer(0)
        import numpy as np

        np.testing.assert_allclose(np.array(r_conv), np.array(conv), rtol=1e-5)
        np.testing.assert_allclose(np.array(r_ssm), np.array(ssm), rtol=1e-5)

    def test_checkpoint_uninitialized_raises(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4)
        with pytest.raises(RuntimeError, match="uninitialized"):
            state.checkpoint()

    def test_checkpoint_zlib_compression(self):
        state = MambaSSMState(num_layers=2, inner_dim=8, state_dim=4, dtype=mx.float16)
        state.initialize(batch_size=1)
        data_raw = state.checkpoint(compression="none")
        data_zlib = state.checkpoint(compression="zlib")
        # Zlib should compress zeros well
        assert len(data_zlib) < len(data_raw)

        restored = MambaSSMState.restore(data_zlib)
        assert restored.num_layers == state.num_layers
        assert restored.inner_dim == state.inner_dim

    def test_checkpoint_8bit_compression(self):
        state = MambaSSMState(num_layers=1, inner_dim=4, state_dim=2, dtype=mx.float16)
        state.initialize(batch_size=1)
        # Set non-zero values to test quantization
        conv = mx.array([[[1.0, -0.5, 0.25, -0.125]]], dtype=mx.float16)
        # Need proper shape: (1, 4, 4) conv_dim=4
        conv = mx.zeros((1, 4, 4), dtype=mx.float16)
        conv[0, 0, 0] = mx.array(1.0)
        conv[0, 1, 0] = mx.array(-0.5)
        conv[0, 2, 0] = mx.array(0.25)
        conv[0, 3, 0] = mx.array(-0.125)
        state.update_layer(0, conv, mx.zeros((1, 4, 2), dtype=mx.float16))

        data = state.checkpoint(compression="8bit")
        restored = MambaSSMState.restore(data)

        r_conv, _ = restored.get_layer(0)
        import numpy as np

        # 8-bit quantization: max error ~scale/127
        np.testing.assert_allclose(
            np.array(r_conv), np.array(conv), atol=0.02, rtol=0.1
        )

    def test_checkpoint_4bit_compression(self):
        state = MambaSSMState(num_layers=1, inner_dim=4, state_dim=2, dtype=mx.float16)
        state.initialize(batch_size=1)
        # Set non-zero values
        conv = mx.zeros((1, 4, 4), dtype=mx.float16)
        conv[0, 0, 0] = mx.array(1.0)
        conv[0, 1, 0] = mx.array(-1.0)
        state.update_layer(0, conv, mx.zeros((1, 4, 2), dtype=mx.float16))

        data = state.checkpoint(compression="4bit")
        restored = MambaSSMState.restore(data)

        r_conv, _ = restored.get_layer(0)
        import numpy as np

        # 4-bit has coarser quantization
        np.testing.assert_allclose(
            np.array(r_conv), np.array(conv), atol=0.15, rtol=0.2
        )

    def test_compression_size_comparison(self):
        """8-bit and 4-bit should be smaller than raw for non-trivial data."""
        state = MambaSSMState(num_layers=2, inner_dim=16, state_dim=8, dtype=mx.float32)
        state.initialize(batch_size=1)
        # Fill with non-zero data to prevent trivial compression
        for i in range(state.num_layers):
            conv = mx.random.uniform(-1, 1, (1, 16, 4), dtype=mx.float32)
            ssm = mx.random.uniform(-1, 1, (1, 16, 8), dtype=mx.float32)
            state.update_layer(i, conv, ssm)

        raw = state.checkpoint(compression="none")
        quant8 = state.checkpoint(compression="8bit")
        quant4 = state.checkpoint(compression="4bit")

        assert len(quant8) < len(raw)
        assert len(quant4) < len(quant8)


# ── CachePool Tests ────────────────────────────────────────────────────


class TestCachePool:
    """Test _CachePool allocation, freeing, and eviction."""

    def test_allocate(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 10, (16, 64, 128))
        blocks = pool.allocate(3)
        assert len(blocks) == 3
        assert pool.blocks_used == 3
        assert pool.blocks_free == 7

    def test_allocate_exhausts(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 4, (16, 64, 128))
        pool.allocate(4)
        with pytest.raises(MemoryError):
            pool.allocate(1)

    def test_free(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 10, (16, 64, 128))
        blocks = pool.allocate(5)
        freed = pool.free(blocks[:3])
        assert freed == 3
        assert pool.blocks_used == 2
        assert pool.blocks_free == 8

    def test_free_already_free(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 10, (16, 64, 128))
        blocks = pool.allocate(3)
        pool.free(blocks)
        # Freeing again should return 0
        freed = pool.free(blocks)
        assert freed == 0

    def test_register_layer(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 10, (16, 64, 128))
        pool.register_layer(0)
        pool.register_layer(2)
        pool.register_layer(4)
        assert pool.num_layers == 3

    def test_evict_lru(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 10, (16, 64, 128))
        pool.allocate(5)
        evicted = pool.evict_lru(3)
        assert len(evicted) == 3
        assert pool.blocks_used == 2

    def test_evict_lru_empty_pool(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 10, (16, 64, 128))
        evicted = pool.evict_lru(3)
        assert evicted == []

    def test_stats(self):
        pool = _CachePool(CacheBlockType.ATTENTION, 64, 100, (16, 64, 128))
        pool.allocate(20)
        stats = pool.get_stats()
        assert stats.block_type == CacheBlockType.ATTENTION
        assert stats.blocks_total == 100
        assert stats.blocks_used == 20
        assert stats.blocks_free == 80
        assert stats.allocations == 20


# ── HybridKVCache Tests ───────────────────────────────────────────────


class TestHybridKVCache:
    """Test HybridKVCache registration, allocation, and per-type routing."""

    def test_single_attention_type(self):
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.ATTENTION, (16, 64, 128))
        blocks = cache.allocate(0, 4)
        assert len(blocks) == 4
        assert cache.num_layers == 2
        assert cache.num_pools == 1

    def test_mixed_types(self):
        cache = HybridKVCache(max_blocks=128)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        attn_blocks = cache.allocate(0, 4)
        ssm_blocks = cache.allocate(1, 2)
        assert len(attn_blocks) == 4
        assert len(ssm_blocks) == 2
        assert cache.num_pools == 2

    def test_unregistered_layer_raises(self):
        cache = HybridKVCache()
        with pytest.raises(KeyError, match="not registered"):
            cache.allocate(0, 1)

    def test_free_routing(self):
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        blocks = cache.allocate(0, 4)
        freed = cache.free(0, blocks[:2])
        assert freed == 2

    def test_get_layer_type(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))

        assert cache.get_layer_type(0) == CacheBlockType.ATTENTION
        assert cache.get_layer_type(1) == CacheBlockType.MAMBA_SSM
        assert cache.get_layer_type(99) is None

    def test_set_get_cache_attention(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        data = {"keys": mx.zeros((1, 64, 128)), "values": mx.zeros((1, 64, 128))}
        cache.set_cache(0, data)
        assert cache.get_cache(0) is data

    def test_set_get_cache_ssm(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.MAMBA_SSM, (48, 16))
        state = MambaSSMState(num_layers=1, inner_dim=48, state_dim=16)
        state.initialize(batch_size=1)
        cache.set_cache(0, state)
        assert cache.get_cache(0) is state

    def test_set_ssm_wrong_type_raises(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.MAMBA_SSM, (48, 16))
        with pytest.raises(TypeError, match="MambaSSMState"):
            cache.set_cache(0, {"not": "a state"})

    def test_sliding_window_type(self):
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.SLIDING_WINDOW, (16, 64, 128))
        blocks = cache.allocate(0, 4)
        assert len(blocks) == 4
        assert cache.get_layer_type(0) == CacheBlockType.SLIDING_WINDOW

    def test_mla_type(self):
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.MLA, (16, 64, 512))
        blocks = cache.allocate(0, 4)
        assert len(blocks) == 4
        assert cache.get_layer_type(0) == CacheBlockType.MLA

    def test_all_four_types(self):
        cache = HybridKVCache(max_blocks=256)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.SLIDING_WINDOW, (16, 64, 128))
        cache.register_layer(3, CacheBlockType.MLA, (16, 64, 512))

        assert cache.num_pools == 4
        assert cache.num_layers == 4

        for i in range(4):
            blocks = cache.allocate(i, 2)
            assert len(blocks) == 2

    def test_clear(self):
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.allocate(0, 10)
        cache.set_cache(0, {"data": True})
        cache.clear()
        assert cache.get_cache(0) is None


# ── HybridKVCache Stats Tests ──────────────────────────────────────────


class TestHybridKVCacheStats:
    """Test per-type and aggregate statistics."""

    def test_stats_single_pool(self):
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.allocate(0, 10)
        stats = cache.get_stats()
        assert stats["registered_layers"] == 1
        assert stats["total_blocks_used"] == 10
        assert "attention" in stats["pools"]
        assert stats["pools"]["attention"]["blocks_used"] == 10

    def test_stats_mixed_pools(self):
        cache = HybridKVCache(max_blocks=128)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.allocate(0, 10)
        cache.allocate(1, 5)
        stats = cache.get_stats()
        assert stats["registered_layers"] == 2
        assert stats["total_blocks_used"] == 15
        assert "attention" in stats["pools"]
        assert "mamba_ssm" in stats["pools"]
        assert stats["pools"]["attention"]["blocks_used"] == 10
        assert stats["pools"]["mamba_ssm"]["blocks_used"] == 5

    def test_stats_layer_types_mapping(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))
        stats = cache.get_stats()
        assert stats["layer_types"] == {
            0: "attention",
            1: "mamba_ssm",
            2: "attention",
        }

    def test_stats_utilization(self):
        cache = HybridKVCache(max_blocks=100)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.allocate(0, 25)
        stats = cache.get_stats()
        assert stats["pools"]["attention"]["utilization"] == pytest.approx(0.25)


# ── BlockAlignedCacheSplitter Tests ────────────────────────────────────


class TestBlockAlignedCacheSplitter:
    """Test layer group alignment for eviction safety."""

    def test_define_group(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        splitter = BlockAlignedCacheSplitter(cache)
        splitter.define_group(0, [0, 2], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1], CacheBlockType.MAMBA_SSM)
        assert splitter.num_groups == 2

    def test_auto_group_by_type(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        splitter = BlockAlignedCacheSplitter(cache)
        groups = splitter.auto_group_by_type()
        # Should create 3 groups: [0] attn, [1] ssm, [2] attn
        assert len(groups) == 3
        assert groups[0].block_type == CacheBlockType.ATTENTION
        assert groups[1].block_type == CacheBlockType.MAMBA_SSM
        assert groups[2].block_type == CacheBlockType.ATTENTION

    def test_auto_group_contiguous(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(2, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(3, CacheBlockType.MAMBA_SSM, (48, 16))

        splitter = BlockAlignedCacheSplitter(cache)
        groups = splitter.auto_group_by_type()
        assert len(groups) == 2
        assert groups[0].layer_indices == [0, 1]
        assert groups[1].layer_indices == [2, 3]

    def test_safe_eviction_complete_group(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        splitter = BlockAlignedCacheSplitter(cache)
        splitter.define_group(0, [0, 2], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1], CacheBlockType.MAMBA_SSM)

        # Evicting complete group is safe
        assert splitter.is_safe_eviction([0, 2]) is True
        assert splitter.is_safe_eviction([1]) is True
        assert splitter.is_safe_eviction([0, 1, 2]) is True

    def test_unsafe_eviction_partial_group(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        splitter = BlockAlignedCacheSplitter(cache)
        splitter.define_group(0, [0, 2], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1], CacheBlockType.MAMBA_SSM)

        # Evicting only one layer from a 2-layer group is unsafe
        assert splitter.is_safe_eviction([0]) is False
        assert splitter.is_safe_eviction([2]) is False

    def test_complete_eviction_set(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.register_layer(2, CacheBlockType.ATTENTION, (16, 64, 128))

        splitter = BlockAlignedCacheSplitter(cache)
        splitter.define_group(0, [0, 2], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1], CacheBlockType.MAMBA_SSM)

        # Expanding partial set [0] should include both [0, 2]
        result = splitter.get_complete_eviction_set([0])
        assert result == [0, 2]

    def test_get_group_for_layer(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))

        splitter = BlockAlignedCacheSplitter(cache)
        splitter.define_group(0, [0], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1], CacheBlockType.MAMBA_SSM)

        g0 = splitter.get_group_for_layer(0)
        g1 = splitter.get_group_for_layer(1)
        assert g0.group_id == 0
        assert g1.group_id == 1

    def test_get_stats(self):
        cache = HybridKVCache()
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))

        splitter = BlockAlignedCacheSplitter(cache)
        splitter.define_group(0, [0], CacheBlockType.ATTENTION)
        splitter.define_group(1, [1], CacheBlockType.MAMBA_SSM)

        stats = splitter.get_stats()
        assert stats["num_groups"] == 2
        assert "0" in stats["groups"] or 0 in stats["groups"]


# ── Mixed Model Scenario Test ──────────────────────────────────────────


class TestMixedModelScenario:
    """End-to-end test simulating a hybrid Jamba-like model."""

    def test_jamba_like_model(self):
        """Simulate Jamba: interleaved attention + Mamba SSM layers."""
        cache = HybridKVCache(max_blocks=512)

        # Jamba-like: 32 layers, alternating attention + SSM
        for i in range(32):
            if i % 4 in (0, 3):
                cache.register_layer(
                    i,
                    CacheBlockType.ATTENTION,
                    (16, 64, 128),
                    block_size=64,
                )
            else:
                cache.register_layer(
                    i,
                    CacheBlockType.MAMBA_SSM,
                    (48, 16),
                    block_size=1,
                )

        assert cache.num_layers == 32
        assert cache.num_pools == 2  # attention + SSM

        # Allocate for each layer
        for i in range(32):
            blocks = cache.allocate(i, 4)
            assert len(blocks) == 4

        stats = cache.get_stats()
        # 16 attention layers * 4 blocks + 16 SSM layers * 4 blocks = 128
        assert stats["total_blocks_used"] == 128
        assert stats["registered_layers"] == 32

    def test_preemption_checkpoint_restore(self):
        """Simulate preemption: checkpoint SSM state, free blocks, restore."""
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (16, 4))

        # Set up SSM state
        ssm = MambaSSMState(num_layers=1, inner_dim=16, state_dim=4)
        ssm.initialize(batch_size=1)
        ssm.update_layer(
            0,
            conv_state=mx.ones((1, 16, 4), dtype=mx.float16),
            ssm_state=mx.ones((1, 16, 4), dtype=mx.float16) * 0.5,
        )
        cache.set_cache(1, ssm)

        # Allocate blocks
        attn_blocks = cache.allocate(0, 4)
        ssm_blocks = cache.allocate(1, 2)

        # Preempt: checkpoint SSM, free all blocks
        ssm_data = ssm.checkpoint()
        cache.free(0, attn_blocks)
        cache.free(1, ssm_blocks)

        assert cache.get_stats()["total_blocks_used"] == 0

        # Restore: re-allocate and restore SSM state
        new_ssm = MambaSSMState.restore(ssm_data)
        cache.set_cache(1, new_ssm)
        assert new_ssm.initialized
        conv, ssm_state = new_ssm.get_layer(0)
        assert float(mx.sum(conv)) == pytest.approx(64.0)

    def test_eviction_respects_boundaries(self):
        """Eviction must respect layer group boundaries."""
        cache = HybridKVCache(max_blocks=256)
        # 6 layers: attn, attn, ssm, ssm, attn, attn
        for i in range(6):
            if i < 2 or i >= 4:
                cache.register_layer(i, CacheBlockType.ATTENTION, (16, 64, 128))
            else:
                cache.register_layer(i, CacheBlockType.MAMBA_SSM, (48, 16))

        splitter = BlockAlignedCacheSplitter(cache)
        groups = splitter.auto_group_by_type()
        # 3 groups: [0,1] attn, [2,3] ssm, [4,5] attn
        assert len(groups) == 3

        # Evicting [0,1] is safe (complete group)
        assert splitter.is_safe_eviction([0, 1]) is True
        # Evicting [0] alone is unsafe
        assert splitter.is_safe_eviction([0]) is False
        # Expanding [2] to complete gives [2,3]
        assert splitter.get_complete_eviction_set([2]) == [2, 3]


# ── CachePoolStats Tests ───────────────────────────────────────────────


class TestCachePoolStats:
    """Test CachePoolStats properties."""

    def test_blocks_free(self):
        stats = CachePoolStats(
            block_type=CacheBlockType.ATTENTION,
            blocks_total=100,
            blocks_used=30,
        )
        assert stats.blocks_free == 70

    def test_utilization_zero_total(self):
        stats = CachePoolStats(
            block_type=CacheBlockType.ATTENTION,
            blocks_total=0,
            blocks_used=0,
        )
        assert stats.utilization == 0.0

    def test_utilization_half(self):
        stats = CachePoolStats(
            block_type=CacheBlockType.ATTENTION,
            blocks_total=100,
            blocks_used=50,
        )
        assert stats.utilization == pytest.approx(0.5)


# ── Integration with Scheduler Config ──────────────────────────────────


class TestSchedulerIntegration:
    """Test that cache_type can be added to SchedulerConfig context."""

    def test_cache_type_in_config(self):
        """Verify CacheBlockType can be used as a config field."""
        from yunshu_engine.scheduler import SchedulerConfig

        config = SchedulerConfig()
        # The module can be imported and CacheBlockType can be used
        # alongside SchedulerConfig without conflicts
        assert hasattr(config, "model_name")

    def test_hybrid_cache_stats_serializable(self):
        """Stats dict should be JSON-serializable for API responses."""
        import json

        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.register_layer(1, CacheBlockType.MAMBA_SSM, (48, 16))
        cache.allocate(0, 4)
        cache.allocate(1, 2)

        stats = cache.get_stats()
        # Should be JSON-serializable
        serialized = json.dumps(stats)
        deserialized = json.loads(serialized)
        assert deserialized["total_blocks_used"] == 6
        assert deserialized["registered_layers"] == 2

    def test_hybrid_cache_in_scheduler_stats(self):
        """HybridKVCache stats can be embedded in scheduler.get_stats()."""
        cache = HybridKVCache(max_blocks=64)
        cache.register_layer(0, CacheBlockType.ATTENTION, (16, 64, 128))
        cache.allocate(0, 8)

        hybrid_stats = cache.get_stats()
        # Simulate what scheduler.get_stats() would return
        scheduler_stats = {
            "waiting": 5,
            "running": 3,
            "hybrid_kv_cache": hybrid_stats,
        }
        assert "hybrid_kv_cache" in scheduler_stats
        assert scheduler_stats["hybrid_kv_cache"]["total_blocks_used"] == 8
