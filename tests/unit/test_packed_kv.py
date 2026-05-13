"""Tests for C19: Packed KV format — Metal SIMD-optimized layout."""
import pytest

from yunshu_kv.packed_kv import (
    PackedKVCache,
    PackedKVConfig,
    PackedKVStats,
    compute_simd_aligned_dim,
)


class TestPackedKVConfig:
    def test_defaults(self):
        cfg = PackedKVConfig()
        assert not cfg.enabled
        assert cfg.simd_width == 32
        assert cfg.interleave_kv
        assert cfg.quantize_bits == 0
        assert cfg.quantize_group_size == 32

    def test_from_env_disabled(self):
        cfg = PackedKVConfig.from_env()
        assert not cfg.enabled

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_PACKED_KV", "1")
        cfg = PackedKVConfig.from_env()
        assert cfg.enabled

    def test_from_env_quantize(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_PACKED_KV", "1")
        monkeypatch.setenv("YUNSHU_KV_QUANT_BITS", "4")
        monkeypatch.setenv("YUNSHU_KV_QUANT_GROUP", "64")
        cfg = PackedKVConfig.from_env()
        assert cfg.quantize_bits == 4
        assert cfg.quantize_group_size == 64

    def test_compute_padded_head_dim_aligned(self):
        cfg = PackedKVConfig(simd_width=32)
        # 128 is already aligned
        assert cfg.compute_padded_head_dim(128) == 128

    def test_compute_padded_head_dim_needs_padding(self):
        cfg = PackedKVConfig(simd_width=32)
        # 80 needs padding to 96
        assert cfg.compute_padded_head_dim(80) == 96

    def test_compute_padded_head_dim_small(self):
        cfg = PackedKVConfig(simd_width=32)
        # 64 is aligned
        assert cfg.compute_padded_head_dim(64) == 64

    def test_compute_padded_head_dim_zero(self):
        cfg = PackedKVConfig(simd_width=32)
        assert cfg.compute_padded_head_dim(0) == 32

    def test_compute_padded_head_dim_custom_simd(self):
        cfg = PackedKVConfig(simd_width=16)
        assert cfg.compute_padded_head_dim(48) == 48  # 48 % 16 == 0
        assert cfg.compute_padded_head_dim(50) == 64  # 50 → 64 (next 16 multiple)

    def test_to_dict(self):
        cfg = PackedKVConfig(enabled=True, quantize_bits=4)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["quantize_bits"] == 4


class TestPackedKVStats:
    def test_initial_state(self):
        s = PackedKVStats()
        assert s.total_conversions == 0
        assert s.memory_overhead_pct == 0.0
        assert s.avg_expansion_ratio == 1.0
        assert s.hit_rate == 0.0

    def test_record_conversion(self):
        s = PackedKVStats()
        s.total_conversions = 1
        s.total_bytes_standard = 1000
        s.total_bytes_packed = 1200
        s.total_pad_overhead_bytes = 200
        assert s.memory_overhead_pct == 20.0
        assert s.avg_expansion_ratio == 1.2

    def test_hit_rate(self):
        s = PackedKVStats()
        s.cache_hits = 7
        s.cache_misses = 3
        assert s.hit_rate == 0.7

    def test_get_stats(self):
        s = PackedKVStats()
        s.total_conversions = 5
        stats = s.get_stats()
        assert stats["total_conversions"] == 5
        assert "memory_overhead_pct" in stats
        assert "hit_rate" in stats

    def test_reset(self):
        s = PackedKVStats()
        s.total_conversions = 10
        s.cache_hits = 5
        s.reset()
        assert s.total_conversions == 0
        assert s.cache_hits == 0


class TestPackedKVCache:
    def test_creation_default(self):
        cache = PackedKVCache()
        assert not cache.config.enabled

    def test_creation_enabled(self):
        cache = PackedKVCache(PackedKVConfig(enabled=True))
        assert cache.config.enabled

    def test_pad_head_dim_no_pad_needed(self):
        cache = PackedKVCache()
        import mlx.core as mx
        t = mx.zeros((2, 3, 32))  # head_dim=32, already aligned
        result = cache.pad_head_dim(t, 32)
        assert result.shape == (2, 3, 32)

    def test_pad_head_dim(self):
        cache = PackedKVCache()
        import mlx.core as mx
        t = mx.zeros((2, 3, 24))  # head_dim=24, pad to 32
        result = cache.pad_head_dim(t, 32)
        assert result.shape == (2, 3, 32)

    def test_pack_kv_layer_interleaved(self):
        cache = PackedKVCache(PackedKVConfig(interleave_kv=True))
        import mlx.core as mx
        keys = mx.zeros((10, 8, 64))   # [seq, heads, dim]
        values = mx.zeros((10, 8, 64))
        packed = cache.pack_kv_layer(keys, values)
        assert packed.shape == (2, 10, 8, 64)  # [K/V, seq, heads, dim]
        assert cache.stats.total_conversions == 1

    def test_pack_kv_layer_concat(self):
        cache = PackedKVCache(PackedKVConfig(interleave_kv=False))
        import mlx.core as mx
        keys = mx.zeros((10, 8, 64))
        values = mx.zeros((10, 8, 64))
        packed = cache.pack_kv_layer(keys, values)
        assert packed.shape == (20, 8, 64)  # [2*seq, heads, dim]

    def test_pack_kv_layer_with_padding(self):
        cache = PackedKVCache(PackedKVConfig(simd_width=32))
        import mlx.core as mx
        keys = mx.zeros((10, 8, 48))    # head_dim=48 → pad to 64
        values = mx.zeros((10, 8, 48))
        packed = cache.pack_kv_layer(keys, values)
        assert packed.shape == (2, 10, 8, 64)  # padded to 64

    def test_unpack_kv_layer_interleaved(self):
        cache = PackedKVCache(PackedKVConfig(interleave_kv=True))
        import mlx.core as mx
        keys = mx.ones((10, 8, 64))
        values = mx.zeros((10, 8, 64))
        packed = cache.pack_kv_layer(keys, values)
        unpacked_k, unpacked_v = cache.unpack_kv_layer(packed, 64)
        assert unpacked_k.shape == (10, 8, 64)
        assert unpacked_v.shape == (10, 8, 64)

    def test_unpack_removes_padding(self):
        cache = PackedKVCache(PackedKVConfig(interleave_kv=True, simd_width=32))
        import mlx.core as mx
        keys = mx.ones((10, 8, 48))
        values = mx.zeros((10, 8, 48))
        packed = cache.pack_kv_layer(keys, values)
        assert packed.shape[-1] == 64  # padded
        unpacked_k, unpacked_v = cache.unpack_kv_layer(packed, 48)
        assert unpacked_k.shape == (10, 8, 48)  # unpadded
        assert unpacked_v.shape == (10, 8, 48)

    def test_roundtrip_interleaved(self):
        cache = PackedKVCache(PackedKVConfig(interleave_kv=True))
        import mlx.core as mx
        keys = mx.random.normal((10, 4, 64))
        values = mx.random.normal((10, 4, 64))
        packed = cache.pack_kv_layer(keys, values)
        unpacked_k, unpacked_v = cache.unpack_kv_layer(packed, 64)
        assert mx.allclose(keys, unpacked_k, atol=1e-6)
        assert mx.allclose(values, unpacked_v, atol=1e-6)

    def test_roundtrip_concat(self):
        cache = PackedKVCache(PackedKVConfig(interleave_kv=False))
        import mlx.core as mx
        keys = mx.random.normal((10, 4, 64))
        values = mx.random.normal((10, 4, 64))
        packed = cache.pack_kv_layer(keys, values)
        unpacked_k, unpacked_v = cache.unpack_kv_layer(packed, 64)
        assert mx.allclose(keys, unpacked_k, atol=1e-6)
        assert mx.allclose(values, unpacked_v, atol=1e-6)

    def test_cache_packed_block(self):
        cache = PackedKVCache()
        import mlx.core as mx
        arr = mx.zeros((2, 10, 8, 64))
        cache.cache_packed_block(12345, arr)
        assert cache.get_cached_block(12345) is not None
        assert cache.stats.cache_hits == 1

    def test_cache_miss(self):
        cache = PackedKVCache()
        assert cache.get_cached_block(999) is None
        assert cache.stats.cache_misses == 1

    def test_clear_cache(self):
        cache = PackedKVCache()
        import mlx.core as mx
        cache.cache_packed_block(1, mx.zeros((2,)))
        cache.clear_cache()
        assert cache.get_cached_block(1) is None

    def test_compute_memory_layout(self):
        cache = PackedKVCache(PackedKVConfig(simd_width=32))
        layout = cache.compute_memory_layout(
            num_blocks=1024,
            num_layers=24,
            block_size=64,
            num_kv_heads=8,
            head_dim=64,
        )
        assert layout["padded_head_dim"] == 64
        assert layout["standard_bytes_per_block"] > 0
        assert layout["packed_bytes_per_block"] > 0
        assert layout["total_standard_bytes"] > 0
        assert layout["simd_width"] == 32

    def test_compute_memory_layout_with_padding(self):
        cache = PackedKVCache(PackedKVConfig(simd_width=32))
        layout = cache.compute_memory_layout(
            num_blocks=1024,
            num_layers=24,
            block_size=64,
            num_kv_heads=8,
            head_dim=48,  # Needs padding to 64
        )
        assert layout["padded_head_dim"] == 64
        assert layout["pad_overhead_pct"] > 0

    def test_compute_memory_layout_quantized(self):
        cache = PackedKVCache(PackedKVConfig(simd_width=32, quantize_bits=4))
        layout = cache.compute_memory_layout(
            num_blocks=1024,
            num_layers=24,
            block_size=64,
            num_kv_heads=8,
            head_dim=64,
        )
        # 4-bit should be 1/4 of FP16
        assert layout["packed_bytes_per_block"] < layout["standard_bytes_per_block"]

    def test_get_stats(self):
        cache = PackedKVCache()
        stats = cache.get_stats()
        assert "config" in stats
        assert "stats" in stats
        assert "cached_blocks" in stats

    def test_reset(self):
        cache = PackedKVCache()
        import mlx.core as mx
        cache.pack_kv_layer(mx.zeros((2, 2, 2)), mx.zeros((2, 2, 2)))
        cache.cache_packed_block(1, mx.zeros((2,)))
        cache.reset()
        assert cache.stats.total_conversions == 0
        assert cache.get_cached_block(1) is None


class TestComputeSimdAlignedDim:
    def test_aligned(self):
        assert compute_simd_aligned_dim(64) == 64

    def test_needs_padding(self):
        assert compute_simd_aligned_dim(48) == 64

    def test_zero(self):
        assert compute_simd_aligned_dim(0) == 32

    def test_custom_width(self):
        assert compute_simd_aligned_dim(50, simd_width=16) == 64

    def test_already_aligned_custom(self):
        assert compute_simd_aligned_dim(48, simd_width=16) == 48
