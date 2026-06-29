"""Tests for MLX cache integration layer."""

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from yunshu_kv.mlx_cache import (
    CacheType,
    concatenate_kv_blocks,
    detect_cache_type,
    extract_cache_state,
    get_cache_seq_length,
    is_sliceable,
    reconstruct_kvcache,
    slice_kv_at_offsets,
)

# ── Mock cache objects for testing without a loaded model ──


class MockKVCache:
    """Mimics mlx-lm KVCache for testing."""

    def __init__(self, keys=None, values=None, offset=0):
        self.keys = keys
        self.values = values
        self.offset = offset

    @property
    def state(self):
        return (self.keys, self.values)


class MockRotatingKVCache:
    """Mimics mlx-lm RotatingKVCache."""

    def __init__(self, keys=None, values=None, offset=0, max_size=512, keep=0, idx=0):
        self.keys = keys
        self.values = values
        self.offset = offset
        self.max_size = max_size
        self.keep = keep
        self._idx = idx
        self.meta_state = "test"

    @property
    def state(self):
        return (self.keys, self.values)


class MockArraysCache:
    """Mimics mlx-lm ArraysCache."""

    def __init__(self, arrays=None):
        self.cache = arrays or []


class MockCacheList:
    """Mimics CacheList wrapper."""

    def __init__(self, caches=None):
        self.caches = caches or []


# ── detect_cache_type ──


class TestDetectCacheType:
    def test_kvcache(self):
        assert detect_cache_type(MockKVCache()) == CacheType.KVCACHE

    def test_rotating_kvcache(self):
        assert detect_cache_type(MockRotatingKVCache()) == CacheType.ROTATING_KVCACHE

    def test_arrays_cache(self):
        assert detect_cache_type(MockArraysCache()) == CacheType.ARRAYS_CACHE

    def test_cache_list(self):
        assert detect_cache_type(MockCacheList()) == CacheType.CACHE_LIST

    def test_unknown_object(self):
        class Foo:
            pass

        assert detect_cache_type(Foo()) == CacheType.UNKNOWN

    def test_heuristic_kv(self):
        """Object with keys+values but not a known class name."""

        class CustomKV:
            def __init__(self):
                self.keys = None
                self.values = None

        assert detect_cache_type(CustomKV()) == CacheType.KVCACHE

    def test_heuristic_rotating(self):
        class CustomRotating:
            def __init__(self):
                self.max_size = 128
                self._idx = 0

        assert detect_cache_type(CustomRotating()) == CacheType.ROTATING_KVCACHE


# ── is_sliceable ──


class TestIsSliceable:
    def test_kvcache_sliceable(self):
        assert is_sliceable(MockKVCache())

    def test_rotating_not_sliceable(self):
        assert not is_sliceable(MockRotatingKVCache())

    def test_arrays_not_sliceable(self):
        assert not is_sliceable(MockArraysCache())


# ── extract_cache_state ──


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestExtractCacheState:
    def test_kvcache_state(self):
        keys = mx.zeros((1, 8, 32, 64))
        values = mx.zeros((1, 8, 32, 64))
        cache = MockKVCache(keys=keys, values=values, offset=32)
        state = extract_cache_state(cache)
        assert state["cache_type"] == CacheType.KVCACHE
        assert state["offset"] == 32
        assert state["keys"].shape == (1, 8, 32, 64)

    def test_rotating_state(self):
        keys = mx.zeros((1, 8, 64, 64))
        values = mx.zeros((1, 8, 64, 64))
        cache = MockRotatingKVCache(keys=keys, values=values, offset=64, max_size=256)
        state = extract_cache_state(cache)
        assert state["cache_type"] == CacheType.ROTATING_KVCACHE
        assert state["max_size"] == 256
        assert state["_idx"] == 0

    def test_arrays_state(self):
        arr1 = mx.zeros((10, 10))
        arr2 = mx.ones((10, 10))
        cache = MockArraysCache(arrays=[arr1, arr2])
        state = extract_cache_state(cache)
        assert state["cache_type"] == CacheType.ARRAYS_CACHE
        assert len(state["arrays"]) == 2

    def test_cache_list_state(self):
        inner1 = MockKVCache(
            keys=mx.zeros((1, 4, 16, 32)), values=mx.zeros((1, 4, 16, 32)), offset=16
        )
        inner2 = MockKVCache(
            keys=mx.zeros((1, 4, 16, 32)), values=mx.zeros((1, 4, 16, 32)), offset=16
        )
        cache_list = MockCacheList(caches=[inner1, inner2])
        state = extract_cache_state(cache_list)
        assert state["cache_type"] == CacheType.CACHE_LIST
        assert len(state["sub_states"]) == 2


# ── slice_kv_at_offsets ──


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestSliceKV:
    def test_basic_slice(self):
        keys = mx.arange(64).reshape(1, 1, 64, 1)
        values = mx.arange(64).reshape(1, 1, 64, 1) + 100
        sk, sv = slice_kv_at_offsets(keys, values, 10, 20)
        assert sk.shape == (1, 1, 10, 1)
        assert int(sk[0, 0, 0, 0]) == 10
        assert int(sk[0, 0, 9, 0]) == 19
        assert int(sv[0, 0, 0, 0]) == 110

    def test_full_range(self):
        keys = mx.ones((2, 4, 32, 64))
        values = mx.ones((2, 4, 32, 64)) * 2
        sk, sv = slice_kv_at_offsets(keys, values, 0, 32)
        assert sk.shape == keys.shape

    def test_single_token_slice(self):
        keys = mx.ones((1, 8, 128, 64))
        values = mx.ones((1, 8, 128, 64))
        sk, sv = slice_kv_at_offsets(keys, values, 50, 51)
        assert sk.shape == (1, 8, 1, 64)


# ── concatenate_kv_blocks ──


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestConcatenateKV:
    def test_two_blocks(self):
        k1 = mx.ones((1, 4, 16, 32))
        v1 = mx.ones((1, 4, 16, 32)) * 2
        k2 = mx.ones((1, 4, 8, 32)) * 3
        v2 = mx.ones((1, 4, 8, 32)) * 4
        ck, cv = concatenate_kv_blocks([(k1, v1), (k2, v2)])
        assert ck.shape == (1, 4, 24, 32)
        assert cv.shape == (1, 4, 24, 32)

    def test_three_blocks(self):
        blocks = [(mx.ones((1, 2, 5, 16)), mx.ones((1, 2, 5, 16))) for _ in range(3)]
        ck, cv = concatenate_kv_blocks(blocks)
        assert ck.shape == (1, 2, 15, 16)


# ── reconstruct_kvcache ──


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestReconstructKVCache:
    def test_basic_reconstruct(self):
        keys = mx.ones((1, 8, 32, 64))
        values = mx.ones((1, 8, 32, 64)) * 2
        cache = reconstruct_kvcache(keys, values)
        assert cache.offset == 32  # defaults to shape[2]
        assert cache.keys is keys
        assert cache.values is values

    def test_custom_offset(self):
        keys = mx.ones((1, 8, 64, 64))
        values = mx.ones((1, 8, 64, 64))
        cache = reconstruct_kvcache(keys, values, offset=100)
        assert cache.offset == 100

    def test_zero_offset(self):
        keys = mx.zeros((1, 4, 0, 32))
        values = mx.zeros((1, 4, 0, 32))
        cache = reconstruct_kvcache(keys, values)
        assert cache.offset == 0


# ── get_cache_seq_length ──


class TestGetCacheSeqLength:
    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_kvcache_length(self):
        cache = MockKVCache(
            keys=mx.zeros((1, 4, 32, 64)), values=mx.zeros((1, 4, 32, 64)), offset=32
        )
        assert get_cache_seq_length(cache) == 32

    def test_rotating_length(self):
        cache = MockRotatingKVCache(offset=128)
        assert get_cache_seq_length(cache) == 128

    def test_cache_list_length(self):
        inner1 = MockKVCache(offset=20)
        inner2 = MockKVCache(offset=30)
        cache_list = MockCacheList(caches=[inner1, inner2])
        assert get_cache_seq_length(cache_list) == 30

    def test_empty_cache_list(self):
        cache_list = MockCacheList(caches=[])
        assert get_cache_seq_length(cache_list) == 0

    def test_unknown_returns_zero(self):
        class Foo:
            pass

        assert get_cache_seq_length(Foo()) == 0
