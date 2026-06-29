"""Tests for SSD-tier KV cache."""

import os
import tempfile

import mlx.core as mx

from yunshu_engine.ssd_kv_cache import (
    SSDCacheStats,
    SSDKVCache,
    _extract_tensor_bytes,
)


class TestExtractTensorBytes:
    def test_float16(self):
        arr = mx.array([1.0, 2.0, 3.0], dtype=mx.float16)
        raw, dtype, shape = _extract_tensor_bytes(arr)
        assert dtype == "F16"
        assert shape == [3]
        assert len(raw) == 6

    def test_float32(self):
        arr = mx.array([1.0, 2.0], dtype=mx.float32)
        raw, dtype, shape = _extract_tensor_bytes(arr)
        assert dtype == "F32"
        assert shape == [2]

    def test_shape_preserved(self):
        arr = mx.zeros((2, 4, 8), dtype=mx.float16)
        raw, dtype, shape = _extract_tensor_bytes(arr)
        assert shape == [2, 4, 8]


def _make_cache(tmp_dir):
    """Create an SSD cache with automatic cleanup."""
    cache = SSDKVCache(cache_dir=tmp_dir, max_size_bytes=10 * 1024**2)
    return cache


class TestSSDKVCacheInit:
    def test_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "kv-test")
            cache = _make_cache(cache_dir)
            assert os.path.isdir(cache_dir)
            for h in "0123456789abcdef":
                assert os.path.isdir(os.path.join(cache_dir, h))
            cache.close()

    def test_initial_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            stats = cache.get_stats()
            assert stats.total_entries == 0
            assert stats.hot_cache_entries == 0
            cache.close()


class TestSSDKVCacheSaveLoad:
    def test_save_and_has_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            block_hash = b"\x01" * 16
            data = [mx.zeros((1, 4, 8)), mx.ones((1, 4, 8))]
            cache.save_block(block_hash, data, token_count=64, model_name="test")
            assert cache.has_block(block_hash)
            cache.close()

    def test_load_from_hot_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            block_hash = b"\x02" * 16
            data = [mx.zeros((1, 4, 8))]
            cache.save_block(block_hash, data, token_count=32)
            result = cache.load_block(block_hash)
            assert result is not None
            cache.close()

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            result = cache.load_block(b"\xff" * 16)
            assert result is None
            cache.close()

    def test_has_block_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            assert not cache.has_block(b"\x00" * 16)
            cache.close()

    def test_delete_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            block_hash = b"\x03" * 16
            data = [mx.zeros((2, 4))]
            cache.save_block(block_hash, data, token_count=32)
            assert cache.has_block(block_hash)
            cache.delete_block(block_hash)
            assert not cache.has_block(block_hash)
            cache.close()

    def test_clear_removes_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            for i in range(3):
                bh = bytes([i]) * 16
                cache.save_block(bh, [mx.zeros((2, 4))], token_count=16)
            count = cache.clear()
            assert count == 3
            assert cache.get_stats().total_entries == 0
            cache.close()


class TestSSDKVCacheStats:
    def test_stats_after_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            cache.save_block(b"\x01" * 16, [mx.zeros((2, 4))], token_count=16)
            stats = cache.get_stats()
            assert stats.total_entries == 1
            assert stats.hot_cache_entries == 1
            cache.close()

    def test_stats_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            stats = cache.get_stats()
            assert isinstance(stats, SSDCacheStats)
            assert hasattr(stats, "hot_cache_entries")
            assert hasattr(stats, "disk_entries")
            assert hasattr(stats, "evictions")
            cache.close()


class TestSSDKVCacheRecovery:
    def test_recover_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Write a block
            cache1 = _make_cache(tmp)
            block_hash = b"\xab" * 16
            cache1.save_block(
                block_hash, [mx.zeros((2, 4))], token_count=16, model_name="test"
            )
            cache1.close()

            # Simulate restart — create new instance pointing at same dir
            cache2 = _make_cache(tmp)
            assert cache2.has_block(block_hash)
            stats = cache2.get_stats()
            assert stats.total_entries == 1
            cache2.close()

    def test_load_corrupt_scale_factors_returns_none(self):
        """Regression test: missing scale factors should return None, not silently
        scale by 1.0 (which produces garbage output).

        Before the fix, scale_factors.get(key, 1.0) silently used 1.0 as default,
        producing incorrectly dequantized data. Now it returns None (block treated
        as corrupt).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            block_hash = b"\xcc" * 16
            data = [mx.zeros((2, 4))]
            cache.save_block(block_hash, data, token_count=16)
            # Wait for write to complete
            cache._process_pending_writes()

            # Corrupt the safetensors file by modifying scale_factors metadata
            import json
            import struct

            block_hash.hex()
            file_path = cache._block_path(block_hash)
            with open(file_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_json = f.read(header_size)
                tensor_data = f.read()

            header = json.loads(header_json)
            # Remove scale_factors from __metadata__ to simulate corruption
            header.get("__metadata__", {}).pop("scale_factors", None)
            corrupt_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
            padding = (8 - (len(corrupt_json) % 8)) % 8
            corrupt_json += b" " * padding

            with open(file_path, "wb") as f:
                f.write(struct.pack("<Q", len(corrupt_json)))
                f.write(corrupt_json)
                f.write(tensor_data)

            # Clear hot cache so it reads from disk
            cache._hot_cache.clear()

            result = cache.load_block(block_hash)
            assert result is None  # Should detect corruption, not return garbage
            cache.close()
