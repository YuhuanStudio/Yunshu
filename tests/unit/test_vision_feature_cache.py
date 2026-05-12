"""Tests for VisionFeatureCache — hashing, LRU, and memory-only mode."""

import time
from unittest.mock import MagicMock

import pytest

from yunshu_engine.vision_feature_cache import (
    VisionFeatureCache,
    _composite_hash,
    _composite_key,
    compute_image_hash,
)


class TestHashFunctions:
    def test_compute_image_hash_deterministic(self):
        data = b"hello world"
        h1 = compute_image_hash(data)
        h2 = compute_image_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest

    def test_compute_image_hash_different_inputs(self):
        h1 = compute_image_hash(b"foo")
        h2 = compute_image_hash(b"bar")
        assert h1 != h2

    def test_composite_key(self):
        key = _composite_key("my-model", "abc123")
        assert key == "my-model:abc123"

    def test_composite_hash_deterministic(self):
        h1 = _composite_hash("model", "hash")
        h2 = _composite_hash("model", "hash")
        assert h1 == h2

    def test_composite_hash_different_models(self):
        h1 = _composite_hash("model-a", "hash")
        h2 = _composite_hash("model-b", "hash")
        assert h1 != h2


class TestVisionFeatureCacheMemoryOnly:
    """Test memory-only mode (no SSD)."""

    def test_miss_returns_none(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        assert cache.get("nonexistent", "model") is None
        assert cache.stats["misses"] == 1

    def test_put_and_get(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        features = MagicMock()
        cache.put("img1", "model-a", features)
        result = cache.get("img1", "model-a")
        assert result is features
        assert cache.stats["hits"] == 1
        assert cache.stats["saves"] == 1

    def test_lru_eviction(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=2)
        cache.put("img1", "model", "feat1")
        cache.put("img2", "model", "feat2")
        cache.put("img3", "model", "feat3")
        # img1 should be evicted
        assert cache.get("img1", "model") is None
        assert cache.get("img2", "model") == "feat2"
        assert cache.get("img3", "model") == "feat3"

    def test_lru_access_renews(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=2)
        cache.put("img1", "model", "feat1")
        cache.put("img2", "model", "feat2")
        # Access img1 to renew it
        cache.get("img1", "model")
        # Add img3 — img2 should be evicted (LRU), not img1
        cache.put("img3", "model", "feat3")
        assert cache.get("img1", "model") == "feat1"
        assert cache.get("img2", "model") is None
        assert cache.get("img3", "model") == "feat3"

    def test_put_overwrites_existing(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        cache.put("img1", "model", "old")
        cache.put("img1", "model", "new")
        assert cache.get("img1", "model") == "new"
        assert cache.stats["saves"] == 2

    def test_different_models_same_hash(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        cache.put("img1", "model-a", "feat-a")
        cache.put("img1", "model-b", "feat-b")
        assert cache.get("img1", "model-a") == "feat-a"
        assert cache.get("img1", "model-b") == "feat-b"

    def test_close_is_safe(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        cache.close()  # Should not crash

    def test_stats(self):
        cache = VisionFeatureCache(cache_dir=None, max_memory_entries=5)
        cache.put("img1", "model", "feat")
        cache.get("img1", "model")
        cache.get("img2", "model")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["saves"] == 1


class TestVisionFeatureCacheSafetensorsWrite:
    """Test the safetensors binary write format."""

    def test_write_safetensors_returns_size(self, tmp_path):
        cache = VisionFeatureCache(cache_dir=str(tmp_path), max_memory_entries=5)
        tensors = {
            "feature": (b"\x00\x01\x02\x03", "F16", [2]),
        }
        metadata = {"model_name": "test"}
        size = cache._write_safetensors(str(tmp_path / "test.safetensors"), tensors, metadata)
        assert size > 0
        assert (tmp_path / "test.safetensors").exists()
        assert (tmp_path / "test.safetensors").stat().st_size == size
