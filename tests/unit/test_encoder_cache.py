"""Tests for encoder_cache.py — encoder output caching for VLM/encoder-decoder models."""

import time

from yunshu_engine.encoder_cache import EncoderCacheManager


class TestEncoderCacheManager:
    def test_put_and_get(self):
        mgr = EncoderCacheManager()
        mgr.put("r1", [1, 2, 3])
        result = mgr.get("r1")
        assert result is not None
        assert result == [1, 2, 3]

    def test_get_nonexistent(self):
        mgr = EncoderCacheManager()
        assert mgr.get("nonexistent") is None

    def test_evict(self):
        mgr = EncoderCacheManager()
        mgr.put("r1", [1, 2, 3])
        evicted = mgr.evict("r1")
        assert evicted
        assert mgr.get("r1") is None

    def test_evict_nonexistent(self):
        mgr = EncoderCacheManager()
        evicted = mgr.evict("nonexistent")
        assert not evicted

    def test_clear(self):
        mgr = EncoderCacheManager()
        mgr.put("r1", [1])
        mgr.put("r2", [2])
        mgr.clear()
        assert mgr.get("r1") is None
        assert mgr.get("r2") is None

    def test_evict_all_expired(self):
        mgr = EncoderCacheManager(ttl_seconds=0.01)
        mgr.put("r1", [1])
        time.sleep(0.02)
        evicted = mgr.evict_all_expired()
        assert evicted >= 1
        assert mgr.get("r1") is None

    def test_capacity_eviction(self):
        mgr = EncoderCacheManager(max_entries=2)
        mgr.put("r1", [1])
        mgr.put("r2", [2])
        mgr.put("r3", [3])
        # Should have evicted oldest
        assert mgr.get("r1") is None
        assert mgr.get("r2") is not None
        assert mgr.get("r3") is not None

    def test_get_stats(self):
        mgr = EncoderCacheManager()
        mgr.put("r1", [1, 2, 3])
        stats = mgr.get_stats()
        assert stats["num_entries"] == 1

    def test_overwrite_existing(self):
        mgr = EncoderCacheManager()
        mgr.put("r1", [1])
        mgr.put("r1", [2])
        result = mgr.get("r1")
        assert result == [2]
