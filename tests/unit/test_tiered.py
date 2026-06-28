"""Tests for tiered KV cache, SSD store, warm tier integration, and background flush."""

import tempfile
import time

import numpy as np

from yunshu_kv.manager import KVCacheConfig, KVCacheManager
from yunshu_kv.tiered import BackgroundSSDFlush, SSDCacheStore, TieredKVCacheManager
from yunshu_kv.warm_tier import KVTierConfig, KVWarmTier


class TestSSDCacheStore:
    """Test SSD-backed cache block storage."""

    def test_create_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir, max_size_bytes=1024**3)
            assert store.get_stats()["num_entries"] == 0

    def test_store_and_contains(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir)
            data = mx.zeros((2, 8, 64, 128), dtype=mx.float16)
            assert store.store(12345, data, num_tokens=64)
            assert store.contains(12345)
            assert not store.contains(99999)

    def test_store_and_load(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir)
            data = mx.ones((2, 8, 64, 128), dtype=mx.float16)
            assert store.store(12345, data, num_tokens=64)
            loaded = store.load(12345)
            assert loaded is not None
            # Shape should match after roundtrip
            assert loaded.shape == data.shape

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir)
            assert store.load(99999) is None

    def test_per_kv_scales_preserve_small_value_tensor(self):
        """the [2,…] block (K=[0], V=[1]) must use PER-SLICE int8 scales. A single
        shared scale (the old bug) was dominated by the larger tensor and crushed the smaller
        one (~79% error on V). With per-slice scales both round-trip cleanly."""
        import mlx.core as mx
        import numpy as np
        rng = np.random.default_rng(0)
        k = (rng.standard_normal((2, 8, 64)).astype(np.float16)) * np.float16(2.0)   # large
        v = (rng.standard_normal((2, 8, 64)).astype(np.float16)) * np.float16(0.01)  # tiny
        block = mx.array(np.stack([k, v], axis=0))  # [2, 2, 8, 64]
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir)
            assert store.store(777, block, num_tokens=64)
            if getattr(store, "_flush_pending_io", None):
                store._flush_pending_io()
            out = np.array(store.load(777))
        ref = np.array(block)
        # V slice (index 1) relative error must be small (was ~0.79 with one shared scale).
        v_rel = np.linalg.norm(out[1] - ref[1]) / (np.linalg.norm(ref[1]) + 1e-12)
        k_rel = np.linalg.norm(out[0] - ref[0]) / (np.linalg.norm(ref[0]) + 1e-12)
        assert v_rel < 0.05, f"V rel-error too high: {v_rel}"
        assert k_rel < 0.05, f"K rel-error too high: {k_rel}"

    def test_double_store(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir)
            data = mx.zeros((2, 8, 64, 128), dtype=mx.float16)
            store.store(12345, data, 64)
            store.store(12345, data, 64)  # Should update access, not duplicate
            assert store.get_stats()["num_entries"] == 1

    def test_persistence(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = SSDCacheStore(tmpdir)
            data = mx.zeros((2, 8, 64, 128), dtype=mx.float16)
            store1.store(12345, data, 64)
            store1._save_index()

            # Create new store from same dir
            store2 = SSDCacheStore(tmpdir)
            assert store2.contains(12345)
            assert store2.get_stats()["num_entries"] == 1

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SSDCacheStore(tmpdir, max_size_bytes=1024**3)
            stats = store.get_stats()
            assert "num_entries" in stats
            assert "size_bytes" in stats
            assert "max_size_bytes" in stats
            assert stats["utilization_pct"] == 0.0


class TestTieredKVCacheManager:
    """Test tiered cache coordination with warm + SSD tiers."""

    def test_create_tiered(self):
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        tiered = TieredKVCacheManager(hot)
        stats = tiered.get_stats()
        assert "hot_usage_pct" in stats

    def test_create_with_warm_tier(self):
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        warm = KVWarmTier(KVTierConfig(max_blocks=50))
        tiered = TieredKVCacheManager(hot, warm_tier=warm)
        stats = tiered.get_stats()
        assert "hot_usage_pct" in stats
        assert "warm" in stats
        assert stats["warm"]["num_blocks"] == 0

    def test_allocate_with_no_ssd(self):
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        tiered = TieredKVCacheManager(hot)
        tokens = list(range(20))  # 5 blocks of 4 tokens
        table, match = tiered.allocate_for_prefill(tokens)
        assert table is not None
        assert match.num_matched_tokens == 0  # First time, no cache

    def test_free_request(self):
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        tiered = TieredKVCacheManager(hot)
        tokens = list(range(12))
        table, _ = tiered.allocate_for_prefill(tokens)
        tiered.free_request(table)
        # Blocks freed (some may be retained in cache)
        assert hot.num_free_blocks >= 90

    def test_with_ssd_store(self):
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            tiered = TieredKVCacheManager(hot, ssd)
            stats = tiered.get_stats()
            assert "ssd" in stats

    def test_with_all_tiers(self):
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        warm = KVWarmTier(KVTierConfig(max_blocks=50))
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            tiered = TieredKVCacheManager(hot, ssd_store=ssd, warm_tier=warm)
            stats = tiered.get_stats()
            assert "hot_usage_pct" in stats
            assert "warm" in stats
            assert "ssd" in stats

    def test_extract_kv_for_block_returns_none_when_no_layers(self):
        """_extract_kv_for_block returns None when hot manager has no KV layers."""
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        tiered = TieredKVCacheManager(hot)
        from yunshu_kv.block import KVBlock
        block = KVBlock(block_id=0)
        assert tiered._extract_kv_for_block(block) is None

    def test_warm_tier_none_not_in_stats(self):
        """When warm_tier is None, 'warm' key should not be in stats."""
        config = KVCacheConfig(
            block_size=4,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
        hot = KVCacheManager(config, num_blocks=100)
        tiered = TieredKVCacheManager(hot)
        stats = tiered.get_stats()
        assert "warm" not in stats


class TestKVWarmTier:
    """Test KVWarmTier (warm_tier.py module) — consolidated from WarmTierManager."""

    def test_construction(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        assert wt.num_blocks == 0
        assert wt.is_full is False

    def test_demote_with_num_tokens_preserved(self):
        """Regression test: demote with num_tokens stores it for SSD flush."""
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        data = np.random.randn(2, 64).astype(np.float32)
        assert wt.demote(0xBEEF, data, num_tokens=64) is True
        entry = wt._store[0xBEEF]
        assert len(entry) >= 4
        assert entry[3] == 64  # num_tokens preserved

    def test_demote_re_insert_preserves_num_tokens(self):
        """Regression test: re-inserting after failed promotion preserves num_tokens.

        Before the fix, TieredKVCacheManager re-inserted without num_tokens,
        causing SSD flush to store 0 tokens (block invisible to prefix matching).
        """
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        data = np.random.randn(2, 64).astype(np.float32)
        wt.demote(0xBEEF, data, num_tokens=64)
        # Simulate re-insert (the bug was calling demote without num_tokens)
        promoted = wt.promote(0xBEEF)
        assert promoted is not None
        # Re-insert with num_tokens
        wt.demote(0xBEEF, promoted, num_tokens=64)
        entry = wt._store[0xBEEF]
        assert entry[3] == 64

    def test_demote_and_contains(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        data = np.random.randn(2, 64).astype(np.float32)
        assert wt.demote(0xABCD, data) is True
        assert wt.contains(0xABCD) is True
        assert wt.num_blocks == 1

    def test_promote(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        data = np.random.randn(2, 64).astype(np.float32)
        wt.demote(0xABCD, data)
        result = wt.promote(0xABCD)
        assert result is not None
        assert wt.num_blocks == 0

    def test_promote_missing(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        assert wt.promote(0x1234) is None

    def test_eviction(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=2))
        data = np.random.randn(2, 64).astype(np.float32)
        wt.demote(1, data)
        wt.demote(2, data)
        assert wt.is_full is True
        wt.demote(3, data)  # Should evict block 1 (LRU)
        assert wt.contains(1) is False
        assert wt.contains(2) is True
        assert wt.contains(3) is True

    def test_get_stats(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        data = np.random.randn(2, 64).astype(np.float32)
        wt.demote(1, data)
        stats = wt.get_stats()
        assert stats["num_blocks"] == 1
        assert stats["utilization_pct"] == 1.0
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_lru_ordering(self):
        """Verify LRU: promoting a block moves it to end, evict removes oldest."""
        wt = KVWarmTier(KVTierConfig(max_blocks=3))
        data = np.random.randn(2, 64).astype(np.float32)
        wt.demote(1, data)
        wt.demote(2, data)
        wt.demote(3, data)
        assert wt.is_full
        # Promote block 1 (moves it to MRU position)
        wt.promote(1)
        # Now block 2 is LRU
        wt.demote(4, data)  # Should evict block 2 (oldest after promote)
        assert wt.contains(1) is False  # Was promoted (removed from warm)
        assert wt.contains(2) is True  # Still there (block 1 was promoted, not just touched)
        # Note: promote removes from warm tier, so 1 was already gone

    def test_manual_evict(self):
        wt = KVWarmTier(KVTierConfig(max_blocks=100))
        data = np.random.randn(2, 64).astype(np.float32)
        wt.demote(1, data)
        wt.demote(2, data)
        evicted = wt.evict(1)
        assert evicted == 1
        assert wt.num_blocks == 1


class TestBackgroundSSDFlush:
    """Test background SSD flush thread."""

    def test_create_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            warm = KVWarmTier(KVTierConfig())
            flusher = BackgroundSSDFlush(ssd, warm)
            stats = flusher.get_stats()
            assert stats["flush_count"] == 0
            assert stats["running"] is False

    def test_start_and_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            warm = KVWarmTier(KVTierConfig())
            flusher = BackgroundSSDFlush(ssd, warm, flush_interval_s=1.0)
            flusher.start()
            assert flusher.get_stats()["running"] is True
            flusher.stop()
            assert flusher.get_stats()["running"] is False

    def test_flush_now_empty(self):
        """Flush on empty warm tier should flush 0 blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            warm = KVWarmTier(KVTierConfig())
            flusher = BackgroundSSDFlush(ssd, warm)
            flushed = flusher.flush_now()
            assert flushed == 0

    def test_flush_now_with_data(self):
        """Flush should persist warm tier blocks to SSD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            warm = KVWarmTier(KVTierConfig())
            data = np.random.randn(2, 64).astype(np.float32)
            warm.demote(0xBEEF, data)

            flusher = BackgroundSSDFlush(ssd, warm)
            flushed = flusher.flush_now()
            assert flushed == 1
            assert ssd.contains(0xBEEF)

    def test_flush_idempotent(self):
        """Flushing twice should not duplicate entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            warm = KVWarmTier(KVTierConfig())
            data = np.random.randn(2, 64).astype(np.float32)
            warm.demote(0xBEEF, data)

            flusher = BackgroundSSDFlush(ssd, warm)
            flusher.flush_now()
            flusher.flush_now()  # Second flush — already in SSD
            assert ssd.get_stats()["num_entries"] == 1

    def test_thread_safety(self):
        """Flush thread should start/stop cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd = SSDCacheStore(tmpdir)
            warm = KVWarmTier(KVTierConfig())
            flusher = BackgroundSSDFlush(ssd, warm, flush_interval_s=0.1)
            flusher.start()
            time.sleep(0.3)  # Let it run a few cycles
            flusher.stop()
            assert flusher.get_stats()["running"] is False
            assert flusher.get_stats()["flush_count"] > 0
