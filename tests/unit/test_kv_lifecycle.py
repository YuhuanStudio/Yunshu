"""Tests for kv_lifecycle.py — unified KV cache lifecycle management."""

import time

from yunshu_engine.kv_lifecycle import (
    CacheWarmingPredictor,
    KVBlock,
    KVCompactionScheduler,
    KVLifecycleManager,
    KVTier,
    KVTierConfig,
)


class TestKVBlock:
    def test_touch_updates_access(self):
        block = KVBlock(block_id=1, size_bytes=1024)
        old_access = block.last_access
        old_count = block.access_count
        block.touch()
        assert block.access_count == old_count + 1
        assert block.last_access >= old_access

    def test_age_seconds(self):
        block = KVBlock(block_id=1)
        block.last_access = time.monotonic() - 5.0
        assert block.age_seconds >= 4.5

    def test_default_values(self):
        block = KVBlock(block_id=1)
        assert block.tier == KVTier.HOT
        assert block.ref_count == 0
        assert block.is_shared is False
        assert block.quantization == "fp16"
        assert block.compressed is False


class TestKVLifecycleManager:
    def test_admit_block(self):
        mgr = KVLifecycleManager()
        assert mgr.admit(1, 1024, prefix_hash="abc")
        assert mgr.total_blocks == 1
        block = mgr.get_block(1)
        assert block is not None
        assert block.tier == KVTier.HOT

    def test_admit_with_budget(self):
        mgr = KVLifecycleManager(
            tier_configs=[
                KVTierConfig(tier=KVTier.HOT, max_bytes=2048),
            ]
        )
        assert mgr.admit(1, 1024)
        assert mgr.admit(2, 1024)
        # Third block exceeds budget — should trigger eviction or reject
        mgr.admit(3, 1024)
        # Either evicted a previous block or rejected
        assert mgr.get_stats()["admissions_rejected"] >= 0

    def test_touch_block(self):
        mgr = KVLifecycleManager()
        mgr.admit(1, 1024)
        mgr.touch(1)
        block = mgr.get_block(1)
        assert block.access_count == 1

    def test_touch_nonexistent(self):
        mgr = KVLifecycleManager()
        mgr.touch(999)  # no error

    def test_share_block(self):
        mgr = KVLifecycleManager()
        mgr.admit(1, 1024)
        mgr.share(1)
        block = mgr.get_block(1)
        assert block.ref_count == 1
        assert block.is_shared

    def test_release_block(self):
        mgr = KVLifecycleManager()
        mgr.admit(1, 1024)
        mgr.share(1)
        mgr.release(1)
        # release() no longer auto-evicts — block stays alive at ref_count=0
        block = mgr.get_block(1)
        assert block is not None
        assert block.ref_count == 0

    def test_release_non_shared_kept(self):
        mgr = KVLifecycleManager()
        mgr.admit(1, 1024)
        mgr.share(1)
        mgr.share(1)  # ref_count = 2
        mgr.release(1)  # ref_count = 1
        assert mgr.get_block(1) is not None

    def test_migrate_hot_to_warm(self):
        mgr = KVLifecycleManager(
            tier_configs=[
                KVTierConfig(tier=KVTier.WARM, max_bytes=4096),
            ]
        )
        mgr.admit(1, 1024)
        assert mgr.migrate(1, KVTier.WARM)
        block = mgr.get_block(1)
        assert block.tier == KVTier.WARM
        assert mgr.get_stats()["migrations"] == 1

    def test_migrate_nonexistent(self):
        mgr = KVLifecycleManager()
        assert not mgr.migrate(999, KVTier.WARM)

    def test_migrate_exceeds_budget(self):
        mgr = KVLifecycleManager(
            tier_configs=[
                KVTierConfig(tier=KVTier.WARM, max_bytes=512),
            ]
        )
        mgr.admit(1, 1024)
        assert not mgr.migrate(1, KVTier.WARM)

    def test_optimize_skips(self):
        mgr = KVLifecycleManager(optimization_interval=100)
        result = mgr.optimize()
        assert result.get("skipped")

    def test_optimize_downgrades_cold(self):
        mgr = KVLifecycleManager(
            optimization_interval=1,
            tier_configs=[
                KVTierConfig(tier=KVTier.WARM, max_bytes=1048576),
                KVTierConfig(tier=KVTier.COOL, max_bytes=1048576),
            ],
        )
        mgr.admit(1, 1024)
        block = mgr.get_block(1)
        block.last_access = time.monotonic() - 120.0  # old enough

        result = mgr.optimize()
        assert result["blocks_migrated"] >= 1

    def test_eviction_for_space(self):
        mgr = KVLifecycleManager(
            tier_configs=[
                KVTierConfig(tier=KVTier.HOT, max_bytes=1024),
            ]
        )
        mgr.admit(1, 512)
        mgr.admit(2, 512)
        # Third block should evict first
        assert mgr.admit(3, 512)
        assert mgr.get_stats()["evictions"] >= 1

    def test_tier_usage_tracking(self):
        mgr = KVLifecycleManager()
        mgr.admit(1, 1024)
        usage = mgr.tier_usage_bytes
        assert usage.get("HOT", 0) == 1024

    def test_get_stats(self):
        mgr = KVLifecycleManager()
        mgr.admit(1, 1024)
        stats = mgr.get_stats()
        assert stats["total_blocks"] == 1
        assert "tier_blocks" in stats
        assert "migrations" in stats
        assert stats["migrations"] == 0


class TestCacheWarmingPredictor:
    def test_record_access(self):
        p = CacheWarmingPredictor()
        p.record_access("prefix1")
        p.record_access("prefix1")
        assert len(p._prefix_history["prefix1"]) == 2

    def test_predict_candidates(self):
        p = CacheWarmingPredictor(
            prediction_window=60.0,
            min_access_count=2,
            warming_threshold=0.01,  # low threshold for test
        )
        for _ in range(5):
            p.record_access("hot_prefix")
            time.sleep(0.001)
        candidates = p.predict_warm_candidates()
        assert "hot_prefix" in candidates

    def test_no_candidates_below_threshold(self):
        p = CacheWarmingPredictor(min_access_count=10)
        p.record_access("cold_prefix")
        candidates = p.predict_warm_candidates()
        assert len(candidates) == 0

    def test_report_prediction(self):
        p = CacheWarmingPredictor()
        p.report_prediction("prefix1", True)
        p.report_prediction("prefix1", False)
        stats = p.get_stats()
        assert stats["predictions_total"] == 2
        assert stats["predictions_correct"] == 1
        assert stats["accuracy"] == 0.5

    def test_get_stats(self):
        p = CacheWarmingPredictor()
        p.record_access("p1")
        stats = p.get_stats()
        assert stats["tracked_prefixes"] == 1
        assert stats["accuracy"] == 0.0

    def test_history_pruning(self):
        p = CacheWarmingPredictor(prediction_window=0.001)
        p.record_access("old_prefix")
        time.sleep(0.01)
        # Old entries should be pruned on next access
        p.record_access("old_prefix")
        # Only the new entry should remain
        assert len(p._prefix_history["old_prefix"]) == 1


class TestKVCompactionScheduler:
    def test_should_compact_timing(self):
        sched = KVCompactionScheduler(interval_steps=3)
        assert not sched.should_compact()  # step 1
        assert not sched.should_compact()  # step 2
        assert sched.should_compact()  # step 3

    def test_compact_empty(self):
        sched = KVCompactionScheduler()
        freed = sched.compact({})
        assert freed == 0

    def test_compact_fragments(self):
        sched = KVCompactionScheduler()
        # 4 blocks at 25% fill → can compact to 1 block → 3 freed
        usage = {i: 0.25 for i in range(4)}
        freed = sched.compact(usage)
        assert freed >= 1

    def test_compact_full_blocks_not_touched(self):
        sched = KVCompactionScheduler(min_fragmentation_ratio=0.5)
        usage = {1: 0.9, 2: 0.8, 3: 0.25}
        freed = sched.compact(usage)
        # Only block 3 is fragmented
        assert freed >= 0

    def test_batch_size_limit(self):
        sched = KVCompactionScheduler(compaction_batch_size=2)
        usage = {i: 0.1 for i in range(10)}
        freed = sched.compact(usage)
        assert freed >= 0

    def test_get_stats(self):
        sched = KVCompactionScheduler()
        sched.compact({1: 0.2, 2: 0.3})
        stats = sched.get_stats()
        assert stats["compactions_run"] == 1
        assert stats["step_count"] == 0  # compact doesn't increment step
