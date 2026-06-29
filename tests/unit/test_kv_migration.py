"""Unit tests for KV cache migration, multi-tier coordination, and cache warming.

Tests cover:
- KVMigrationManager: tier transitions, temperature tracking, auto-migration
- MultiTierCacheCoordinator: unified lookup, store, promote, evict, rebalance
- CacheWarmingScheduler: predictions, warming operations, verification
"""

import time

from yunshu_engine.kv_migration import (
    BlockTemperature,
    CacheWarmingScheduler,
    KVMigrationManager,
    MigrationRecord,
    MigrationStats,
    MultiTierCacheCoordinator,
    TierStats,
    WarmingStats,
    _prefix_hash,
    _tier_faster,
)
from yunshu_engine.kv_offload import KVTier

# ═══════════════════════════════════════════════════════════════════════════
# KVMigrationManager Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKVMigrationManagerBasic:
    """Basic lifecycle and registration tests."""

    def test_create_defaults(self):
        mgr = KVMigrationManager()
        assert mgr._gpu_capacity == 1024
        assert mgr._cpu_capacity == 4096
        assert mgr._ssd_capacity == 65536

    def test_start_stop(self):
        mgr = KVMigrationManager()
        mgr.start()
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        mgr.stop()
        assert mgr._thread is None

    def test_double_start_idempotent(self):
        mgr = KVMigrationManager()
        mgr.start()
        thread1 = mgr._thread
        mgr.start()  # Should not create a second thread
        assert mgr._thread is thread1
        mgr.stop()

    def test_register_block(self):
        mgr = KVMigrationManager()
        mgr.register_block(1, tier=KVTier.HOT, byte_size=4096)
        assert 1 in mgr._temperatures
        assert mgr._temperatures[1].tier == KVTier.HOT
        assert mgr._temperatures[1].byte_size == 4096
        assert 1 in mgr._tier_blocks[KVTier.HOT]

    def test_register_block_with_data(self):
        mgr = KVMigrationManager()
        data = b"kv_data_block_0"
        mgr.register_block(5, tier=KVTier.HOT, data=data)
        assert 5 in mgr._gpu_store
        assert mgr._gpu_store[5] == bytearray(data)

    def test_unregister_block(self):
        mgr = KVMigrationManager()
        mgr.register_block(10, tier=KVTier.WARM)
        mgr.unregister_block(10)
        assert 10 not in mgr._temperatures
        assert 10 not in mgr._tier_blocks[KVTier.WARM]

    def test_unregister_nonexistent_noop(self):
        mgr = KVMigrationManager()
        mgr.unregister_block(999)  # Should not raise


class TestKVMigrationManagerAccess:
    """Access frequency and temperature tests."""

    def test_set_access_frequency(self):
        mgr = KVMigrationManager()
        mgr.register_block(1)
        mgr.set_access_frequency(1, 50)
        assert mgr._temperatures[1].access_frequency == 50

    def test_record_access_increments(self):
        mgr = KVMigrationManager()
        mgr.register_block(1)
        mgr.record_access(1)
        mgr.record_access(1)
        mgr.record_access(1)
        assert mgr._temperatures[1].access_frequency == 3

    def test_record_access_updates_timestamp(self):
        mgr = KVMigrationManager()
        mgr.register_block(1)
        old_time = mgr._temperatures[1].last_access_time
        time.sleep(0.01)
        mgr.record_access(1)
        assert mgr._temperatures[1].last_access_time > old_time

    def test_set_frequency_nonexistent_noop(self):
        mgr = KVMigrationManager()
        mgr.set_access_frequency(999, 10)  # Should not raise

    def test_frequency_to_tier_hot(self):
        mgr = KVMigrationManager(hot_threshold=10)
        assert mgr._frequency_to_tier(10) == KVTier.HOT
        assert mgr._frequency_to_tier(100) == KVTier.HOT

    def test_frequency_to_tier_warm(self):
        mgr = KVMigrationManager(hot_threshold=10, cold_threshold=2)
        assert mgr._frequency_to_tier(5) == KVTier.WARM
        assert mgr._frequency_to_tier(2) == KVTier.WARM

    def test_frequency_to_tier_ssd(self):
        mgr = KVMigrationManager(cold_threshold=2)
        assert mgr._frequency_to_tier(1) == KVTier.SSD
        assert mgr._frequency_to_tier(0) == KVTier.SSD


class TestKVMigrationManagerMigrations:
    """Tier transition migration tests."""

    def test_migrate_to_cpu(self):
        mgr = KVMigrationManager()
        data = b"gpu_data"
        mgr.register_block(1, tier=KVTier.HOT, data=data)
        results = mgr.migrate_to_cpu([1])
        assert len(results) == 1
        assert results[0].success
        assert results[0].source_tier == KVTier.HOT
        assert results[0].dest_tier == KVTier.WARM
        assert mgr._temperatures[1].tier == KVTier.WARM
        assert 1 not in mgr._gpu_store
        assert 1 in mgr._cpu_store

    def test_migrate_to_ssd(self):
        mgr = KVMigrationManager()
        data = b"cpu_data"
        mgr.register_block(2, tier=KVTier.WARM, data=data)
        results = mgr.migrate_to_ssd([2])
        assert results[0].success
        assert mgr._temperatures[2].tier == KVTier.SSD
        assert 2 not in mgr._cpu_store
        assert 2 in mgr._ssd_store

    def test_migrate_to_gpu_from_cpu(self):
        mgr = KVMigrationManager()
        data = b"warm_data"
        mgr.register_block(3, tier=KVTier.WARM, data=data)
        results = mgr.migrate_to_gpu([3])
        assert results[0].success
        assert results[0].source_tier == KVTier.WARM
        assert mgr._temperatures[3].tier == KVTier.HOT

    def test_migrate_to_gpu_from_ssd(self):
        mgr = KVMigrationManager()
        data = b"ssd_data"
        mgr.register_block(4, tier=KVTier.SSD, data=data)
        results = mgr.migrate_to_gpu([4])
        assert results[0].success
        assert mgr._temperatures[4].tier == KVTier.HOT

    def test_migrate_to_gpu_already_on_gpu(self):
        mgr = KVMigrationManager()
        mgr.register_block(5, tier=KVTier.HOT)
        results = mgr.migrate_to_gpu([5])
        assert results[0].success  # No-op success

    def test_migrate_to_gpu_unknown_block(self):
        mgr = KVMigrationManager()
        results = mgr.migrate_to_gpu([999])
        assert not results[0].success

    def test_migrate_wrong_source_tier_fails(self):
        mgr = KVMigrationManager()
        mgr.register_block(6, tier=KVTier.SSD)
        # Try to migrate from HOT -> WARM, but block is on SSD
        results = mgr.migrate_to_cpu([6])
        assert not results[0].success

    def test_migration_preserves_data(self):
        mgr = KVMigrationManager()
        data = b"preserved_data_12345"
        mgr.register_block(10, tier=KVTier.HOT, data=data)
        mgr.migrate_to_cpu([10])
        assert mgr._cpu_store[10] == bytearray(data)
        mgr.migrate_to_ssd([10])
        assert mgr._ssd_store[10] == bytearray(data)
        mgr.migrate_to_gpu([10])
        assert mgr._gpu_store[10] == bytearray(data)

    def test_migration_updates_migration_count(self):
        mgr = KVMigrationManager()
        mgr.register_block(11, tier=KVTier.HOT)
        mgr.migrate_to_cpu([11])
        assert mgr._temperatures[11].migration_count == 1
        mgr.migrate_to_ssd([11])
        assert mgr._temperatures[11].migration_count == 2

    def test_batch_migration(self):
        mgr = KVMigrationManager()
        for i in range(5):
            mgr.register_block(i, tier=KVTier.HOT, data=f"block_{i}".encode())
        results = mgr.migrate_to_cpu([0, 1, 2, 3, 4])
        assert len(results) == 5
        assert all(r.success for r in results)
        assert all(mgr._temperatures[i].tier == KVTier.WARM for i in range(5))


class TestKVMigrationManagerAutoMigration:
    """Background auto-migration tests."""

    def test_schedule_auto_migration(self):
        mgr = KVMigrationManager(hot_threshold=10, cold_threshold=2)
        # Hot block that should stay on GPU
        mgr.register_block(1, tier=KVTier.HOT)
        mgr.set_access_frequency(1, 15)
        # Cold block that should go to SSD
        mgr.register_block(2, tier=KVTier.HOT)
        mgr.set_access_frequency(2, 1)

        scheduled = mgr.schedule_auto_migration()
        assert scheduled == 1  # Only block 2 needs to move
        assert mgr._temperatures[1].tier == KVTier.HOT
        assert mgr._temperatures[2].tier == KVTier.SSD

    def test_auto_migration_promotes_to_gpu(self):
        mgr = KVMigrationManager(hot_threshold=10)
        mgr.register_block(1, tier=KVTier.WARM)
        mgr.set_access_frequency(1, 20)
        mgr.schedule_auto_migration()
        assert mgr._temperatures[1].tier == KVTier.HOT

    def test_background_thread_processes_queue(self):
        mgr = KVMigrationManager(
            migration_interval=0.1,
            hot_threshold=10,
            cold_threshold=2,
        )
        mgr.start()
        # Register a cold block on GPU
        mgr.register_block(1, tier=KVTier.HOT)
        mgr.set_access_frequency(1, 0)
        # Wait for background thread to process
        time.sleep(0.5)
        mgr.stop()
        # Block should have been migrated to SSD
        assert mgr._temperatures[1].tier == KVTier.SSD


class TestKVMigrationManagerStats:
    """Statistics tests."""

    def test_get_stats_initial(self):
        mgr = KVMigrationManager()
        stats = mgr.get_stats()
        assert stats["total_migrations"] == 0
        assert stats["total_bytes_transferred"] == 0
        assert stats["failed_migrations"] == 0
        assert stats["tracked_blocks"] == 0

    def test_get_stats_after_migration(self):
        mgr = KVMigrationManager(bytes_per_block=4096)
        mgr.register_block(1, tier=KVTier.HOT)
        mgr.migrate_to_cpu([1])
        stats = mgr.get_stats()
        assert stats["gpu_to_cpu_count"] == 1
        assert stats["total_migrations"] == 1
        assert stats["total_bytes_transferred"] == 4096
        assert stats["tracked_blocks"] == 1

    def test_tier_counts(self):
        mgr = KVMigrationManager()
        mgr.register_block(1, tier=KVTier.HOT)
        mgr.register_block(2, tier=KVTier.WARM)
        mgr.register_block(3, tier=KVTier.SSD)
        stats = mgr.get_stats()
        assert stats["tier_counts"]["hot"] == 1
        assert stats["tier_counts"]["warm"] == 1
        assert stats["tier_counts"]["ssd"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# MultiTierCacheCoordinator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiTierCacheCoordinatorBasic:
    """Basic store and lookup tests."""

    def test_create(self):
        coord = MultiTierCacheCoordinator()
        assert coord._gpu_capacity == 1024
        assert coord._cpu_capacity == 4096

    def test_store_returns_tier(self):
        coord = MultiTierCacheCoordinator()
        tier = coord.store("prefix_a", b"data_a", predicted_access=15)
        assert tier == KVTier.HOT

    def test_store_medium_access(self):
        coord = MultiTierCacheCoordinator()
        tier = coord.store("prefix_b", b"data_b", predicted_access=5)
        assert tier == KVTier.WARM

    def test_store_low_access(self):
        coord = MultiTierCacheCoordinator()
        tier = coord.store("prefix_c", b"data_c", predicted_access=0)
        assert tier == KVTier.SSD

    def test_lookup_hit_gpu(self):
        coord = MultiTierCacheCoordinator()
        coord.store("test_prefix", b"gpu_data", predicted_access=15)
        result = coord.lookup("test_prefix")
        assert result == b"gpu_data"

    def test_lookup_miss(self):
        coord = MultiTierCacheCoordinator()
        result = coord.lookup("nonexistent")
        assert result is None

    def test_lookup_stable_hash(self):
        coord = MultiTierCacheCoordinator()
        coord.store("abc", b"data", predicted_access=15)
        key = _prefix_hash("abc")
        assert key in coord._gpu_cache

    def test_lookup_bytes_prefix(self):
        coord = MultiTierCacheCoordinator()
        coord.store(b"\x01\x02\x03", b"data")
        result = coord.lookup(b"\x01\x02\x03")
        assert result == b"data"


class TestMultiTierCacheCoordinatorPromotion:
    """Promotion and tier movement tests."""

    def test_promote_from_cpu_to_gpu(self):
        coord = MultiTierCacheCoordinator(promote_on_hit=False)
        coord.store("warm_prefix", b"warm_data", predicted_access=5)
        # Verify it's on CPU
        key = _prefix_hash("warm_prefix")
        assert key in coord._cpu_cache
        # Promote to GPU
        result = coord.promote("warm_prefix")
        assert result is True
        assert key in coord._gpu_cache
        assert key not in coord._cpu_cache

    def test_promote_from_ssd_to_gpu(self):
        coord = MultiTierCacheCoordinator(promote_on_hit=False)
        coord.store("cold_prefix", b"cold_data", predicted_access=0)
        key = _prefix_hash("cold_prefix")
        assert key in coord._ssd_cache
        result = coord.promote("cold_prefix")
        assert result is True
        assert key in coord._gpu_cache

    def test_promote_already_on_gpu(self):
        coord = MultiTierCacheCoordinator()
        coord.store("hot_prefix", b"hot_data", predicted_access=15)
        result = coord.promote("hot_prefix")
        assert result is True

    def test_promote_nonexistent(self):
        coord = MultiTierCacheCoordinator()
        result = coord.promote("ghost")
        assert result is False

    def test_auto_promote_on_lookup(self):
        coord = MultiTierCacheCoordinator(promote_on_hit=True)
        coord.store("auto_promote", b"auto_data", predicted_access=0)
        key = _prefix_hash("auto_promote")
        # Initially on SSD
        assert key in coord._ssd_cache
        # Lookup should auto-promote to GPU
        result = coord.lookup("auto_promote")
        assert result == b"auto_data"
        assert key in coord._gpu_cache


class TestMultiTierCacheCoordinatorEviction:
    """Eviction tests."""

    def test_evict_from_all_tiers(self):
        coord = MultiTierCacheCoordinator()
        coord.store("evict_me", b"data", predicted_access=0)
        key = _prefix_hash("evict_me")
        assert key in coord._ssd_cache
        result = coord.evict("evict_me")
        assert result is True
        assert key not in coord._ssd_cache

    def test_evict_nonexistent(self):
        coord = MultiTierCacheCoordinator()
        result = coord.evict("ghost")
        assert result is False

    def test_evict_cascades_from_all(self):
        coord = MultiTierCacheCoordinator()
        coord.store("multi", b"data", predicted_access=15)
        key = _prefix_hash("multi")
        # Manually copy to all tiers
        data = coord._gpu_cache[key]
        coord._cpu_cache[key] = data
        coord._ssd_cache[key] = data
        coord.evict("multi")
        assert key not in coord._gpu_cache
        assert key not in coord._cpu_cache
        assert key not in coord._ssd_cache


class TestMultiTierCacheCoordinatorCapacity:
    """Capacity and LRU eviction tests."""

    def test_gpu_capacity_eviction(self):
        coord = MultiTierCacheCoordinator(gpu_capacity=2)
        coord.store("a", b"data_a", predicted_access=15)
        coord.store("b", b"data_b", predicted_access=15)
        coord.store("c", b"data_c", predicted_access=15)
        # "a" should have been evicted from GPU to CPU
        key_a = _prefix_hash("a")
        assert key_a not in coord._gpu_cache
        assert key_a in coord._cpu_cache

    def test_store_overwrites_existing(self):
        coord = MultiTierCacheCoordinator()
        coord.store("dup", b"old_data", predicted_access=15)
        coord.store("dup", b"new_data", predicted_access=15)
        result = coord.lookup("dup")
        assert result == b"new_data"


class TestMultiTierCacheCoordinatorRebalance:
    """Rebalancing tests."""

    def test_rebalance_promotes_high_access(self):
        coord = MultiTierCacheCoordinator(promote_on_hit=False)
        coord.store("hot_on_cpu", b"data", predicted_access=5)
        key = _prefix_hash("hot_on_cpu")
        # Simulate high access on CPU tier
        data, ts, count = coord._cpu_cache[key]
        coord._cpu_cache[key] = (data, ts, 15)
        moved = coord.rebalance()
        assert moved >= 1
        assert key in coord._gpu_cache

    def test_rebalance_demotes_low_access(self):
        coord = MultiTierCacheCoordinator(promote_on_hit=False)
        coord.store("cold_on_gpu", b"data", predicted_access=15)
        key = _prefix_hash("cold_on_gpu")
        # Simulate low access on GPU tier (access_count < 2 triggers demote)
        data, ts, count = coord._gpu_cache[key]
        coord._gpu_cache[key] = (data, ts, 1)
        moved = coord.rebalance()
        assert moved >= 1
        # Block was demoted from GPU -> CPU, then CPU -> SSD (both have access < 2)
        assert key not in coord._gpu_cache

    def test_rebalance_no_moves_needed(self):
        coord = MultiTierCacheCoordinator(promote_on_hit=False)
        # All blocks already in correct tiers with matching access counts
        coord.store("good", b"data", predicted_access=15)
        key = _prefix_hash("good")
        # Set access count high so it's not demoted
        data, ts, count = coord._gpu_cache[key]
        coord._gpu_cache[key] = (data, ts, 15)
        moved = coord.rebalance()
        assert moved == 0


class TestMultiTierCacheCoordinatorStats:
    """Statistics tests."""

    def test_tier_stats_initial(self):
        coord = MultiTierCacheCoordinator()
        stats = coord.get_tier_stats()
        assert stats["hot"]["entry_count"] == 0
        assert stats["warm"]["entry_count"] == 0
        assert stats["ssd"]["entry_count"] == 0

    def test_tier_stats_after_operations(self):
        coord = MultiTierCacheCoordinator()
        coord.store("a", b"data_a", predicted_access=15)
        coord.store("b", b"data_b", predicted_access=5)
        stats = coord.get_tier_stats()
        assert stats["hot"]["entry_count"] == 1
        assert stats["warm"]["entry_count"] == 1

    def test_tier_stats_hit_rate(self):
        coord = MultiTierCacheCoordinator()
        coord.store("hit_test", b"data", predicted_access=15)
        coord.lookup("hit_test")
        coord.lookup("miss_test")
        stats = coord.get_tier_stats()
        assert stats["hot"]["hit_count"] == 1
        assert stats["hot"]["miss_count"] == 1
        assert stats["hot"]["hit_rate"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# CacheWarmingScheduler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheWarmingSchedulerBasic:
    """Basic warming scheduler tests."""

    def test_create(self):
        sched = CacheWarmingScheduler()
        assert sched._window_size == 100
        assert sched._prediction_threshold == 2

    def test_start_stop(self):
        sched = CacheWarmingScheduler()
        sched.start()
        assert sched._thread is not None
        assert sched._thread.is_alive()
        sched.stop()
        assert sched._thread is None

    def test_record_request(self):
        sched = CacheWarmingScheduler()
        sched.record_request("prefix_a")
        sched.record_request("prefix_a")
        sched.record_request("prefix_b")
        assert len(sched._access_window) == 3


class TestCacheWarmingSchedulerPrediction:
    """Prediction tests."""

    def test_predict_hot_prefixes_empty(self):
        sched = CacheWarmingScheduler()
        preds = sched.predict_hot_prefixes()
        assert preds == []

    def test_predict_hot_prefixes_below_threshold(self):
        sched = CacheWarmingScheduler(prediction_threshold=3)
        sched.record_request("low_freq")
        sched.record_request("low_freq")
        preds = sched.predict_hot_prefixes()
        assert len(preds) == 0  # Below threshold

    def test_predict_hot_prefixes_above_threshold(self):
        sched = CacheWarmingScheduler(prediction_threshold=2)
        for _ in range(5):
            sched.record_request("hot_prefix")
        preds = sched.predict_hot_prefixes()
        assert len(preds) == 1
        assert preds[0].prefix_hash == _prefix_hash("hot_prefix")
        assert preds[0].confidence > 0.0

    def test_predictions_sorted_by_confidence(self):
        sched = CacheWarmingScheduler(prediction_threshold=2, window_size=20)
        # Add higher frequency for prefix_a
        for _ in range(8):
            sched.record_request("prefix_a")
        for _ in range(3):
            sched.record_request("prefix_b")
        preds = sched.predict_hot_prefixes()
        assert len(preds) == 2
        assert preds[0].confidence > preds[1].confidence

    def test_prediction_accuracy_stats(self):
        sched = CacheWarmingScheduler(prediction_threshold=2)
        for _ in range(3):
            sched.record_request("pred")
        preds = sched.predict_hot_prefixes()
        assert len(preds) == 1
        stats = sched.get_stats()
        assert stats["predictions_made"] == 1


class TestCacheWarmingSchedulerWarming:
    """Warming operation tests."""

    def test_warm_cache_with_coordinator(self):
        coord = MultiTierCacheCoordinator()
        coord.store("warm_target", b"ssd_data", predicted_access=0)
        sched = CacheWarmingScheduler(coordinator=coord)
        key = _prefix_hash("warm_target")
        assert key in coord._ssd_cache
        result = sched.warm_cache("warm_target")
        assert result is True
        assert key in coord._gpu_cache

    def test_warm_cache_without_coordinator(self):
        sched = CacheWarmingScheduler()
        result = sched.warm_cache("anything")
        assert result is False

    def test_schedule_warming(self):
        coord = MultiTierCacheCoordinator()
        coord.store("prefix_1", b"data1", predicted_access=0)
        coord.store("prefix_2", b"data2", predicted_access=0)
        sched = CacheWarmingScheduler(
            coordinator=coord, prediction_threshold=2, max_warming_ops=10
        )
        for _ in range(3):
            sched.record_request("prefix_1")
            sched.record_request("prefix_2")
        preds = sched.predict_hot_prefixes()
        warmed = sched.schedule_warming(preds)
        assert warmed == 2

    def test_schedule_warming_respects_limit(self):
        coord = MultiTierCacheCoordinator()
        for i in range(10):
            coord.store(f"p_{i}", f"data_{i}".encode(), predicted_access=0)
        sched = CacheWarmingScheduler(
            coordinator=coord, prediction_threshold=2, max_warming_ops=3
        )
        for _ in range(3):
            for i in range(10):
                sched.record_request(f"p_{i}")
        preds = sched.predict_hot_prefixes()
        assert len(preds) >= 3
        warmed = sched.schedule_warming(preds)
        assert warmed <= 3

    def test_warm_cache_already_hot_noop(self):
        coord = MultiTierCacheCoordinator()
        coord.store("hot_already", b"data", predicted_access=15)
        sched = CacheWarmingScheduler(coordinator=coord)
        _prefix_hash("hot_already")
        # Already on GPU, warming returns True (no-op)
        result = sched.warm_cache("hot_already")
        assert result is True


class TestCacheWarmingSchedulerVerification:
    """Prediction verification tests."""

    def test_verify_correct_prediction(self):
        sched = CacheWarmingScheduler(prediction_threshold=2)
        for _ in range(3):
            sched.record_request("verified_prefix")
        preds = sched.predict_hot_prefixes()
        assert len(preds) == 1
        key = preds[0].prefix_hash
        sched.verify_prediction(key, was_hit=True)
        stats = sched.get_stats()
        assert stats["predictions_correct"] == 1
        assert stats["predictions_incorrect"] == 0

    def test_verify_incorrect_prediction(self):
        sched = CacheWarmingScheduler(prediction_threshold=2)
        for _ in range(3):
            sched.record_request("bad_pred")
        preds = sched.predict_hot_prefixes()
        key = preds[0].prefix_hash
        sched.verify_prediction(key, was_hit=False)
        stats = sched.get_stats()
        assert stats["predictions_correct"] == 0
        assert stats["predictions_incorrect"] == 1

    def test_verify_non_predicted_noop(self):
        sched = CacheWarmingScheduler()
        sched.verify_prediction("unknown_prefix", was_hit=True)
        stats = sched.get_stats()
        assert stats["predictions_correct"] == 0

    def test_prediction_accuracy_computed(self):
        sched = CacheWarmingScheduler(prediction_threshold=2)
        for _ in range(3):
            sched.record_request("a")
        preds = sched.predict_hot_prefixes()
        key_a = preds[0].prefix_hash
        sched.verify_prediction(key_a, was_hit=True)
        sched.verify_prediction(_prefix_hash("unknown"), was_hit=False)
        stats = sched.get_stats()
        assert stats["prediction_accuracy"] == 1.0


class TestCacheWarmingSchedulerStats:
    """Statistics tests."""

    def test_get_stats_initial(self):
        sched = CacheWarmingScheduler()
        stats = sched.get_stats()
        assert stats["predictions_made"] == 0
        assert stats["warming_operations"] == 0
        assert stats["prediction_accuracy"] == 0.0

    def test_get_stats_after_warming(self):
        coord = MultiTierCacheCoordinator()
        coord.store("stat_test", b"data", predicted_access=0)
        sched = CacheWarmingScheduler(coordinator=coord, prediction_threshold=2)
        for _ in range(3):
            sched.record_request("stat_test")
        preds = sched.predict_hot_prefixes()
        sched.schedule_warming(preds)
        stats = sched.get_stats()
        assert stats["predictions_made"] >= 1
        assert stats["warming_operations"] >= 1

    def test_window_size_in_stats(self):
        sched = CacheWarmingScheduler(window_size=50)
        stats = sched.get_stats()
        assert stats["window_size"] == 50


class TestCacheWarmingSchedulerBackground:
    """Background thread tests."""

    def test_background_warming_loop(self):
        coord = MultiTierCacheCoordinator()
        coord.store("bg_warm", b"data", predicted_access=0)
        sched = CacheWarmingScheduler(
            coordinator=coord,
            warming_interval=0.1,
            prediction_threshold=2,
        )
        sched.start()
        # Record enough accesses
        for _ in range(5):
            sched.record_request("bg_warm")
        time.sleep(0.5)
        sched.stop()
        # Data should have been promoted
        key = _prefix_hash("bg_warm")
        assert key in coord._gpu_cache


# ═══════════════════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Test utility functions."""

    def test_prefix_hash_string(self):
        h = _prefix_hash("hello")
        assert isinstance(h, str)
        assert len(h) == 32  # blake2b digest_size=16 -> 32 hex chars

    def test_prefix_hash_bytes(self):
        h = _prefix_hash(b"\x00\x01")
        assert isinstance(h, str)
        assert len(h) == 32

    def test_prefix_hash_deterministic(self):
        h1 = _prefix_hash("test")
        h2 = _prefix_hash("test")
        assert h1 == h2

    def test_prefix_hash_different_inputs(self):
        h1 = _prefix_hash("a")
        h2 = _prefix_hash("b")
        assert h1 != h2

    def test_tier_faster_hot_vs_warm(self):
        assert _tier_faster(KVTier.HOT, KVTier.WARM) == KVTier.HOT

    def test_tier_faster_ssd_vs_cold(self):
        assert _tier_faster(KVTier.SSD, KVTier.COLD) == KVTier.SSD

    def test_tier_faster_same(self):
        assert _tier_faster(KVTier.HOT, KVTier.HOT) == KVTier.HOT


# ═══════════════════════════════════════════════════════════════════════════
# Data class tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDataClasses:
    """Test data class properties."""

    def test_migration_stats_total(self):
        s = MigrationStats(gpu_to_cpu_count=3, cpu_to_ssd_count=2)
        assert s.total_migrations == 5

    def test_migration_stats_avg_time(self):
        s = MigrationStats(total_migration_time=1.0, gpu_to_cpu_count=5)
        assert abs(s.avg_migration_time - 0.2) < 0.001

    def test_migration_stats_avg_time_zero(self):
        s = MigrationStats()
        assert s.avg_migration_time == 0.0

    def test_tier_stats_hit_rate(self):
        s = TierStats(tier=KVTier.HOT, hit_count=8, miss_count=2)
        assert s.hit_rate == 0.8

    def test_tier_stats_hit_rate_zero(self):
        s = TierStats(tier=KVTier.HOT)
        assert s.hit_rate == 0.0

    def test_tier_stats_utilization(self):
        s = TierStats(tier=KVTier.HOT, entry_count=50, max_capacity=100)
        assert s.utilization_pct == 50.0

    def test_warming_stats_accuracy(self):
        s = WarmingStats(predictions_correct=7, predictions_incorrect=3)
        assert abs(s.prediction_accuracy - 0.7) < 0.001

    def test_block_temperature_defaults(self):
        bt = BlockTemperature(block_id=0)
        assert bt.tier == KVTier.HOT
        assert bt.access_frequency == 0
        assert bt.migration_count == 0

    def test_migration_record(self):
        r = MigrationRecord(block_id=1, source_tier=KVTier.HOT, dest_tier=KVTier.WARM)
        assert r.success is True
        assert r.bytes_transferred == 0
