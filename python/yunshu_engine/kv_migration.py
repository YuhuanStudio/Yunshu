from __future__ import annotations
"""Yunshu KV Cache Migration & Multi-Tier Cache Coordination.

Handles asynchronous migration of KV cache blocks between tiers
(GPU hot -> CPU warm -> SSD cold), unified tier coordination,
and proactive cache warming based on request pattern prediction.

Three main components:

- **KVMigrationManager**: Background migration thread that moves KV blocks
  between tiers based on access frequency / temperature. Integrates with
  the existing kv_offload.py infrastructure (KVTier, KVOffloadConfig, etc.).

- **MultiTierCacheCoordinator**: Unified lookup/store/promote/evict interface
  that coordinates GPU, CPU, and SSD cache tiers. Provides a single search
  path across all tiers and auto-rebalancing based on access patterns.

- **CacheWarmingScheduler**: Proactive cache warmer that analyses recent
  request patterns to predict which KV prefixes will be needed next and
  pre-loads them from SSD into GPU before they are requested.

Design follows existing patterns:
- KVTier enum from kv_offload.py (HOT/WARM/SSD/COLD)
- Block hashing from kv_prefix_cache.py (_compute_block_hashes)
- SSD storage from ssd_kv_cache.py (SSDKVCache)
- TieredKVCacheManager from yunshu_kv/tiered.py

Thread safety:
- KVMigrationManager uses a background threading.Thread for async migration.
- MultiTierCacheCoordinator uses threading.Lock for state mutations.
- CacheWarmingScheduler uses a background daemon thread for prediction/warming.
"""

import collections
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .kv_offload import KVTier

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────


@dataclass
class BlockTemperature:
    """Temperature metadata for a KV block in the migration system."""
    block_id: int
    tier: KVTier = KVTier.HOT
    access_frequency: int = 0
    last_access_time: float = 0.0
    byte_size: int = 2048  # default estimate
    migration_count: int = 0
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class MigrationRecord:
    """Record of a single migration operation."""
    block_id: int
    source_tier: KVTier
    dest_tier: KVTier
    bytes_transferred: int = 0
    latency_seconds: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    success: bool = True


@dataclass
class MigrationStats:
    """Aggregate statistics for the migration manager."""
    gpu_to_cpu_count: int = 0
    cpu_to_ssd_count: int = 0
    ssd_to_cpu_count: int = 0
    cpu_to_gpu_count: int = 0
    gpu_to_ssd_count: int = 0
    ssd_to_gpu_count: int = 0
    total_bytes_transferred: int = 0
    total_migration_time: float = 0.0
    failed_migrations: int = 0
    active_migrations: int = 0

    @property
    def total_migrations(self) -> int:
        return (
            self.gpu_to_cpu_count
            + self.cpu_to_ssd_count
            + self.ssd_to_cpu_count
            + self.cpu_to_gpu_count
            + self.gpu_to_ssd_count
            + self.ssd_to_gpu_count
        )

    @property
    def avg_migration_time(self) -> float:
        n = self.total_migrations
        return self.total_migration_time / n if n > 0 else 0.0


@dataclass
class TierStats:
    """Per-tier utilization statistics."""
    tier: KVTier
    entry_count: int = 0
    byte_size: int = 0
    max_capacity: int = 0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def utilization_pct(self) -> float:
        return (
            (self.entry_count / self.max_capacity * 100)
            if self.max_capacity > 0
            else 0.0
        )


@dataclass
class WarmingPrediction:
    """A prediction about a prefix that will be needed soon."""
    prefix_hash: str
    confidence: float  # 0.0-1.0
    predicted_access_time: float  # monotonic timestamp
    source_tier: KVTier
    access_count_in_window: int = 1


@dataclass
class WarmingStats:
    """Statistics for the cache warming scheduler."""
    predictions_made: int = 0
    predictions_correct: int = 0
    predictions_incorrect: int = 0
    warming_operations: int = 0
    warming_bytes_transferred: int = 0
    hit_rate_before: float = 0.0
    hit_rate_after: float = 0.0

    @property
    def prediction_accuracy(self) -> float:
        total = self.predictions_correct + self.predictions_incorrect
        return self.predictions_correct / total if total > 0 else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────


def _prefix_hash(token_prefix: str | bytes) -> str:
    """Hash a token prefix string/bytes to a stable key."""
    if isinstance(token_prefix, str):
        token_prefix = token_prefix.encode("utf-8")
    return hashlib.blake2b(token_prefix, digest_size=16).hexdigest()


def _tier_faster(t1: KVTier, t2: KVTier) -> KVTier:
    """Return the faster of two tiers (HOT > WARM > SSD > COLD)."""
    order = {KVTier.HOT: 0, KVTier.WARM: 1, KVTier.SSD: 2, KVTier.COLD: 3}
    return t1 if order.get(t1, 99) < order.get(t2, 99) else t2


# ── KVMigrationManager ────────────────────────────────────────────────────


class KVMigrationManager:
    """Handles KV cache migration between tiers (GPU -> CPU -> SSD).

    Runs a background thread that moves KV blocks asynchronously based
    on block temperature (access frequency). Hot blocks stay on GPU,
    warm blocks on CPU, cold blocks on SSD.

    Uses the existing kv_offload.py KVTier enum and infrastructure for
    the actual tier transfers, adding:
    - Priority-based migration (access frequency -> temperature)
    - Background migration thread with queue
    - Per-block temperature tracking

    Usage::

        mgr = KVMigrationManager()
        mgr.start()

        # Register blocks
        mgr.register_block(42, tier=KVTier.HOT, byte_size=4096)

        # Set access frequency (higher = hotter)
        mgr.set_access_frequency(42, frequency=100)

        # Migrate specific blocks
        mgr.migrate_to_cpu([42, 43])

        # Background thread auto-migrates based on temperature
        stats = mgr.get_stats()

        mgr.stop()
    """

    def __init__(
        self,
        gpu_capacity: int = 1024,
        cpu_capacity: int = 4096,
        ssd_capacity: int = 65536,
        migration_interval: float = 5.0,
        bytes_per_block: int = 2048,
        hot_threshold: int = 10,
        cold_threshold: int = 2,
    ) -> None:
        self._gpu_capacity = gpu_capacity
        self._cpu_capacity = cpu_capacity
        self._ssd_capacity = ssd_capacity
        self._migration_interval = migration_interval
        self._bytes_per_block = bytes_per_block
        self._hot_threshold = hot_threshold
        self._cold_threshold = cold_threshold

        # Block temperature tracking: block_id -> BlockTemperature
        self._temperatures: dict[int, BlockTemperature] = {}
        self._lock = threading.Lock()

        # Per-tier block sets: tier -> set of block_ids
        self._tier_blocks: dict[KVTier, set[int]] = {
            KVTier.HOT: set(),
            KVTier.WARM: set(),
            KVTier.SSD: set(),
            KVTier.COLD: set(),
        }

        # Migration queue and background thread
        self._migration_queue: list[tuple[int, KVTier, KVTier]] = []
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Simulated tier storage (in production, these are real caches)
        # block_id -> data (bytes or bytearray)
        self._gpu_store: dict[int, bytearray] = {}
        self._cpu_store: dict[int, bytearray] = {}
        self._ssd_store: dict[int, bytearray] = {}

        # Statistics
        self._stats = MigrationStats()
        self._migration_history: list[MigrationRecord] = []
        self._max_history = 1000

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background migration thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._migration_loop,
            name="kv-migration",
            daemon=True,
        )
        self._thread.start()
        logger.info("KV migration manager started")

    def stop(self) -> None:
        """Stop the background migration thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("KV migration manager stopped")

    # ── Block Registration ──────────────────────────────────────────

    def register_block(
        self,
        block_id: int,
        tier: KVTier = KVTier.HOT,
        byte_size: int = 0,
        data: bytes | None = None,
    ) -> None:
        """Register a block with the migration manager.

        Args:
            block_id: Unique block identifier.
            tier: Current tier of the block.
            byte_size: Size of the block in bytes.
            data: Optional data payload to store in the tier.
        """
        with self._lock:
            size = byte_size or self._bytes_per_block
            temp = BlockTemperature(
                block_id=block_id,
                tier=tier,
                byte_size=size,
                last_access_time=time.monotonic(),
            )
            self._temperatures[block_id] = temp
            self._tier_blocks[tier].add(block_id)

            # Store data in appropriate tier
            if data is not None:
                self._store_in_tier(block_id, data, tier)

    def unregister_block(self, block_id: int) -> None:
        """Remove a block from tracking."""
        with self._lock:
            temp = self._temperatures.pop(block_id, None)
            if temp is not None:
                self._tier_blocks[temp.tier].discard(block_id)
            # Remove from all stores
            self._gpu_store.pop(block_id, None)
            self._cpu_store.pop(block_id, None)
            self._ssd_store.pop(block_id, None)

    # ── Access Tracking ─────────────────────────────────────────────

    def set_access_frequency(self, block_id: int, frequency: int) -> None:
        """Update block temperature by setting access frequency.

        Higher frequency = hotter block = stays on faster tier.

        Args:
            block_id: The block to update.
            frequency: New access frequency count.
        """
        with self._lock:
            temp = self._temperatures.get(block_id)
            if temp is not None:
                temp.access_frequency = frequency
                temp.last_access_time = time.monotonic()

    def record_access(self, block_id: int) -> None:
        """Record a single access to a block (increments frequency)."""
        with self._lock:
            temp = self._temperatures.get(block_id)
            if temp is not None:
                temp.access_frequency += 1
                temp.last_access_time = time.monotonic()

    # ── Migration API ───────────────────────────────────────────────

    def migrate_to_cpu(self, block_ids: list[int]) -> list[MigrationRecord]:
        """Move blocks from GPU to CPU memory.

        Args:
            block_ids: Blocks to migrate.

        Returns:
            List of MigrationRecords for each block.
        """
        return self._migrate_blocks(block_ids, KVTier.HOT, KVTier.WARM)

    def migrate_to_ssd(self, block_ids: list[int]) -> list[MigrationRecord]:
        """Move blocks from CPU to SSD storage.

        Args:
            block_ids: Blocks to migrate.

        Returns:
            List of MigrationRecords for each block.
        """
        return self._migrate_blocks(block_ids, KVTier.WARM, KVTier.SSD)

    def migrate_to_gpu(self, block_ids: list[int]) -> list[MigrationRecord]:
        """Promote blocks from slower tier back to GPU.

        Checks CPU first, then SSD, then returns failure record.

        Args:
            block_ids: Blocks to promote to GPU.

        Returns:
            List of MigrationRecords for each block.
        """
        results = []
        for block_id in block_ids:
            with self._lock:
                temp = self._temperatures.get(block_id)
                if temp is None:
                    results.append(MigrationRecord(
                        block_id=block_id,
                        source_tier=KVTier.COLD,
                        dest_tier=KVTier.HOT,
                        success=False,
                    ))
                    self._stats.failed_migrations += 1
                    continue

                source = temp.tier
                if source == KVTier.HOT:
                    # Already on GPU
                    results.append(MigrationRecord(
                        block_id=block_id,
                        source_tier=source,
                        dest_tier=KVTier.HOT,
                        success=True,
                    ))
                    continue

                record = self._do_migrate(block_id, source, KVTier.HOT)
                results.append(record)

        return results

    # ── Background Auto-Migration ───────────────────────────────────

    def schedule_auto_migration(self) -> int:
        """Analyze block temperatures and schedule auto-migrations.

        Hot blocks (freq >= hot_threshold) -> promoted to GPU.
        Warm blocks (cold_threshold <= freq < hot_threshold) -> kept on CPU.
        Cold blocks (freq < cold_threshold) -> demoted to SSD.

        Returns:
            Number of migrations scheduled.
        """
        scheduled = 0
        with self._lock:
            for block_id, temp in list(self._temperatures.items()):
                target_tier = self._frequency_to_tier(temp.access_frequency)
                if target_tier != temp.tier:
                    with self._queue_lock:
                        self._migration_queue.append(
                            (block_id, temp.tier, target_tier)
                        )
                    scheduled += 1

        if scheduled > 0:
            logger.debug("Auto-migration: scheduled %d block moves", scheduled)
            self._drain_queue()
        return scheduled

    def _migration_loop(self) -> None:
        """Background thread: periodically run auto-migration."""
        while not self._stop_event.wait(timeout=self._migration_interval):
            try:
                self.schedule_auto_migration()
                self._drain_queue()
            except Exception:
                logger.debug("Migration loop error", exc_info=True)

    def _drain_queue(self) -> int:
        """Process all pending migrations in the queue."""
        processed = 0
        while True:
            with self._lock:
                with self._queue_lock:
                    if not self._migration_queue:
                        break
                    block_id, source, dest = self._migration_queue.pop(0)
                self._do_migrate(block_id, source, dest)
            processed += 1
        return processed

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return migration statistics."""
        with self._lock:
            return {
                "gpu_to_cpu_count": self._stats.gpu_to_cpu_count,
                "cpu_to_ssd_count": self._stats.cpu_to_ssd_count,
                "ssd_to_cpu_count": self._stats.ssd_to_cpu_count,
                "cpu_to_gpu_count": self._stats.cpu_to_gpu_count,
                "gpu_to_ssd_count": self._stats.gpu_to_ssd_count,
                "ssd_to_gpu_count": self._stats.ssd_to_gpu_count,
                "total_migrations": self._stats.total_migrations,
                "total_bytes_transferred": self._stats.total_bytes_transferred,
                "total_migration_time_s": round(self._stats.total_migration_time, 4),
                "avg_migration_time_s": round(self._stats.avg_migration_time, 6),
                "failed_migrations": self._stats.failed_migrations,
                "active_migrations": self._stats.active_migrations,
                "pending_queue_size": len(self._migration_queue),
                "tracked_blocks": len(self._temperatures),
                "tier_counts": {
                    tier.value: len(blocks)
                    for tier, blocks in self._tier_blocks.items()
                },
            }

    # ── Internal ────────────────────────────────────────────────────

    def _migrate_blocks(
        self,
        block_ids: list[int],
        source_tier: KVTier,
        dest_tier: KVTier,
    ) -> list[MigrationRecord]:
        """Migrate a batch of blocks from source to destination tier."""
        results = []
        for block_id in block_ids:
            with self._lock:
                temp = self._temperatures.get(block_id)
                if temp is None:
                    results.append(MigrationRecord(
                        block_id=block_id,
                        source_tier=source_tier,
                        dest_tier=dest_tier,
                        success=False,
                    ))
                    self._stats.failed_migrations += 1
                    continue

                actual_source = temp.tier
                if actual_source != source_tier:
                    # Block not in expected source tier
                    results.append(MigrationRecord(
                        block_id=block_id,
                        source_tier=actual_source,
                        dest_tier=dest_tier,
                        success=False,
                    ))
                    self._stats.failed_migrations += 1
                    continue

                record = self._do_migrate(block_id, source_tier, dest_tier)
                results.append(record)
        return results

    def _do_migrate(
        self,
        block_id: int,
        source: KVTier,
        dest: KVTier,
    ) -> MigrationRecord:
        """Execute a single block migration. Must be called with self._lock held."""
        start = time.monotonic()
        temp = self._temperatures.get(block_id)

        if temp is None:
            self._stats.failed_migrations += 1
            return MigrationRecord(
                block_id=block_id,
                source_tier=source,
                dest_tier=dest,
                success=False,
            )

        # Transfer data between simulated stores
        data = self._pop_from_tier(block_id, source)
        if data is not None:
            self._store_in_tier(block_id, data, dest)
        else:
            # No data to transfer, still update tier tracking
            pass

        # Update tier tracking
        self._tier_blocks[source].discard(block_id)
        self._tier_blocks[dest].add(block_id)
        temp.tier = dest
        temp.migration_count += 1

        latency = time.monotonic() - start
        self._stats.total_migration_time += latency
        self._stats.total_bytes_transferred += temp.byte_size

        # Update directional counter
        counter_key = f"{source.value}_to_{dest.value}_count"
        if hasattr(self._stats, counter_key):
            setattr(self._stats, counter_key, getattr(self._stats, counter_key) + 1)
        # Map tier names to stat fields
        direction_map = {
            (KVTier.HOT, KVTier.WARM): "gpu_to_cpu_count",
            (KVTier.WARM, KVTier.SSD): "cpu_to_ssd_count",
            (KVTier.SSD, KVTier.WARM): "ssd_to_cpu_count",
            (KVTier.WARM, KVTier.HOT): "cpu_to_gpu_count",
            (KVTier.HOT, KVTier.SSD): "gpu_to_ssd_count",
            (KVTier.SSD, KVTier.HOT): "ssd_to_gpu_count",
        }
        key = direction_map.get((source, dest))
        if key is not None:
            setattr(self._stats, key, getattr(self._stats, key) + 1)

        record = MigrationRecord(
            block_id=block_id,
            source_tier=source,
            dest_tier=dest,
            bytes_transferred=temp.byte_size,
            latency_seconds=latency,
            success=True,
        )

        # Record in history
        self._migration_history.append(record)
        if len(self._migration_history) > self._max_history:
            self._migration_history = self._migration_history[-self._max_history // 2:]

        return record

    def _store_in_tier(self, block_id: int, data: bytes, tier: KVTier) -> None:
        """Store data in the specified tier's simulated store."""
        store_map = {
            KVTier.HOT: self._gpu_store,
            KVTier.WARM: self._cpu_store,
            KVTier.SSD: self._ssd_store,
        }
        store = store_map.get(tier)
        if store is not None:
            store[block_id] = bytearray(data)

    def _pop_from_tier(self, block_id: int, tier: KVTier) -> bytearray | None:
        """Pop data from a tier's simulated store."""
        store_map = {
            KVTier.HOT: self._gpu_store,
            KVTier.WARM: self._cpu_store,
            KVTier.SSD: self._ssd_store,
        }
        store = store_map.get(tier)
        if store is not None:
            return store.pop(block_id, None)
        return None

    def _frequency_to_tier(self, frequency: int) -> KVTier:
        """Map access frequency to target tier."""
        if frequency >= self._hot_threshold:
            return KVTier.HOT
        elif frequency >= self._cold_threshold:
            return KVTier.WARM
        else:
            return KVTier.SSD


# ── MultiTierCacheCoordinator ─────────────────────────────────────────────


class MultiTierCacheCoordinator:
    """Coordinates between GPU, CPU, and SSD KV cache tiers.

    Provides a unified view of all cache tiers with a single lookup path
    that searches from fastest (GPU) to slowest (SSD). Manages block
    placement, promotion, and eviction across all tiers.

    Auto-rebalancing moves blocks between tiers based on access patterns
    to maintain optimal distribution.

    Usage::

        coord = MultiTierCacheCoordinator()
        coord.store("prefix_hash_abc", kv_data=b"...")

        # Lookup searches GPU -> CPU -> SSD
        data = coord.lookup("prefix_hash_abc")

        # Promote from SSD to GPU
        coord.promote("prefix_hash_abc")

        # Auto-rebalance based on access
        coord.rebalance()
    """

    def __init__(
        self,
        gpu_capacity: int = 1024,
        cpu_capacity: int = 4096,
        ssd_capacity: int = 65536,
        rebalance_interval: float = 10.0,
        promote_on_hit: bool = True,
    ) -> None:
        self._gpu_capacity = gpu_capacity
        self._cpu_capacity = cpu_capacity
        self._ssd_capacity = ssd_capacity
        self._rebalance_interval = rebalance_interval
        self._promote_on_hit = promote_on_hit

        self._lock = threading.Lock()

        # Per-tier stores: prefix_hash -> (data, last_access_time, access_count)
        self._gpu_cache: OrderedDict[str, tuple[bytes, float, int]] = OrderedDict()
        self._cpu_cache: OrderedDict[str, tuple[bytes, float, int]] = OrderedDict()
        self._ssd_cache: OrderedDict[str, tuple[bytes, float, int]] = OrderedDict()

        # Tier ordering for lookup
        self._tier_stores: list[tuple[KVTier, OrderedDict]] = [
            (KVTier.HOT, self._gpu_cache),
            (KVTier.WARM, self._cpu_cache),
            (KVTier.SSD, self._ssd_cache),
        ]

        # Per-tier stats
        self._tier_stats: dict[KVTier, TierStats] = {
            KVTier.HOT: TierStats(tier=KVTier.HOT, max_capacity=gpu_capacity),
            KVTier.WARM: TierStats(tier=KVTier.WARM, max_capacity=cpu_capacity),
            KVTier.SSD: TierStats(tier=KVTier.SSD, max_capacity=ssd_capacity),
        }

        self._last_rebalance_time = time.monotonic()

    def _locate(self, prefix_hash: str) -> tuple[KVTier, OrderedDict] | None:
        """Find which tier holds a prefix."""
        for tier, store in self._tier_stores:
            if prefix_hash in store:
                return tier, store
        return None

    def _store_for_tier(self, tier: KVTier) -> OrderedDict:
        """Return the store for a given tier."""
        mapping = {
            KVTier.HOT: self._gpu_cache,
            KVTier.WARM: self._cpu_cache,
            KVTier.SSD: self._ssd_cache,
        }
        return mapping[tier]

    # ── Public API ──────────────────────────────────────────────────

    def lookup(self, token_prefix: str | bytes) -> bytes | None:
        """Search all tiers for a token prefix, return from fastest.

        Searches GPU -> CPU -> SSD. If found on a slower tier and
        promote_on_hit is True, promotes to GPU.

        Args:
            token_prefix: Prefix string or bytes to look up.

        Returns:
            Data if found, None otherwise.
        """
        key = _prefix_hash(token_prefix)
        with self._lock:
            location = self._locate(key)
            if location is None:
                # Record miss on all tiers
                for ts in self._tier_stats.values():
                    ts.miss_count += 1
                return None

            found_tier, store = location
            data, last_access, access_count = store[key]
            access_count += 1
            last_access = time.monotonic()
            store[key] = (data, last_access, access_count)

            # Move to end for LRU ordering
            store.move_to_end(key)

            # Record hit on the tier where found, miss on faster tiers
            tier_order = [KVTier.HOT, KVTier.WARM, KVTier.SSD]
            for t in tier_order:
                if t == found_tier:
                    self._tier_stats[t].hit_count += 1
                    break
                else:
                    self._tier_stats[t].miss_count += 1

            # Auto-promote if found on slower tier
            if self._promote_on_hit and found_tier != KVTier.HOT:
                self._promote_internal(key, found_tier, KVTier.HOT)

            return data

    def store(
        self,
        token_prefix: str | bytes,
        kv_data: bytes,
        predicted_access: int = 0,
    ) -> KVTier:
        """Store KV data in the appropriate tier based on predicted access.

        High predicted access -> GPU, medium -> CPU, low -> SSD.

        Args:
            token_prefix: Prefix string or bytes.
            kv_data: KV cache data bytes.
            predicted_access: Predicted number of future accesses.

        Returns:
            Tier where data was stored.
        """
        key = _prefix_hash(token_prefix)

        # Determine target tier
        if predicted_access >= 10:
            tier = KVTier.HOT
        elif predicted_access >= 2:
            tier = KVTier.WARM
        else:
            tier = KVTier.SSD

        with self._lock:
            # Remove from any existing tier
            for _, store in self._tier_stores:
                store.pop(key, None)

            # Evict from target tier if at capacity
            self._evict_if_full(tier)

            # Store in target tier
            store = self._store_for_tier(tier)
            store[key] = (kv_data, time.monotonic(), 0)
            store.move_to_end(key)

            # Update tier stats
            ts = self._tier_stats[tier]
            ts.entry_count = len(store)

        return tier

    def promote(self, token_prefix: str | bytes) -> bool:
        """Move a prefix from slower tier to faster tier (GPU).

        Searches CPU then SSD for the prefix. If found, promotes to GPU.

        Args:
            token_prefix: Prefix to promote.

        Returns:
            True if promotion succeeded.
        """
        key = _prefix_hash(token_prefix)
        with self._lock:
            location = self._locate(key)
            if location is None:
                return False
            found_tier, _ = location
            if found_tier == KVTier.HOT:
                return True  # Already on fastest tier
            return self._promote_internal(key, found_tier, KVTier.HOT)

    def evict(self, token_prefix: str | bytes) -> bool:
        """Remove a prefix from all tiers.

        Args:
            token_prefix: Prefix to evict.

        Returns:
            True if the prefix was found and evicted.
        """
        key = _prefix_hash(token_prefix)
        with self._lock:
            found = False
            for _, store in self._tier_stores:
                if store.pop(key, None) is not None:
                    found = True
            return found

    def rebalance(self) -> int:
        """Rebalance block distribution across tiers based on access patterns.

        Moves blocks between tiers based on their access counts:
        - High access (>=10) on CPU/SSD -> promote to GPU
        - Low access (<2) on GPU/CPU -> demote to SSD
        - Medium access on SSD -> promote to CPU

        Returns:
            Number of blocks moved.
        """
        moved = 0
        now = time.monotonic()
        with self._lock:
            # Promote high-access blocks from slower tiers to GPU
            for tier, store in [(KVTier.WARM, self._cpu_cache), (KVTier.SSD, self._ssd_cache)]:
                to_promote = []
                for key, (data, last_access, access_count) in list(store.items()):
                    if access_count >= 10:
                        to_promote.append(key)
                for key in to_promote:
                    if self._promote_internal(key, tier, KVTier.HOT):
                        moved += 1

            # Demote low-access blocks from GPU to CPU
            to_demote = []
            for key, (data, last_access, access_count) in list(self._gpu_cache.items()):
                if access_count < 2:
                    to_demote.append(key)
            for key in to_demote:
                if self._promote_internal(key, KVTier.HOT, KVTier.WARM):
                    moved += 1

            # Demote low-access blocks from CPU to SSD
            to_demote = []
            for key, (data, last_access, access_count) in list(self._cpu_cache.items()):
                if access_count < 2:
                    to_demote.append(key)
            for key in to_demote:
                if self._promote_internal(key, KVTier.WARM, KVTier.SSD):
                    moved += 1

            # Update tier stats
            for ts in self._tier_stats.values():
                store = self._store_for_tier(ts.tier)
                ts.entry_count = len(store)

            self._last_rebalance_time = now

        if moved > 0:
            logger.debug("Cache rebalance: moved %d blocks", moved)
        return moved

    def get_tier_stats(self) -> dict[str, dict]:
        """Return per-tier utilization, hit rate, and capacity statistics."""
        with self._lock:
            result = {}
            for tier, ts in self._tier_stats.items():
                store = self._store_for_tier(tier)
                ts.entry_count = len(store)
                ts.byte_size = sum(len(d) for d, _, _ in store.values())
                result[tier.value] = {
                    "entry_count": ts.entry_count,
                    "byte_size": ts.byte_size,
                    "max_capacity": ts.max_capacity,
                    "hit_count": ts.hit_count,
                    "miss_count": ts.miss_count,
                    "hit_rate": round(ts.hit_rate, 4),
                    "utilization_pct": round(ts.utilization_pct, 1),
                }
            return result

    # ── Internal ────────────────────────────────────────────────────

    def promote_by_key(self, key: str) -> bool:
        """Promote a pre-hashed key from any slower tier to GPU.

        Unlike promote(), this takes an already-hashed key so callers
        that have pre-computed the hash don't need to pass the raw prefix.

        Args:
            key: The pre-computed blake2b hash key.

        Returns:
            True if promotion succeeded.
        """
        with self._lock:
            location = self._locate(key)
            if location is None:
                return False
            found_tier, _ = location
            if found_tier == KVTier.HOT:
                return True
            return self._promote_internal(key, found_tier, KVTier.HOT)

    def _promote_internal(
        self, key: str, source_tier: KVTier, dest_tier: KVTier
    ) -> bool:
        """Move a key from source to destination tier. Must hold _lock."""
        source_store = self._store_for_tier(source_tier)
        dest_store = self._store_for_tier(dest_tier)

        entry = source_store.pop(key, None)
        if entry is None:
            return False

        self._evict_if_full(dest_tier)
        dest_store[key] = entry
        dest_store.move_to_end(key)

        # Update stats
        self._tier_stats[source_tier].entry_count = len(source_store)
        self._tier_stats[dest_tier].entry_count = len(dest_store)

        return True

    def _evict_if_full(self, tier: KVTier) -> int:
        """Evict LRU entries from a tier if at capacity. Must hold _lock."""
        store = self._store_for_tier(tier)
        capacity = {
            KVTier.HOT: self._gpu_capacity,
            KVTier.WARM: self._cpu_capacity,
            KVTier.SSD: self._ssd_capacity,
        }[tier]

        evicted = 0
        while len(store) >= capacity:
            # Evict oldest (LRU)
            key, entry = store.popitem(last=False)
            # Demote to next slower tier
            if tier == KVTier.HOT:
                self._cpu_cache[key] = entry
            elif tier == KVTier.WARM:
                self._ssd_cache[key] = entry
            evicted += 1
        return evicted


# ── CacheWarmingScheduler ─────────────────────────────────────────────────


class CacheWarmingScheduler:
    """Proactively warms caches based on predicted future requests.

    Analyzes recent request patterns to predict which KV prefixes will
    be needed soon and preloads them from SSD to GPU before they are
    actually requested. Runs in a background thread that doesn't
    interfere with active inference.

    Prediction strategy:
    - Maintains a sliding window of recent prefix accesses.
    - Counts access frequency per prefix in the window.
    - Predicts that high-frequency prefixes will be accessed again.
    - Pre-loads those prefixes from slower tiers.

    Usage::

        scheduler = CacheWarmingScheduler(coordinator)
        scheduler.start()

        # Record request patterns
        scheduler.record_request("system_prompt_v2")
        scheduler.record_request("system_prompt_v2")

        # Background thread predicts and warms
        predictions = scheduler.predict_hot_prefixes(window_size=100)

        scheduler.stop()
    """

    def __init__(
        self,
        coordinator: MultiTierCacheCoordinator | None = None,
        window_size: int = 100,
        prediction_threshold: int = 2,
        warming_interval: float = 5.0,
        max_warming_ops: int = 32,
    ) -> None:
        self._coordinator = coordinator
        self._window_size = window_size
        self._prediction_threshold = prediction_threshold
        self._warming_interval = warming_interval
        self._max_warming_ops = max_warming_ops

        # Sliding window of recent prefix accesses
        self._access_window: collections.deque[str] = collections.deque(
            maxlen=window_size
        )
        self._lock = threading.Lock()

        # Background thread
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Statistics
        self._stats = WarmingStats()

        # Prediction tracking: prefix_hash -> predicted_at_time
        self._active_predictions: dict[str, float] = {}

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background warming thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._warming_loop,
            name="cache-warming",
            daemon=True,
        )
        self._thread.start()
        logger.info("Cache warming scheduler started")

    def stop(self) -> None:
        """Stop the background warming thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("Cache warming scheduler stopped")

    # ── Request Recording ───────────────────────────────────────────

    def record_request(self, token_prefix: str | bytes) -> None:
        """Record a prefix access for pattern analysis.

        Call this whenever a request uses a prefix (from the scheduler
        or engine request path).

        Args:
            token_prefix: The token prefix that was accessed.
        """
        key = _prefix_hash(token_prefix)
        with self._lock:
            self._access_window.append(key)

    # ── Prediction ──────────────────────────────────────────────────

    def predict_hot_prefixes(
        self, window_size: int | None = None
    ) -> list[WarmingPrediction]:
        """Predict which prefixes will be needed based on recent access patterns.

        Analyzes the sliding window of recent accesses and identifies
        prefixes accessed more than the threshold. Returns predictions
        sorted by confidence (highest first).

        Args:
            window_size: Override the default window size for this prediction.

        Returns:
            List of WarmingPredictions, sorted by confidence descending.
        """
        size = window_size or self._window_size
        with self._lock:
            # Get the last N accesses
            recent = list(self._access_window)[-size:]

        # Count frequency in window
        freq: dict[str, int] = {}
        for key in recent:
            freq[key] = freq.get(key, 0) + 1

        # Build predictions for high-frequency prefixes
        predictions = []
        total_accesses = len(recent) or 1
        now = time.monotonic()

        for key, count in freq.items():
            if count >= self._prediction_threshold:
                confidence = count / total_accesses
                # Estimate when it will next be accessed
                predicted_time = now + (total_accesses / max(count, 1))

                # Determine current tier from coordinator
                source_tier = KVTier.SSD  # default assumption
                if self._coordinator is not None:
                    loc = self._coordinator._locate(key)
                    if loc is not None:
                        source_tier = loc[0]

                predictions.append(WarmingPrediction(
                    prefix_hash=key,
                    confidence=confidence,
                    predicted_access_time=predicted_time,
                    source_tier=source_tier,
                    access_count_in_window=count,
                ))

        # Sort by confidence (highest first)
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        with self._lock:
            self._stats.predictions_made += len(predictions)
            for p in predictions:
                self._active_predictions[p.prefix_hash] = now

        return predictions

    # ── Warming Operations ──────────────────────────────────────────

    def warm_cache(self, prefix_hash: str) -> bool:
        """Preload a specific prefix from SSD to GPU.

        If the prefix is on a slower tier, promotes it to GPU.
        No-op if already on GPU.

        Args:
            prefix_hash: The hash of the prefix to warm, or a raw prefix
                         string that will be hashed automatically.

        Returns:
            True if warming was performed or data was already hot.
        """
        if self._coordinator is None:
            return False

        with self._lock:
            self._stats.warming_operations += 1

        # Normalize: if it looks like a hash (32 hex chars), use directly;
        # otherwise hash it. This supports both pre-hashed keys from
        # predictions and raw prefix strings from direct calls.
        key = prefix_hash
        if len(prefix_hash) != 32 or not all(c in "0123456789abcdef" for c in prefix_hash):
            key = _prefix_hash(prefix_hash)

        # Use coordinator's promote_by_key (key is now properly hashed)
        result = self._coordinator.promote_by_key(key)

        if result:
            with self._lock:
                self._stats.warming_bytes_transferred += 2048  # estimate

        return result

    def schedule_warming(
        self, predictions: list[WarmingPrediction]
    ) -> int:
        """Schedule warming operations for predicted hot prefixes.

        Takes a list of predictions and warms the top N (limited by
        max_warming_ops). Only warms prefixes on slower tiers.

        Args:
            predictions: List of predictions to warm.

        Returns:
            Number of prefixes successfully warmed.
        """
        warmed = 0
        limit = min(len(predictions), self._max_warming_ops)

        for pred in predictions[:limit]:
            if pred.source_tier == KVTier.HOT:
                # Already on fastest tier
                continue

            if self.warm_cache(pred.prefix_hash):
                warmed += 1

        return warmed

    def verify_prediction(self, prefix_hash: str, was_hit: bool) -> None:
        """Verify whether a prediction was correct.

        Call this when a prefix is actually accessed to check if it
        was previously predicted and warmed.

        Args:
            prefix_hash: The prefix that was accessed.
            was_hit: Whether it was a cache hit (warmed correctly).
        """
        with self._lock:
            if prefix_hash in self._active_predictions:
                if was_hit:
                    self._stats.predictions_correct += 1
                else:
                    self._stats.predictions_incorrect += 1
                self._active_predictions.pop(prefix_hash, None)

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return warming scheduler statistics."""
        with self._lock:
            return {
                "predictions_made": self._stats.predictions_made,
                "predictions_correct": self._stats.predictions_correct,
                "predictions_incorrect": self._stats.predictions_incorrect,
                "prediction_accuracy": round(self._stats.prediction_accuracy, 4),
                "warming_operations": self._stats.warming_operations,
                "warming_bytes_transferred": self._stats.warming_bytes_transferred,
                "hit_rate_before": round(self._stats.hit_rate_before, 4),
                "hit_rate_after": round(self._stats.hit_rate_after, 4),
                "window_size": self._window_size,
                "active_predictions": len(self._active_predictions),
            }

    # ── Internal ────────────────────────────────────────────────────

    def _warming_loop(self) -> None:
        """Background thread: periodically predict and warm."""
        while not self._stop_event.wait(timeout=self._warming_interval):
            try:
                predictions = self.predict_hot_prefixes()
                if predictions:
                    self.schedule_warming(predictions)
            except Exception:
                logger.debug("Cache warming loop error", exc_info=True)
