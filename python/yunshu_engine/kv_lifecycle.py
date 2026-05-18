from __future__ import annotations
"""KV lifecycle integration — unified KV cache lifecycle management.

Coordinates kv_migration (multi-tier migration), kv_optimizations
(adaptive quantization, eviction prediction, compaction), and
kv_prefix_compression (prefix compression, sliding window) into a
coherent pipeline managed by the engine.

Lifecycle phases:
  1. Admission — decide if KV block should be created (budget check)
  2. Placement — assign to hot/warm/cool/cold tier
  3. Optimization — quantize, compress, compact
  4. Migration — move between tiers based on access patterns
  5. Eviction — remove when memory pressure requires it
  6. Warming — pre-load predicted future blocks
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class KVTier(Enum):
    HOT = auto()    # GPU memory (MLX active)
    WARM = auto()   # CPU memory (MLX managed)
    COOL = auto()   # SSD (SQLite-backed)
    COLD = auto()   # Remote (compressed transfer)


@dataclass
class KVBlock:
    """Represents a KV cache block in the lifecycle."""
    block_id: int
    tier: KVTier = KVTier.HOT
    size_bytes: int = 0
    ref_count: int = 0
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0
    is_shared: bool = False
    quantization: str = "fp16"  # fp16, int8, int4
    compressed: bool = False
    model_hash: str = ""
    prefix_hash: str = ""

    def touch(self) -> None:
        self.last_access = time.monotonic()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_access


@dataclass
class KVTierConfig:
    """Configuration for a KV tier."""
    tier: KVTier
    max_bytes: int = 0
    block_size: int = 64
    eviction_policy: str = "lru"  # lru, lfu, priority, predictive
    quantization: str = "fp16"
    compression: bool = False


class KVLifecycleManager:
    """Manages the full KV cache lifecycle across tiers.

    Integration point:
    - BatchedEngine creates KVLifecycleManager in __init__
    - KV prefix cache operations go through lifecycle manager
    - EngineCore step loop triggers periodic optimization
    """

    def __init__(
        self,
        tier_configs: list[KVTierConfig] | None = None,
        optimization_interval: int = 100,
        migration_batch_size: int = 16,
    ) -> None:
        self._tier_configs: dict[KVTier, KVTierConfig] = {}
        if tier_configs:
            for cfg in tier_configs:
                self._tier_configs[cfg.tier] = cfg

        self._blocks: dict[int, KVBlock] = {}
        self._tier_usage: dict[KVTier, int] = defaultdict(int)
        self._optimization_interval = optimization_interval
        self._migration_batch_size = migration_batch_size
        self._step_counter = 0

        # Migration stats
        self._migrations = 0
        self._evictions = 0
        self._admissions_rejected = 0
        self._warming_hits = 0

        # Optimization triggers
        self._last_optimization = time.monotonic()

    def admit(self, block_id: int, size_bytes: int, prefix_hash: str = "", model_hash: str = "") -> bool:
        """Decide whether to admit a new KV block."""
        hot_config = self._tier_configs.get(KVTier.HOT)
        if hot_config and hot_config.max_bytes > 0:
            current = self._tier_usage.get(KVTier.HOT, 0)
            if current + size_bytes > hot_config.max_bytes:
                # Try to make room
                if not self._evict_for_space(size_bytes):
                    self._admissions_rejected += 1
                    return False

        block = KVBlock(
            block_id=block_id,
            tier=KVTier.HOT,
            size_bytes=size_bytes,
            prefix_hash=prefix_hash,
            model_hash=model_hash,
        )
        self._blocks[block_id] = block
        self._tier_usage[KVTier.HOT] += size_bytes
        return True

    def touch(self, block_id: int) -> None:
        """Mark block as recently accessed."""
        block = self._blocks.get(block_id)
        if block:
            block.touch()

    def share(self, block_id: int) -> None:
        """Increment reference count (shared block)."""
        block = self._blocks.get(block_id)
        if block:
            block.ref_count += 1
            block.is_shared = True

    def release(self, block_id: int) -> None:
        """Decrement reference count. Evict if zero and not actively shared."""
        block = self._blocks.get(block_id)
        if block:
            block.ref_count = max(0, block.ref_count - 1)
            if block.ref_count == 0:
                block.is_shared = False
                self._evict_block(block)

    def migrate(self, block_id: int, target_tier: KVTier) -> bool:
        """Migrate a block to a different tier."""
        block = self._blocks.get(block_id)
        if block is None:
            return False

        target_config = self._tier_configs.get(target_tier)
        if target_config and target_config.max_bytes > 0:
            current = self._tier_usage.get(target_tier, 0)
            if current + block.size_bytes > target_config.max_bytes:
                return False

        source_tier = block.tier
        self._tier_usage[source_tier] -= block.size_bytes
        block.tier = target_tier
        self._tier_usage[target_tier] += block.size_bytes

        # Apply tier-specific optimizations
        if target_config:
            block.quantization = target_config.quantization
            block.compressed = target_config.compression

        self._migrations += 1
        return True

    def optimize(self) -> dict:
        """Run periodic optimization pass.

        Returns stats about what was optimized.
        """
        self._step_counter += 1
        if self._step_counter % self._optimization_interval != 0:
            return {"skipped": True}

        start = time.monotonic()
        migrated = 0
        compacted = 0
        quantized = 0

        # Phase 1: Downgrade cold blocks
        for block in list(self._blocks.values()):
            if block.tier == KVTier.HOT and block.age_seconds > 60.0 and block.ref_count == 0:
                if self.migrate(block.block_id, KVTier.WARM):
                    migrated += 1
            elif block.tier == KVTier.WARM and block.age_seconds > 300.0:
                cool_config = self._tier_configs.get(KVTier.COOL)
                if cool_config:
                    if self.migrate(block.block_id, KVTier.COOL):
                        migrated += 1

        elapsed = time.monotonic() - start
        self._last_optimization = time.monotonic()

        return {
            "elapsed_ms": round(elapsed * 1000, 2),
            "blocks_migrated": migrated,
            "blocks_compacted": compacted,
            "blocks_quantized": quantized,
            "total_blocks": len(self._blocks),
        }

    def _evict_for_space(self, needed_bytes: int) -> bool:
        """Evict blocks to make room for new allocation."""
        hot_blocks = [
            b for b in self._blocks.values()
            if b.tier == KVTier.HOT and b.ref_count == 0 and not b.is_shared
        ]
        hot_blocks.sort(key=lambda b: b.last_access)

        freed = 0
        for block in hot_blocks:
            if freed >= needed_bytes:
                break
            freed += block.size_bytes
            self._evict_block(block)

        return freed >= needed_bytes

    def _evict_block(self, block: KVBlock) -> None:
        """Evict a single block."""
        self._tier_usage[block.tier] -= block.size_bytes
        self._blocks.pop(block.block_id, None)
        self._evictions += 1

    def get_block(self, block_id: int) -> KVBlock | None:
        return self._blocks.get(block_id)

    @property
    def total_blocks(self) -> int:
        return len(self._blocks)

    @property
    def tier_usage_bytes(self) -> dict[str, int]:
        return {tier.name: usage for tier, usage in self._tier_usage.items()}

    def get_stats(self) -> dict:
        tier_blocks = defaultdict(int)
        tier_bytes = defaultdict(int)
        for block in self._blocks.values():
            tier_blocks[block.tier.name] += 1
            tier_bytes[block.tier.name] += block.size_bytes

        return {
            "total_blocks": len(self._blocks),
            "tier_blocks": dict(tier_blocks),
            "tier_bytes": {k: v for k, v in tier_bytes.items()},
            "migrations": self._migrations,
            "evictions": self._evictions,
            "admissions_rejected": self._admissions_rejected,
            "warming_hits": self._warming_hits,
            "optimization_interval": self._optimization_interval,
            "last_optimization_ago_seconds": round(
                time.monotonic() - self._last_optimization, 1
            ),
        }


class CacheWarmingPredictor:
    """Predicts which KV blocks will be needed soon and pre-warms them.

    Uses a simple frequency + recency model to predict future access:
    - Track per-prefix access patterns
    - Predict which prefixes are likely to be reused
    - Pre-load from cool/cold tiers before they're requested
    """

    def __init__(
        self,
        prediction_window: float = 60.0,
        min_access_count: int = 2,
        warming_threshold: float = 0.5,
    ) -> None:
        self._prediction_window = prediction_window
        self._min_access = min_access_count
        self._warming_threshold = warming_threshold
        self._prefix_history: dict[str, list[float]] = defaultdict(list)
        self._predictions_correct = 0
        self._predictions_total = 0

    def record_access(self, prefix_hash: str) -> None:
        """Record a prefix access for pattern learning."""
        now = time.monotonic()
        history = self._prefix_history[prefix_hash]
        history.append(now)
        # Prune old entries
        cutoff = now - self._prediction_window * 2
        self._prefix_history[prefix_hash] = [
            t for t in history if t > cutoff
        ]

    def predict_warm_candidates(self) -> list[str]:
        """Return prefixes likely to be accessed soon."""
        now = time.monotonic()
        candidates = []

        for prefix_hash, history in self._prefix_history.items():
            if len(history) < self._min_access:
                continue

            # Compute access frequency (accesses per prediction_window)
            recent = [t for t in history if t > now - self._prediction_window]
            frequency = len(recent) / self._prediction_window

            # Recency boost: last access was recent
            last_access = max(history)
            recency = 1.0 / (1.0 + (now - last_access))

            # Score = frequency * recency
            score = frequency * recency

            if score >= self._warming_threshold:
                candidates.append(prefix_hash)

        return candidates

    def report_prediction(self, prefix_hash: str, was_correct: bool) -> None:
        self._predictions_total += 1
        if was_correct:
            self._predictions_correct += 1

    def get_stats(self) -> dict:
        accuracy = (
            self._predictions_correct / self._predictions_total
            if self._predictions_total > 0
            else 0.0
        )
        return {
            "tracked_prefixes": len(self._prefix_history),
            "predictions_total": self._predictions_total,
            "predictions_correct": self._predictions_correct,
            "accuracy": round(accuracy, 4),
        }


class KVCompactionScheduler:
    """Schedules KV block compaction to reduce fragmentation.

    Periodically scans partially-filled blocks and merges their contents
    into fuller blocks, freeing up blocks for reuse.
    """

    def __init__(
        self,
        min_fragmentation_ratio: float = 0.5,
        compaction_batch_size: int = 32,
        interval_steps: int = 200,
    ) -> None:
        self._min_frag = min_fragmentation_ratio
        self._batch_size = compaction_batch_size
        self._interval = interval_steps
        self._step_count = 0
        self._compactions_run = 0
        self._blocks_freed = 0

    def should_compact(self) -> bool:
        self._step_count += 1
        return self._step_count % self._interval == 0

    def compact(self, block_usage: dict[int, float]) -> int:
        """Compact partially-filled blocks.

        Args:
            block_usage: {block_id: fill_ratio} where 0.0-1.0

        Returns:
            Number of blocks freed.
        """
        fragmented = [
            (bid, ratio) for bid, ratio in block_usage.items()
            if ratio < self._min_frag
        ]
        fragmented.sort(key=lambda x: x[1])

        freed = 0
        batch: list[tuple[int, float]] = []

        for bid, ratio in fragmented:
            batch.append((bid, ratio))
            if len(batch) >= self._batch_size:
                freed += self._compact_batch(batch)
                batch = []

        if batch:
            freed += self._compact_batch(batch)

        self._compactions_run += 1
        self._blocks_freed += freed
        return freed

    def _compact_batch(self, batch: list[tuple[int, float]]) -> int:
        """Simulate compaction of a batch of partially-filled blocks.

        In production, this would merge KV data from partially-filled
        blocks into fuller blocks, then free the empty ones.
        """
        total_fill = sum(ratio for _, ratio in batch)
        # Each block can hold 1.0; N blocks with total_fill can be
        # compacted into ceil(total_fill) blocks
        needed = int(total_fill) + (1 if total_fill % 1 > 0 else 0)
        if needed == 0:
            needed = 1
        freed = len(batch) - needed
        return max(0, freed)

    def get_stats(self) -> dict:
        return {
            "compactions_run": self._compactions_run,
            "blocks_freed": self._blocks_freed,
            "step_count": self._step_count,
            "interval_steps": self._interval,
        }
