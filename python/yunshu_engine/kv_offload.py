from __future__ import annotations

"""Yunshu KV Offload Framework — async KV block offloading between tiers.

Provides a formal, policy-driven framework for moving KV cache blocks between
the four storage tiers (hot → warm → SSD → cold) with:

- **KVOffloadManager**: Orchestrates async offload operations with configurable
  policies. Integrates with both the scheduler (periodic memory-pressure
  offload) and the KV prefix cache (prefix-eviction offload).
- **OffloadPolicy ABC**: Pluggable eviction policies. Three implementations:
  - ThresholdPolicy: offload when memory exceeds a fraction of UMA.
  - LRUPolicy: offload least-recently-used blocks.
  - PriorityPolicy: offload by assigned priority (lowest priority first).
- **OffloadRequest / OffloadResult**: Dataclasses for tracking individual
  async offload operations (block_hash, source_tier, dest_tier, latency, etc.).
- **KVOffloadConfig**: Centralized configuration, including env var support
  (``YUNSHU_KV_OFFLOAD=1`` to enable).

Usage::

    from yunshu_engine.kv_offload import KVOffloadManager, KVOffloadConfig

    config = KVOffloadConfig.from_env()
    manager = KVOffloadManager(config, kv_manager=kv_mgr)
    await manager.start()

    # Periodic offload check (called from scheduler step loop)
    await manager.maybe_offload()

    # Per-request offload (from KV prefix cache eviction)
    result = await manager.offload_blocks(block_hashes=[h1, h2])

    # On-demand promotion (hot miss → warm/SSD lookup)
    kv_data = await manager.promote_block(block_hash)

    await manager.stop()

Thread safety:
- All async methods are safe to call from the asyncio event loop.
- Sync methods (offload_blocks_sync, promote_block_sync) are safe to call
  from the MLX executor thread (no await, no asyncio dependency).
"""

import asyncio
import contextlib
import enum
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────


class KVTier(enum.Enum):
    """KV cache storage tier."""

    HOT = "hot"  # UMA-resident FP16 (fastest, limited by RAM)
    WARM = "warm"  # In-memory 4-bit quantized (KVWarmTier)
    SSD = "ssd"  # SSD-backed (SSDCacheStore or SSDKVCache)
    COLD = "cold"  # Evicted / not stored (placeholder for future)


class OffloadStatus(enum.Enum):
    """Status of an offload operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Block already at destination tier
    CANCELLED = "cancelled"


# ── Data Classes ───────────────────────────────────────────────────────


@dataclass
class KVOffloadConfig:
    """Configuration for the KV offload framework.

    All fields have sensible defaults. Use ``from_env()`` to enable via
    environment variables.
    """

    enabled: bool = False
    # Policy selection: "threshold", "lru", "priority"
    policy: str = "threshold"
    # ThresholdPolicy parameters
    memory_threshold: float = 0.80  # Offload when active_mem/total > threshold
    target_memory_fraction: float = 0.65  # Offload until this fraction is reached
    # LRU policy parameters
    lru_max_blocks_per_cycle: int = 64  # Max blocks to offload per cycle
    lru_min_age_seconds: float = 30.0  # Min age before eligible for offload
    # PriorityPolicy parameters
    priority_low_watermark: int = 1  # Priority at or below this is offload-eligible
    # General parameters
    offload_interval_steps: int = 32  # Check every N scheduler steps
    max_concurrent_offloads: int = 4  # Max in-flight offload operations
    async_mode: bool = True  # True = asyncio, False = sync (executor thread)
    block_size: int = 64  # Tokens per block (for stats estimation)
    bytes_per_block: int = 0  # Estimated bytes per block (0 = auto-detect)
    # Promotion (warm/SSD → hot)
    promote_on_miss: bool = True  # Auto-promote on hot cache miss
    # SSD persistence
    ssd_cache_dir: str = ""  # SSD cache directory (empty = no SSD tier)

    @classmethod
    def from_env(cls) -> KVOffloadConfig:
        """Create config from environment variables.

        Env vars:
            YUNSHU_KV_OFFLOAD=1             Enable KV offloading
            YUNSHU_KV_OFFLOAD_POLICY        Policy name (threshold/lru/priority)
            YUNSHU_KV_OFFLOAD_THRESHOLD     Memory threshold (0.0–1.0)
            YUNSHU_KV_OFFLOAD_INTERVAL      Steps between offload checks
            YUNSHU_KV_OFFLOAD_ASYNC=0       Disable async mode (use sync)
            YUNSHU_SSD_CACHE_DIR            SSD cache directory
        """
        return cls(
            enabled=os.environ.get("YUNSHU_KV_OFFLOAD", "0") == "1",
            policy=os.environ.get("YUNSHU_KV_OFFLOAD_POLICY", "threshold"),
            memory_threshold=float(
                os.environ.get("YUNSHU_KV_OFFLOAD_THRESHOLD", "0.80")
            ),
            offload_interval_steps=int(
                os.environ.get("YUNSHU_KV_OFFLOAD_INTERVAL", "32")
            ),
            async_mode=os.environ.get("YUNSHU_KV_OFFLOAD_ASYNC", "1") != "0",
            ssd_cache_dir=os.environ.get("YUNSHU_SSD_CACHE_DIR", ""),
        )


@dataclass
class OffloadRequest:
    """Tracks a single async KV block offload operation.

    An offload request represents the intent to move one or more KV blocks
    from a source tier to a destination tier. The request is created by the
    KVOffloadManager and tracked through completion.
    """

    request_id: str
    block_hashes: list[int]
    source_tier: KVTier
    dest_tier: KVTier
    priority: int = 0
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    status: OffloadStatus = OffloadStatus.PENDING
    error: str | None = None
    # Estimated size (bytes) of all blocks in this request
    estimated_bytes: int = 0

    @property
    def latency_seconds(self) -> float | None:
        """Time from creation to completion."""
        if self.completed_at is not None and self.started_at is not None:
            return self.completed_at - self.started_at
        return None


@dataclass
class OffloadResult:
    """Result of a completed offload operation.

    Contains statistics about what was offloaded, any failures,
    and timing information.
    """

    request_id: str
    status: OffloadStatus
    blocks_offloaded: int = 0
    blocks_skipped: int = 0
    blocks_failed: int = 0
    bytes_offloaded: int = 0
    latency_seconds: float = 0.0
    source_tier: KVTier = KVTier.HOT
    dest_tier: KVTier = KVTier.WARM
    errors: list[str] = field(default_factory=list)


@dataclass
class OffloadStats:
    """Aggregate offload statistics (thread-safe via caller-held lock)."""

    total_offloads: int = 0
    total_blocks_offloaded: int = 0
    total_blocks_skipped: int = 0
    total_blocks_failed: int = 0
    total_bytes_offloaded: int = 0
    total_promotions: int = 0
    total_blocks_promoted: int = 0
    total_bytes_promoted: int = 0
    total_promotion_hits: int = 0
    total_promotion_misses: int = 0
    # Latency tracking (seconds)
    offload_latencies: list[float] = field(default_factory=list)
    promotion_latencies: list[float] = field(default_factory=list)
    # Last cycle info
    last_offload_step: int = 0
    last_offload_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_offload(self, result: OffloadResult, step: int) -> None:
        with self._lock:
            self.total_offloads += 1
            self.total_blocks_offloaded += result.blocks_offloaded
            self.total_blocks_skipped += result.blocks_skipped
            self.total_blocks_failed += result.blocks_failed
            self.total_bytes_offloaded += result.bytes_offloaded
            self.offload_latencies.append(result.latency_seconds)
            if len(self.offload_latencies) > 1000:
                self.offload_latencies = self.offload_latencies[-500:]
            self.last_offload_step = step
            self.last_offload_time = time.monotonic()

    def record_promotion(self, bytes_promoted: int, latency: float) -> None:
        with self._lock:
            self.total_promotions += 1
            self.total_blocks_promoted += 1
            self.total_promotion_hits += 1
            self.total_bytes_promoted += bytes_promoted
            self.promotion_latencies.append(latency)
            if len(self.promotion_latencies) > 1000:
                self.promotion_latencies = self.promotion_latencies[-500:]

    def record_miss(self) -> None:
        with self._lock:
            self.total_promotion_misses += 1

    @property
    def avg_offload_latency(self) -> float:
        """Average offload latency in seconds."""
        with self._lock:
            if not self.offload_latencies:
                return 0.0
            return sum(self.offload_latencies) / len(self.offload_latencies)

    @property
    def avg_promotion_latency(self) -> float:
        """Average promotion latency in seconds."""
        with self._lock:
            if not self.promotion_latencies:
                return 0.0
            return sum(self.promotion_latencies) / len(self.promotion_latencies)

    @property
    def promotion_hit_rate(self) -> float:
        """Promotion hit rate (warm/SSD → hot success rate)."""
        total = self.total_promotion_hits + self.total_promotion_misses
        if total == 0:
            return 0.0
        return self.total_promotion_hits / total


# ── Offload Policies ──────────────────────────────────────────────────


class OffloadPolicy(ABC):
    """Abstract base class for KV block offload policies.

    A policy determines WHICH blocks should be offloaded when the
    KVOffloadManager decides that offloading is necessary. The policy
    receives context about the current KV state and returns a list of
    block hashes eligible for offloading.

    Implementations:
    - ThresholdPolicy: offload based on memory usage threshold.
    - LRUPolicy: offload least-recently-used blocks.
    - PriorityPolicy: offload by priority (lowest first).
    """

    @abstractmethod
    def select_blocks(
        self,
        hot_manager: Any,
        max_blocks: int,
        context: dict[str, Any] | None = None,
    ) -> list[int]:
        """Select blocks for offloading.

        Args:
            hot_manager: The hot-tier KVCacheManager (or TieredKVCacheManager).
            max_blocks: Maximum number of blocks to select.
            context: Additional context (memory info, request state, etc.).

        Returns:
            List of block hashes eligible for offloading.
        """
        ...

    @abstractmethod
    def should_offload(self, context: dict[str, Any]) -> bool:
        """Determine whether offloading should occur.

        Args:
            context: Context dict with keys like 'memory_usage', 'free_blocks',
                     'step_counter', etc.

        Returns:
            True if offloading should be triggered.
        """
        ...


class ThresholdPolicy(OffloadPolicy):
    """Offload blocks when memory usage exceeds a threshold.

    This is the default policy. It checks the current memory utilization
    (active_mem / total_mem) against a configurable threshold. When exceeded,
    it selects blocks for offloading until the target memory fraction is reached.

    This matches the pattern used in vLLM's BlockSpaceManager and oMLX's
    memory-pressure eviction.
    """

    def __init__(
        self,
        memory_threshold: float = 0.80,
        target_fraction: float = 0.65,
        max_blocks_per_cycle: int = 64,
    ) -> None:
        self.memory_threshold = memory_threshold
        self.target_fraction = target_fraction
        self.max_blocks_per_cycle = max_blocks_per_cycle

    def should_offload(self, context: dict[str, Any]) -> bool:
        memory_usage = context.get("memory_usage", 0.0)
        return memory_usage >= self.memory_threshold

    def select_blocks(
        self,
        hot_manager: Any,
        max_blocks: int,
        context: dict[str, Any] | None = None,
    ) -> list[int]:
        """Select blocks for offloading based on memory pressure.

        Walks the hot manager's block pool to find unreferenced blocks
        that can be safely offloaded to warm/SSD.
        """
        blocks_to_offload: list = []
        try:
            pool = getattr(hot_manager, "block_pool", None)
            if pool is None:
                return blocks_to_offload

            # Find cache-only blocks (ref_count==0, in prefix cache, not actively used)
            # that have a hash (meaningful content)
            for block in pool.blocks:
                if len(blocks_to_offload) >= max_blocks:
                    break
                if (
                    block.ref_count == 0
                    and block.block_hash is not None
                    and block.cache_only
                ):
                    blocks_to_offload.append(block.block_hash)
        except Exception:
            logger.debug("ThresholdPolicy block selection failed", exc_info=True)

        return blocks_to_offload


class LRUPolicy(OffloadPolicy):
    """Offload least-recently-used blocks.

    Tracks access timestamps for each block and selects the oldest
    unreferenced blocks for offloading. This is particularly effective
    for workloads with temporal locality (multi-turn conversations).

    The warm tier already uses an OrderedDict for LRU tracking. This
    policy extends that to the hot tier selection process.
    """

    def __init__(
        self,
        max_blocks_per_cycle: int = 64,
        min_age_seconds: float = 30.0,
        memory_threshold: float = 0.70,
    ) -> None:
        self.max_blocks_per_cycle = max_blocks_per_cycle
        self.min_age_seconds = min_age_seconds
        self.memory_threshold = memory_threshold
        # Track block access times: block_hash → last_access_time
        self._access_times: dict[int, float] = {}

    def record_access(self, block_hash: int) -> None:
        """Record that a block was accessed (for LRU tracking)."""
        self._access_times[block_hash] = time.monotonic()

    def should_offload(self, context: dict[str, Any]) -> bool:
        memory_usage = context.get("memory_usage", 0.0)
        return memory_usage >= self.memory_threshold

    def select_blocks(
        self,
        hot_manager: Any,
        max_blocks: int,
        context: dict[str, Any] | None = None,
    ) -> list[int]:
        """Select the least recently used unreferenced blocks."""
        blocks_to_offload: list = []
        now = time.monotonic()

        try:
            pool = getattr(hot_manager, "block_pool", None)
            if pool is None:
                return blocks_to_offload

            candidates = []
            for block in pool.blocks:
                if not hasattr(block, "block_hash") or block.block_hash is None:
                    continue
                # Update access time for referenced (in-use) blocks so they
                # won't appear stale when later unreferenced.
                if hasattr(block, "ref_count") and block.ref_count > 0:
                    self._access_times[block.block_hash] = now
                    continue
                # Unreferenced block — check age
                last_access = self._access_times.get(
                    block.block_hash, block.block_id * 1000.0
                )
                age = now - last_access
                if age >= self.min_age_seconds:
                    candidates.append((last_access, block.block_hash))

            # Sort by access time (oldest first)
            candidates.sort(key=lambda x: x[0])

            limit = min(max_blocks, self.max_blocks_per_cycle)
            blocks_to_offload = [h for _, h in candidates[:limit]]

        except Exception:
            logger.debug("LRUPolicy block selection failed", exc_info=True)

        return blocks_to_offload


class PriorityPolicy(OffloadPolicy):
    """Offload blocks by priority level (lowest priority first).

    Each block can be assigned a priority (e.g., system prompt = high,
    long context = low). When offloading is needed, blocks with the
    lowest priority are offloaded first.

    Priority is determined by:
    1. Explicit priority assignment (if tracked in block metadata)
    2. Fallback: block recency (older = lower priority)
    """

    def __init__(
        self,
        low_watermark: int = 1,
        max_blocks_per_cycle: int = 64,
        memory_threshold: float = 0.75,
    ) -> None:
        self.low_watermark = low_watermark
        self.max_blocks_per_cycle = max_blocks_per_cycle
        self.memory_threshold = memory_threshold
        # Block priorities: block_hash → priority (higher = more important)
        self._priorities: dict[int, int] = {}

    def set_priority(self, block_hash: int, priority: int) -> None:
        """Set the priority for a block."""
        self._priorities[block_hash] = priority

    def should_offload(self, context: dict[str, Any]) -> bool:
        memory_usage = context.get("memory_usage", 0.0)
        return memory_usage >= self.memory_threshold

    def select_blocks(
        self,
        hot_manager: Any,
        max_blocks: int,
        context: dict[str, Any] | None = None,
    ) -> list[int]:
        """Select blocks with lowest priority for offloading."""
        blocks_to_offload: list = []

        try:
            pool = getattr(hot_manager, "block_pool", None)
            if pool is None:
                return blocks_to_offload

            candidates = []
            for block in pool.blocks:
                if (
                    hasattr(block, "ref_count")
                    and block.ref_count == 0
                    and hasattr(block, "block_hash")
                    and block.block_hash is not None
                ):
                    priority = self._priorities.get(block.block_hash, 0)
                    if priority <= self.low_watermark:
                        candidates.append((priority, block.block_hash))

            # Sort by priority (lowest first)
            candidates.sort(key=lambda x: x[0])

            limit = min(max_blocks, self.max_blocks_per_cycle)
            blocks_to_offload = [h for _, h in candidates[:limit]]

        except Exception:
            logger.debug("PriorityPolicy block selection failed", exc_info=True)

        return blocks_to_offload


# ── Policy Factory ─────────────────────────────────────────────────────


def create_offload_policy(config: KVOffloadConfig) -> OffloadPolicy:
    """Create an OffloadPolicy from configuration.

    Args:
        config: KV offload configuration.

    Returns:
        An OffloadPolicy instance.

    Raises:
        ValueError: If the policy name is not recognized.
    """
    policy_name = config.policy.lower()
    if policy_name == "threshold":
        return ThresholdPolicy(
            memory_threshold=config.memory_threshold,
            target_fraction=config.target_memory_fraction,
            max_blocks_per_cycle=config.lru_max_blocks_per_cycle,
        )
    elif policy_name == "lru":
        return LRUPolicy(
            max_blocks_per_cycle=config.lru_max_blocks_per_cycle,
            min_age_seconds=config.lru_min_age_seconds,
            memory_threshold=config.memory_threshold,
        )
    elif policy_name == "priority":
        return PriorityPolicy(
            low_watermark=config.priority_low_watermark,
            max_blocks_per_cycle=config.lru_max_blocks_per_cycle,
            memory_threshold=config.memory_threshold,
        )
    else:
        raise ValueError(
            f"Unknown offload policy: {policy_name!r}. "
            f"Supported: threshold, lru, priority"
        )


# ── KV Offload Manager ────────────────────────────────────────────────


class KVOffloadManager:
    """Manages async KV block offloading between tiers.

    This is the central orchestrator for the KV offload framework. It:
    1. Periodically checks if offloading is needed (based on policy)
    2. Selects blocks for offloading (delegated to OffloadPolicy)
    3. Executes the actual tier-to-tier transfer (hot → warm → SSD)
    4. Tracks statistics (bytes, latency, hit/miss rates)
    5. Supports both async (event loop) and sync (executor thread) modes

    Integration points:
    - **Scheduler**: calls ``maybe_offload()`` every N steps
    - **KV prefix cache**: calls ``offload_blocks()`` for explicit eviction
    - **TieredKVCacheManager**: ``promote_block()`` on hot cache miss

    The manager is created by EngineCore during initialization and
    stored as ``EngineCore._kv_offload_manager``. The scheduler accesses
    it via a reference passed during setup.
    """

    def __init__(
        self,
        config: KVOffloadConfig,
        kv_manager: Any | None = None,
        policy: OffloadPolicy | None = None,
    ) -> None:
        self.config = config
        self._kv_manager = kv_manager
        self._policy = policy or create_offload_policy(config)

        # State tracking
        self._step_counter: int = 0
        self._pending_requests: dict[str, OffloadRequest] = {}
        self._completed_results: list[OffloadResult] = []
        self._max_completed_history = 100  # Keep last N results

        # Statistics
        self._stats = OffloadStats()
        self._stats_lock = threading.Lock()

        # Background task for async offload
        self._offload_queue: asyncio.Queue[OffloadRequest] | None = None
        self._worker_task: asyncio.Task | None = None
        self._running = False

        # Estimate bytes per block (auto-detect from kv_manager)
        self._bytes_per_block = config.bytes_per_block
        if self._bytes_per_block == 0 and kv_manager is not None:
            getattr(kv_manager, "block_size", config.block_size)
            # Rough estimate: num_layers * 2 (K+V) * block_size * head_dim * num_kv_heads * 2 bytes (fp16)
            # Default conservative estimate: 2KB per block
            self._bytes_per_block = 2048

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the offload manager (begins background worker if async)."""
        if not self.config.enabled:
            logger.debug("KV offload manager disabled")
            return

        self._running = True

        if self.config.async_mode:
            self._offload_queue = asyncio.Queue(
                maxsize=self.config.max_concurrent_offloads * 4
            )
            self._worker_task = asyncio.get_running_loop().create_task(
                self._async_worker()
            )
            logger.info(
                "KV offload manager started (async mode, policy=%s, threshold=%.0f%%)",
                self.config.policy,
                self.config.memory_threshold * 100,
            )
        else:
            logger.info(
                "KV offload manager started (sync mode, policy=%s, threshold=%.0f%%)",
                self.config.policy,
                self.config.memory_threshold * 100,
            )

    async def stop(self) -> None:
        """Stop the offload manager and cancel pending operations."""
        self._running = False

        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        # Cancel any pending requests
        for req in self._pending_requests.values():
            if req.status == OffloadStatus.PENDING:
                req.status = OffloadStatus.CANCELLED
        self._pending_requests.clear()

        logger.info("KV offload manager stopped")

    # ── Main API ───────────────────────────────────────────────────

    async def maybe_offload(
        self, context: dict[str, Any] | None = None
    ) -> OffloadResult | None:
        """Check if offloading is needed and execute if so.

        Called periodically from the scheduler's step loop. Uses the
        configured policy to determine:
        1. Whether offloading is needed (should_offload)
        2. Which blocks to offload (select_blocks)
        3. Executes the tier transfer

        Args:
            context: Additional context (memory_usage, free_blocks, etc.).
                     If None, gathers context automatically.

        Returns:
            OffloadResult if offloading was performed, None otherwise.
        """
        if not self.config.enabled or self._kv_manager is None:
            return None

        self._step_counter += 1

        # Check if it's time for an offload check
        if self._step_counter % self.config.offload_interval_steps != 0:
            return None

        # Gather context if not provided
        if context is None:
            context = self._gather_context()

        # Ask policy if offloading is needed
        if not self._policy.should_offload(context):
            return None

        # Select blocks to offload
        max_blocks = context.get("max_blocks", self.config.lru_max_blocks_per_cycle)
        block_hashes = self._policy.select_blocks(
            self._get_hot_manager(), max_blocks, context
        )

        if not block_hashes:
            return None

        # Execute offload
        if self.config.async_mode:
            return await self.offload_blocks(block_hashes)
        else:
            return self.offload_blocks_sync(block_hashes)

    async def offload_blocks(
        self,
        block_hashes: list[int],
        source_tier: KVTier = KVTier.HOT,
        dest_tier: KVTier = KVTier.WARM,
        priority: int = 0,
    ) -> OffloadResult:
        """Offload blocks from source tier to destination tier (async).

        Creates an OffloadRequest, enqueues it, and waits for completion.

        Args:
            block_hashes: Block hashes to offload.
            source_tier: Current tier of the blocks.
            dest_tier: Target tier for offloading.
            priority: Priority for the offload operation.

        Returns:
            OffloadResult with operation statistics.
        """
        request_id = f"offload-{self._step_counter:06d}-{len(block_hashes)}"
        request = OffloadRequest(
            request_id=request_id,
            block_hashes=block_hashes,
            source_tier=source_tier,
            dest_tier=dest_tier,
            priority=priority,
            estimated_bytes=len(block_hashes) * self._bytes_per_block,
        )

        if not self.config.enabled or self._kv_manager is None:
            request.status = OffloadStatus.SKIPPED
            return self._request_to_result(request)

        if self.config.async_mode and self._offload_queue is not None:
            # Async path: enqueue and wait
            self._pending_requests[request_id] = request
            await self._offload_queue.put(request)

            # Wait for completion (with timeout)
            try:
                result = await asyncio.wait_for(
                    self._wait_for_completion(request_id),
                    timeout=30.0,
                )
                return result
            except TimeoutError:
                request.status = OffloadStatus.FAILED
                request.error = "Offload timed out after 30s"
                self._pending_requests.pop(request_id, None)
                return self._request_to_result(request)
        else:
            # Fallback to sync path
            return self._execute_offload(request)

    def offload_blocks_sync(
        self,
        block_hashes: list[int],
        source_tier: KVTier = KVTier.HOT,
        dest_tier: KVTier = KVTier.WARM,
    ) -> OffloadResult:
        """Offload blocks synchronously (for use on executor thread).

        This is the sync variant of ``offload_blocks()`` that does not
        require an asyncio event loop. Safe to call from the MLX executor.

        Args:
            block_hashes: Block hashes to offload.
            source_tier: Current tier of the blocks.
            dest_tier: Target tier for offloading.

        Returns:
            OffloadResult with operation statistics.
        """
        request_id = f"offload-sync-{self._step_counter:06d}-{len(block_hashes)}"
        request = OffloadRequest(
            request_id=request_id,
            block_hashes=block_hashes,
            source_tier=source_tier,
            dest_tier=dest_tier,
        )

        if not self.config.enabled or self._kv_manager is None:
            request.status = OffloadStatus.SKIPPED
            return self._request_to_result(request)

        return self._execute_offload(request)

    async def promote_block(
        self,
        block_hash: int,
    ) -> Any | None:
        """Promote a block from warm/SSD tier back to hot tier.

        Called on hot cache miss. Checks warm tier first, then SSD.
        Returns the KV data if found, None otherwise.

        Args:
            block_hash: Hash of the block to promote.

        Returns:
            KV data if promotion succeeded, None otherwise.
        """
        if not self.config.promote_on_miss:
            return None

        start = time.monotonic()
        self._get_hot_manager()

        # Get the tiered manager's warm and SSD stores
        warm_tier = self._get_warm_tier()
        ssd_store = self._get_ssd_store()

        # Try warm tier first
        if warm_tier is not None:
            try:
                kv_data = warm_tier.promote(block_hash)
                if kv_data is not None:
                    latency = time.monotonic() - start
                    self._stats.record_promotion(self._bytes_per_block, latency)
                    logger.debug(
                        "Promoted block 0x%x from warm tier (%.1f ms)",
                        block_hash,
                        latency * 1000,
                    )
                    return kv_data
            except Exception:
                logger.debug(
                    "Warm tier promotion failed for 0x%x", block_hash, exc_info=True
                )

        # Try SSD store
        if ssd_store is not None:
            try:
                kv_data = ssd_store.load(block_hash)
                if kv_data is not None:
                    latency = time.monotonic() - start
                    self._stats.record_promotion(self._bytes_per_block, latency)
                    logger.debug(
                        "Promoted block 0x%x from SSD (%.1f ms)",
                        block_hash,
                        latency * 1000,
                    )
                    return kv_data
            except Exception:
                logger.debug("SSD promotion failed for 0x%x", block_hash, exc_info=True)

        # Miss
        self._stats.record_miss()
        return None

    def promote_block_sync(self, block_hash: int) -> Any | None:
        """Synchronous variant of promote_block (for executor thread).

        This does the same work as ``promote_block()`` but without
        the async wrapper, making it safe for the MLX executor thread.

        Args:
            block_hash: Hash of the block to promote.

        Returns:
            KV data if promotion succeeded, None otherwise.
        """
        if not self.config.promote_on_miss:
            return None

        start = time.monotonic()
        warm_tier = self._get_warm_tier()
        ssd_store = self._get_ssd_store()

        # Try warm tier
        if warm_tier is not None:
            try:
                kv_data = warm_tier.promote(block_hash)
                if kv_data is not None:
                    latency = time.monotonic() - start
                    self._stats.record_promotion(self._bytes_per_block, latency)
                    return kv_data
            except Exception:
                logger.debug("Warm tier promotion failed", exc_info=True)

        # Try SSD
        if ssd_store is not None:
            try:
                kv_data = ssd_store.load(block_hash)
                if kv_data is not None:
                    latency = time.monotonic() - start
                    self._stats.record_promotion(self._bytes_per_block, latency)
                    return kv_data
            except Exception:
                logger.debug("SSD promotion failed", exc_info=True)

        self._stats.record_miss()
        return None

    # ── Context / Stats ────────────────────────────────────────────

    def set_kv_manager(self, kv_manager: Any) -> None:
        """Update the KV manager reference (called after tiered setup)."""
        self._kv_manager = kv_manager

    def get_stats(self) -> dict:
        """Return offload manager statistics."""
        return {
            "enabled": self.config.enabled,
            "policy": self.config.policy,
            "async_mode": self.config.async_mode,
            "step_counter": self._step_counter,
            "total_offloads": self._stats.total_offloads,
            "total_blocks_offloaded": self._stats.total_blocks_offloaded,
            "total_bytes_offloaded": self._stats.total_bytes_offloaded,
            "total_blocks_skipped": self._stats.total_blocks_skipped,
            "total_blocks_failed": self._stats.total_blocks_failed,
            "total_promotions": self._stats.total_promotions,
            "total_blocks_promoted": self._stats.total_blocks_promoted,
            "total_bytes_promoted": self._stats.total_bytes_promoted,
            "promotion_hit_rate": round(self._stats.promotion_hit_rate, 4),
            "avg_offload_latency_ms": round(self._stats.avg_offload_latency * 1000, 2),
            "avg_promotion_latency_ms": round(
                self._stats.avg_promotion_latency * 1000, 2
            ),
            "last_offload_step": self._stats.last_offload_step,
            "pending_requests": len(self._pending_requests),
            "completed_history": len(self._completed_results),
        }

    # ── Internal ───────────────────────────────────────────────────

    def _gather_context(self) -> dict[str, Any]:
        """Gather context for the offload policy decision."""
        context: dict[str, Any] = {
            "step_counter": self._step_counter,
            "memory_usage": 0.0,
            "free_blocks": 0,
        }

        try:
            import mlx.core as mx

            active_mem = mx.get_active_memory()
            from .utils.hardware import get_hardware_info

            hw = get_hardware_info()
            total_mem = hw.total_memory_bytes
            if total_mem > 0:
                context["memory_usage"] = active_mem / total_mem
        except Exception:
            logger.debug("memory context gather failed", exc_info=True)

        if self._kv_manager is not None:
            context["free_blocks"] = getattr(self._kv_manager, "num_free_blocks", 0)
            pool = getattr(self._kv_manager, "block_pool", None)
            if pool is not None:
                context["total_blocks"] = getattr(pool, "num_blocks", 0)

        return context

    def _get_hot_manager(self) -> Any:
        """Get the hot-tier KVCacheManager (from TieredKVCacheManager or direct)."""
        if self._kv_manager is None:
            return None
        # If tiered, the hot manager is self._kv_manager.hot
        if hasattr(self._kv_manager, "hot"):
            return self._kv_manager.hot
        return self._kv_manager

    def _get_warm_tier(self) -> Any:
        """Get the warm tier (KVWarmTier) if available."""
        if self._kv_manager is None:
            return None
        if hasattr(self._kv_manager, "warm"):
            return self._kv_manager.warm
        return None

    def _get_ssd_store(self) -> Any:
        """Get the SSD store if available."""
        if self._kv_manager is None:
            return None
        if hasattr(self._kv_manager, "ssd"):
            return self._kv_manager.ssd
        return None

    def _execute_offload(self, request: OffloadRequest) -> OffloadResult:
        """Execute an offload operation (works in both sync and async contexts).

        Moves blocks from source tier to destination tier by:
        1. Extracting KV data from the source tier
        2. Storing to the destination tier
        3. Updating statistics
        """
        request.started_at = time.monotonic()
        request.status = OffloadStatus.IN_PROGRESS

        result = OffloadResult(
            request_id=request.request_id,
            status=OffloadStatus.COMPLETED,
            source_tier=request.source_tier,
            dest_tier=request.dest_tier,
        )

        hot_mgr = self._get_hot_manager()
        warm_tier = self._get_warm_tier()
        ssd_store = self._get_ssd_store()

        for block_hash in request.block_hashes:
            try:
                offloaded = False

                if (
                    request.source_tier == KVTier.HOT
                    and request.dest_tier == KVTier.WARM
                ):
                    # Hot → Warm: demote via warm tier, then free hot block
                    if warm_tier is not None:
                        kv_data = self._extract_hot_kv(hot_mgr, block_hash)
                        if kv_data is not None:
                            # Pass num_tokens : without it the warm block
                            # defaults to -1 → invisible to prefix matching if it
                            # later flushes to SSD (matches the SSD branch below).
                            if warm_tier.demote(
                                block_hash, kv_data, num_tokens=self.config.block_size
                            ):
                                self._free_hot_block(hot_mgr, block_hash)
                                offloaded = True

                elif (
                    request.source_tier == KVTier.HOT
                    and request.dest_tier == KVTier.SSD
                ):
                    # Hot → SSD: persist directly, then free hot block
                    if ssd_store is not None:
                        kv_data = self._extract_hot_kv(hot_mgr, block_hash)
                        if kv_data is not None:
                            if ssd_store.store(
                                block_hash, kv_data, num_tokens=self.config.block_size
                            ):
                                self._free_hot_block(hot_mgr, block_hash)
                                offloaded = True

                elif (
                    request.source_tier == KVTier.WARM
                    and request.dest_tier == KVTier.SSD
                ):
                    # Warm → SSD: promote from warm, persist to SSD, then evict warm
                    # If SSD write fails, re-insert back into warm to prevent data loss
                    if warm_tier is not None and ssd_store is not None:
                        kv_data = warm_tier.promote(block_hash)
                        if kv_data is not None:
                            if ssd_store.store(
                                block_hash, kv_data, num_tokens=self.config.block_size
                            ):
                                offloaded = True
                            else:
                                # SSD write failed — re-insert into warm tier
                                re_inserted = warm_tier.demote(
                                    block_hash,
                                    kv_data,
                                    num_tokens=self.config.block_size,
                                )
                                if not re_inserted:
                                    logger.error(
                                        "KV data loss risk: block 0x%x failed SSD write "
                                        "AND warm tier re-insertion failed",
                                        block_hash,
                                    )

                if offloaded:
                    result.blocks_offloaded += 1
                    result.bytes_offloaded += self._bytes_per_block
                else:
                    result.blocks_skipped += 1

            except Exception as e:
                result.blocks_failed += 1
                result.errors.append(f"Block 0x{block_hash:x}: {e}")
                logger.debug("Offload failed for block 0x%x: %s", block_hash, e)

        request.completed_at = time.monotonic()
        if result.blocks_offloaded > 0 and result.blocks_failed == 0:
            request.status = OffloadStatus.COMPLETED
        elif result.blocks_offloaded > 0:
            request.status = OffloadStatus.COMPLETED
            result.errors.insert(
                0,
                f"Partial: {result.blocks_failed} blocks failed out of {result.blocks_offloaded + result.blocks_failed + result.blocks_skipped}",
            )
        elif result.blocks_failed > 0:
            request.status = OffloadStatus.FAILED
        else:
            request.status = OffloadStatus.SKIPPED
        result.status = request.status
        result.latency_seconds = request.latency_seconds or 0.0

        # Update aggregate stats
        self._stats.record_offload(result, self._step_counter)

        # Store result in history
        self._completed_results.append(result)
        if len(self._completed_results) > self._max_completed_history:
            self._completed_results = self._completed_results[
                -self._max_completed_history // 2 :
            ]

        if result.blocks_offloaded > 0:
            logger.info(
                "KV offload: %d blocks (%s → %s), %.1f KB, %.1f ms",
                result.blocks_offloaded,
                request.source_tier.value,
                request.dest_tier.value,
                result.bytes_offloaded / 1024,
                result.latency_seconds * 1000,
            )

        return result

    def _extract_hot_kv(self, hot_mgr: Any, block_hash: int) -> Any:
        """Extract KV data for a block from the hot tier.

        Uses the TieredKVCacheManager's _extract_kv_for_block method
        if available, or falls back to direct access.
        """
        if hot_mgr is None:
            return None

        # Try TieredKVCacheManager extraction method
        if hasattr(self._kv_manager, "_extract_kv_for_block"):
            pool = getattr(hot_mgr, "block_pool", None)
            if pool is not None:
                block = pool.lookup_hash(block_hash)
                if block is not None:
                    return self._kv_manager._extract_kv_for_block(block)

        # Fallback: direct extraction from hot manager's KV layers
        if hasattr(hot_mgr, "_kv_layers") and hot_mgr._kv_layers:
            try:
                pool = getattr(hot_mgr, "block_pool", None)
                if pool is not None:
                    block = pool.lookup_hash(block_hash)
                    if block is not None:
                        import mlx.core as mx

                        kv_parts = []
                        for layer_caches in hot_mgr._kv_layers:
                            if block.block_id < len(layer_caches):
                                key_cache, val_cache = layer_caches[block.block_id]
                                if val_cache is not None:
                                    kv_parts.append(
                                        mx.stack([key_cache, val_cache], axis=0)
                                    )
                                else:
                                    kv_parts.append(key_cache)
                        if kv_parts:
                            return (
                                mx.stack(kv_parts, axis=0)
                                if len(kv_parts) > 1
                                else kv_parts[0]
                            )
            except Exception:
                logger.debug("Direct KV extraction failed", exc_info=True)

        return None

    def _free_hot_block(self, hot_mgr: Any, block_hash: int) -> None:
        """Evict and free a hot-tier block after successful offload.

        Uses BlockPool.lookup_and_free() to combine hash lookup and eviction
        into a single lock scope, preventing TOCTOU races where a concurrent
        allocate() could claim the block between the separate lookup and free
        operations.
        """
        if hot_mgr is None:
            return
        pool = getattr(hot_mgr, "block_pool", None)
        if pool is None:
            return
        pool.lookup_and_free(block_hash)

    async def _wait_for_completion(self, request_id: str) -> OffloadResult:
        """Wait for an offload request to complete.

        Polls the pending requests dict until the status changes
        from PENDING/IN_PROGRESS to a terminal state.
        """
        max_wait = 30.0  # seconds
        poll_interval = 0.001  # 1ms
        elapsed = 0.0

        while elapsed < max_wait:
            request = self._pending_requests.get(request_id)
            if request is None:
                break
            if request.status in (
                OffloadStatus.COMPLETED,
                OffloadStatus.FAILED,
                OffloadStatus.CANCELLED,
                OffloadStatus.SKIPPED,
            ):
                self._pending_requests.pop(request_id, None)
                return self._request_to_result(request)

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Offload {request_id} did not complete in {max_wait}s")

    async def _async_worker(self) -> None:
        """Background worker for async offload operations.

        Processes OffloadRequests from the queue one at a time.
        """
        try:
            while self._running:
                try:
                    request = await asyncio.wait_for(
                        self._offload_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                # Execute the offload (this does the actual I/O)
                self._execute_offload(request)
                # Result is now tracked; _wait_for_completion will pick it up
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("KV offload worker error", exc_info=True)

    @staticmethod
    def _request_to_result(request: OffloadRequest) -> OffloadResult:
        """Convert an OffloadRequest to an OffloadResult."""
        return OffloadResult(
            request_id=request.request_id,
            status=request.status,
            blocks_offloaded=0
            if request.status != OffloadStatus.COMPLETED
            else len(request.block_hashes),
            source_tier=request.source_tier,
            dest_tier=request.dest_tier,
            latency_seconds=request.latency_seconds or 0.0,
            errors=[request.error] if request.error else [],
        )
