from __future__ import annotations

"""Yunshu KV Warm Tier — in-memory 4-bit quantized KV block cache.

Sits between the hot tier (FP16 UMA) and cold tier (SSD). When the hot
BlockPool is under memory pressure, KV blocks are demoted here with
4-bit quantization (4x compression). On cache hit, they are promoted
back to FP16 transparently.

Uses the existing compression module (quantize_kv_4bit / dequantize_kv_4bit)
for the actual quantization path. Metal kernel will replace the numpy path
in Phase 2.
"""


import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KVTierConfig:
    """Configuration for the warm KV tier."""

    max_blocks: int = 10000
    compression: str = "4bit"  # Only 4bit supported for now
    storage_path: str = ""  # SSD persistence path (stub)
    flush_interval_s: float = 60.0


class KVWarmTier:
    """In-memory compressed KV block cache for warm-tier storage.

    Blocks are stored as (packed_uint8, scales_fp16) tuples via 4-bit
    symmetric per-group quantization. An OrderedDict gives O(1) LRU
    eviction without the overhead of a separate linked list.
    """

    def __init__(self, config: KVTierConfig | None = None) -> None:
        self.config = config or KVTierConfig()
        self._store: OrderedDict[int, tuple] = OrderedDict()
        # Hit / miss tracking
        self._hits: int = 0
        self._misses: int = 0
        # Approximate memory accounting
        self._memory_used: int = 0
        # Flush bookkeeping
        self._last_flush: float = time.monotonic()
        # Thread safety for concurrent access from flush/demote/promote/evict
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        return len(self._store)

    @property
    def is_full(self) -> bool:
        return len(self._store) >= self.config.max_blocks

    # ── Core API ──────────────────────────────────────────────────

    def demote(self, block_hash: int, kv_data, num_tokens: int = -1) -> bool:
        """Accept a KV block from the hot tier, compress and store it.

        Args:
            block_hash: Hash identifying the block content.
            kv_data: FP16 KV data (MLX array or numpy array).
            num_tokens: Number of tokens in this block (for SSD flush fidelity).
                Defaults to -1 (sentinel). A warning is logged if <= 0 is
                explicitly passed, as zero token counts make SSD-flushed
                blocks invisible to prefix matching.

        Returns:
            True if the block was stored successfully.
        """
        if num_tokens <= 0:
            logger.warning(
                "demote(block_hash=0x%x): num_tokens=%d — "
                "SSD-flushed blocks will have zero token counts "
                "and be invisible to prefix matching",
                block_hash,
                num_tokens,
            )
        with self._lock:
            try:
                from .compression import quantize_kv_4bit

                packed, scales = quantize_kv_4bit(kv_data)
                # Recover original head_dim from the kv_data shape for correct dequantize
                head_dim = kv_data.shape[-1] if hasattr(kv_data, 'shape') else 0

                # Approximate memory accounting for the NEW entry
                packed_nbytes = (
                    np.array(packed).nbytes if not isinstance(packed, np.ndarray) else packed.nbytes
                )
                scales_nbytes = (
                    np.array(scales).nbytes if not isinstance(scales, np.ndarray) else scales.nbytes
                )
                new_entry_bytes = packed_nbytes + scales_nbytes

                # Subtract old entry's memory if overwriting an existing key
                if block_hash in self._store:
                    old_entry = self._store[block_hash]
                    old_packed_nbytes = (
                        np.array(old_entry[0]).nbytes
                        if not isinstance(old_entry[0], np.ndarray)
                        else old_entry[0].nbytes
                    )
                    old_scales_nbytes = (
                        np.array(old_entry[1]).nbytes
                        if not isinstance(old_entry[1], np.ndarray)
                        else old_entry[1].nbytes
                    )
                    self._memory_used -= old_packed_nbytes + old_scales_nbytes

                # Evict AFTER compression succeeds — avoids losing a valid
                # block if quantize_kv_4bit raises (Bug: premature eviction).
                if len(self._store) >= self.config.max_blocks and block_hash not in self._store:
                    self._evict_unlocked(1)

                # Store num_tokens alongside packed data so SSD flush can
                # propagate it accurately instead of always writing 0.
                self._store[block_hash] = (packed, scales, head_dim, num_tokens)
                # Move to end (most recently used)
                self._store.move_to_end(block_hash)

                self._memory_used += new_entry_bytes

                return True
            except Exception:
                logger.warning("Failed to demote block 0x%x to warm tier", block_hash, exc_info=True)
                return False

    def promote(self, block_hash: int) -> bytes | None:
        """Retrieve and decompress a KV block for promotion back to hot tier.

        Uses a peek-then-pop pattern: first dequantize the entry in-place,
        then pop on success.  If dequantization fails, the entry is left
        in the store so a concurrent demode for the same hash cannot cause
        silent data loss.

        Args:
            block_hash: Hash identifying the block.

        Returns:
            Dequantized data (MLX array or numpy array), or None if not found.
        """
        with self._lock:
            # Peek: read without removing.
            if block_hash not in self._store:
                self._misses += 1
                return None

            entry = self._store[block_hash]
            self._hits += 1
            # Handle both old 3-tuple (packed, scales, head_dim) and new
            # 4-tuple (packed, scales, head_dim, num_tokens) formats.
            if len(entry) >= 4:
                packed, scales, head_dim, _num_tokens = entry[0], entry[1], entry[2], entry[3]
            elif len(entry) == 3:
                packed, scales, head_dim = entry[0], entry[1], entry[2]
            else:
                packed, scales = entry[0], entry[1]
                head_dim = 0

            # Try dequantization first — only pop on success.
            try:
                from .compression import dequantize_kv_4bit

                result = dequantize_kv_4bit(packed, scales, head_dim=head_dim)
            except Exception:
                logger.warning(
                    "Failed to promote block 0x%x from warm tier — "
                    "leaving entry in store for retry",
                    block_hash, exc_info=True,
                )
                return None

            # Dequantization succeeded — safe to pop.
            self._store.pop(block_hash, None)

            # Adjust memory accounting
            try:
                packed_nbytes = (
                    np.array(packed).nbytes if not isinstance(packed, np.ndarray) else packed.nbytes
                )
                scales_nbytes = (
                    np.array(scales).nbytes if not isinstance(scales, np.ndarray) else scales.nbytes
                )
                self._memory_used -= packed_nbytes + scales_nbytes
                self._memory_used = max(0, self._memory_used)
            except Exception:
                logger.debug("memory accounting adjustment in promote failed", exc_info=True)

            return result

    def contains(self, block_hash: int) -> bool:
        """Check whether a block is present in the warm tier."""
        with self._lock:
            return block_hash in self._store

    def remove(self, block_hash: int) -> bool:
        """Remove a specific entry by hash. Thread-safe."""
        with self._lock:
            entry = self._store.pop(block_hash, None)
            if entry is not None:
                packed, scales = entry[0], entry[1]
                packed_nbytes = (
                    np.array(packed).nbytes if not isinstance(packed, np.ndarray) else packed.nbytes
                )
                scales_nbytes = (
                    np.array(scales).nbytes if not isinstance(scales, np.ndarray) else scales.nbytes
                )
                self._memory_used -= packed_nbytes + scales_nbytes
                self._memory_used = max(0, self._memory_used)
                return True
            return False

    def evict(self, count: int) -> int:
        """Evict the oldest (least recently used) blocks.

        Args:
            count: Maximum number of blocks to evict.

        Returns:
            Actual number of blocks evicted.
        """
        with self._lock:
            return self._evict_unlocked(count)

    def _evict_unlocked(self, count: int) -> int:
        """Evict without acquiring the lock (caller must hold it)."""
        evicted = 0
        for _ in range(min(count, len(self._store))):
            _block_hash, entry = self._store.popitem(last=False)  # FIFO = LRU
            packed, scales = entry[0], entry[1]
            try:
                packed_nbytes = (
                    np.array(packed).nbytes
                    if not isinstance(packed, np.ndarray)
                    else packed.nbytes
                )
                scales_nbytes = (
                    np.array(scales).nbytes
                    if not isinstance(scales, np.ndarray)
                    else scales.nbytes
                )
                self._memory_used -= packed_nbytes + scales_nbytes
                self._memory_used = max(0, self._memory_used)
            except Exception:
                logger.debug("memory accounting adjustment in evict failed", exc_info=True)
            evicted += 1
        return evicted

    def get_stats(self) -> dict:
        """Return warm tier statistics."""
        with self._lock:
            total_lookups = self._hits + self._misses
            hit_rate = (self._hits / total_lookups) if total_lookups > 0 else 0.0
            return {
                "num_blocks": len(self._store),
                "max_blocks": self.config.max_blocks,
                "memory_used_bytes": self._memory_used,
                "memory_used_mb": round(self._memory_used / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "utilization_pct": round(
                    len(self._store) / self.config.max_blocks * 100, 1
                ) if self.config.max_blocks > 0 else 0.0,
            }

    def flush(self) -> None:
        """Stub for SSD persistence.

        In a future phase, this will flush all warm-tier blocks to the
        storage_path on SSD so they survive process restarts.
        """
        self._last_flush = time.monotonic()
        if self.config.storage_path:
            with self._lock:
                num = len(self._store)
            logger.debug(
                "Warm tier flush: %d blocks (SSD stub, not yet implemented)",
                num,
            )
