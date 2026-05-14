"""Yunshu KV Warm Tier — in-memory 4-bit quantized KV block cache.

Sits between the hot tier (FP16 UMA) and cold tier (SSD). When the hot
BlockPool is under memory pressure, KV blocks are demoted here with
4-bit quantization (4x compression). On cache hit, they are promoted
back to FP16 transparently.

Uses the existing compression module (quantize_kv_4bit / dequantize_kv_4bit)
for the actual quantization path. Metal kernel will replace the numpy path
in Phase 2.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

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

    # ── Properties ────────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        return len(self._store)

    @property
    def is_full(self) -> bool:
        return len(self._store) >= self.config.max_blocks

    # ── Core API ──────────────────────────────────────────────────

    def demote(self, block_hash: int, kv_data) -> bool:
        """Accept a KV block from the hot tier, compress and store it.

        Args:
            block_hash: Hash identifying the block content.
            kv_data: FP16 KV data (MLX array or numpy array).

        Returns:
            True if the block was stored successfully.
        """
        if self.is_full:
            self.evict(1)

        try:
            from .compression import quantize_kv_4bit

            packed, scales = quantize_kv_4bit(kv_data)
            self._store[block_hash] = (packed, scales)
            # Move to end (most recently used)
            self._store.move_to_end(block_hash)

            # Approximate memory accounting
            packed_nbytes = (
                np.array(packed).nbytes if not isinstance(packed, np.ndarray) else packed.nbytes
            )
            scales_nbytes = (
                np.array(scales).nbytes if not isinstance(scales, np.ndarray) else scales.nbytes
            )
            self._memory_used += packed_nbytes + scales_nbytes

            return True
        except Exception:
            logger.warning("Failed to demote block 0x%x to warm tier", block_hash, exc_info=True)
            return False

    def promote(self, block_hash: int) -> Optional[bytes]:
        """Retrieve and decompress a KV block for promotion back to hot tier.

        Args:
            block_hash: Hash identifying the block.

        Returns:
            Dequantized data (MLX array or numpy array), or None if not found.
        """
        entry = self._store.pop(block_hash, None)
        if entry is None:
            self._misses += 1
            return None

        self._hits += 1
        packed, scales = entry

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

        try:
            from .compression import dequantize_kv_4bit

            return dequantize_kv_4bit(packed, scales)
        except Exception:
            logger.warning("Failed to promote block 0x%x from warm tier", block_hash, exc_info=True)
            return None

    def contains(self, block_hash: int) -> bool:
        """Check whether a block is present in the warm tier."""
        return block_hash in self._store

    def evict(self, count: int) -> int:
        """Evict the oldest (least recently used) blocks.

        Args:
            count: Maximum number of blocks to evict.

        Returns:
            Actual number of blocks evicted.
        """
        evicted = 0
        for _ in range(min(count, len(self._store))):
            _block_hash, entry = self._store.popitem(last=False)  # FIFO = LRU
            packed, scales = entry
            # Adjust memory accounting
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
            logger.debug(
                "Warm tier flush: %d blocks (SSD stub, not yet implemented)",
                len(self._store),
            )
