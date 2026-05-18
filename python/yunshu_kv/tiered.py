from __future__ import annotations
"""Yunshu Tiered KV Cache Manager — Hot/Cold tier coordination.

Based on oMLX's TieredCacheManager pattern:
- Hot tier: UMA-resident FP16 KV cache (fastest, limited by RAM)
- Warm tier: Quantized 4-bit in UMA (2x more blocks, slightly slower)
- Cold tier: SSD-backed KV cache (unlimited, slowest, for long context)

In Apple Silicon's UMA architecture, all tiers share the same physical memory.
The "hot" vs "warm" distinction is about quantization level, not memory location.
The SSD tier stores to NVMe for truly unlimited context.

oMLX only supports SSD-only mode in production. Yunshu adds the hot/warm
distinction for performance optimization on Apple Silicon.
"""


import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .block import KVBlock
from .block_table import BlockTable
from .hash import compute_block_hash
from .manager import KVCacheManager, PrefixMatch
from .warm_tier import KVWarmTier

logger = logging.getLogger(__name__)


# ── SSD Cache Block Storage ──


@dataclass
class SSDCacheEntry:
    """Metadata for a cached block stored on SSD."""
    block_hash: int
    block_index: int
    num_tokens: int
    size_bytes: int
    last_access: float = 0.0
    ref_count: int = 0


class SSDCacheStore:
    """SSD-backed KV cache block storage.

    Stores serialized KV cache blocks to disk for cold tier.
    Follows oMLX's PagedSSDCacheManager pattern:
    - One directory per model
    - Block data stored as numpy arrays
    - Metadata in index.json for fast lookup
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_size_bytes: int = 100 * 1024 ** 3,  # 100GB default
        block_size: int = 64,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self.block_size = block_size

        # Block index: hash -> SSDCacheEntry
        self._index: dict[int, SSDCacheEntry] = {}
        self._current_size_bytes: int = 0
        self._next_block_index: int = 0

        # Load existing index
        self._load_index()

    def _index_path(self) -> Path:
        return self.cache_dir / "ssd_index.json"

    def _load_index(self) -> None:
        """Load cache index from disk."""
        path = self._index_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for entry_data in data.get("entries", []):
                entry = SSDCacheEntry(
                    block_hash=entry_data["block_hash"],
                    block_index=entry_data["block_index"],
                    num_tokens=entry_data["num_tokens"],
                    size_bytes=entry_data["size_bytes"],
                    last_access=entry_data.get("last_access", 0.0),
                    ref_count=entry_data.get("ref_count", 0),
                )
                self._index[entry.block_hash] = entry
                self._current_size_bytes += entry.size_bytes
                self._next_block_index = max(
                    self._next_block_index, entry.block_index + 1
                )
            logger.info(
                f"SSD cache: loaded {len(self._index)} entries, "
                f"{self._current_size_bytes / 1024**3:.1f} GB"
            )
        except Exception as e:
            logger.warning(f"Failed to load SSD cache index: {e}")

    def _save_index(self) -> None:
        """Persist cache index to disk."""
        entries = [
            {
                "block_hash": e.block_hash,
                "block_index": e.block_index,
                "num_tokens": e.num_tokens,
                "size_bytes": e.size_bytes,
                "last_access": e.last_access,
                "ref_count": e.ref_count,
            }
            for e in self._index.values()
        ]
        try:
            import tempfile
            index_path = self._index_path()
            fd, tmp_path = tempfile.mkstemp(
                dir=str(index_path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"entries": entries}, f)
                os.replace(tmp_path, str(index_path))
            except Exception:
                os.unlink(tmp_path) if os.path.exists(tmp_path) else None
                raise
        except Exception as e:
            logger.warning(f"Failed to save SSD cache index: {e}")

    def _block_path(self, block_index: int) -> Path:
        return self.cache_dir / f"block_{block_index:08d}.bin"

    def store(self, block_hash: int, kv_data: mx.array, num_tokens: int) -> bool:
        """Store a KV block to SSD. Returns True if stored successfully."""
        if self._current_size_bytes >= self.max_size_bytes:
            self._evict_lru()

        if block_hash in self._index:
            # Already stored — just update access time
            self._index[block_hash].last_access = time.monotonic()
            return True

        # Serialize KV data to bytes with shape metadata
        try:
            import numpy as np
            data_mx = mx.array(kv_data).astype(mx.float16)
            numpy_data = np.array(data_mx)
            shape = numpy_data.shape
            raw_bytes = numpy_data.tobytes()
        except Exception as e:
            logger.warning(f"Failed to serialize KV block: {e}")
            return False

        block_index = self._next_block_index
        self._next_block_index += 1

        try:
            import zlib
            crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
            with open(self._block_path(block_index), "wb") as f:
                # Header: ndim (4 bytes) + shape values (4 bytes each) + CRC32
                f.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<I", dim))
                f.write(struct.pack("<I", crc))
                f.write(raw_bytes)
        except OSError as e:
            logger.warning(f"Failed to write SSD cache block: {e}")
            return False

        header_size = 4 + 4 * len(shape) + 4  # ndim + shape dims + CRC32
        entry = SSDCacheEntry(
            block_hash=block_hash,
            block_index=block_index,
            num_tokens=num_tokens,
            size_bytes=header_size + len(raw_bytes),
            last_access=time.monotonic(),
        )
        self._index[block_hash] = entry
        self._current_size_bytes += entry.size_bytes
        return True

    def load(self, block_hash: int) -> Optional[mx.array]:
        """Load a KV block from SSD. Returns None if not found."""
        entry = self._index.get(block_hash)
        if entry is None:
            return None

        path = self._block_path(entry.block_index)
        if not path.exists():
            self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
            del self._index[block_hash]
            return None

        try:
            with open(path, "rb") as f:
                ndim = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
                stored_crc = struct.unpack("<I", f.read(4))[0]
                raw_bytes = f.read()
            import zlib
            actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
            if stored_crc != actual_crc:
                logger.warning(
                    "CRC mismatch for SSD block 0x%x (stored=%08x actual=%08x), discarding",
                    block_hash, stored_crc, actual_crc,
                )
                self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                del self._index[block_hash]
                return None
            import numpy as np
            numpy_data = np.frombuffer(raw_bytes, dtype=np.float16).copy().reshape(shape)
            return mx.array(numpy_data)
        except Exception as e:
            logger.warning(f"Failed to load SSD cache block: {e}")
            return None

    def contains(self, block_hash: int) -> bool:
        return block_hash in self._index

    def _evict_lru(self) -> None:
        """Evict least recently used blocks to free space."""
        if not self._index:
            return

        # Sort by access time, evict oldest 10%. Skip blocks with active refs.
        sorted_entries = sorted(
            [e for e in self._index.values() if e.ref_count == 0],
            key=lambda e: e.last_access,
        )
        to_evict = max(1, len(sorted_entries) // 10)

        for i in range(to_evict):
            entry = sorted_entries[i]
            path = self._block_path(entry.block_index)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            del self._index[entry.block_hash]
            self._current_size_bytes -= entry.size_bytes

        self._current_size_bytes = max(0, self._current_size_bytes)
        self._save_index()

    def get_stats(self) -> dict:
        return {
            "num_entries": len(self._index),
            "size_bytes": self._current_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "size_gb": round(self._current_size_bytes / 1024**3, 2),
            "max_size_gb": round(self.max_size_bytes / 1024**3, 2),
            "utilization_pct": (
                self._current_size_bytes / self.max_size_bytes * 100
                if self.max_size_bytes > 0 else 0
            ),
        }


# ── Tiered Cache Manager ──


class TieredKVCacheManager:
    """Manages hot/warm/cold KV cache tiers (oMLX pattern).

    Coordination strategy:
    - New requests: check hot (UMA) cache first → warm tier → SSD cache
    - Hot cache miss but warm hit: promote from warm to hot
    - Hot cache miss but SSD hit: load from SSD into hot cache
    - Hot cache pressure: evict LRU blocks to warm tier, then SSD
    - Request completion: free hot blocks, persist to SSD if caching enabled

    This follows oMLX's TieredCacheManager but adds the warm tier
    (quantized) for Apple Silicon optimization.
    """

    def __init__(
        self,
        hot_manager: KVCacheManager,
        ssd_store: SSDCacheStore | None = None,
        warm_tier: KVWarmTier | None = None,
    ):
        self.hot = hot_manager
        self.ssd = ssd_store
        # Warm tier: 4-bit quantized in-memory cache (uses KVWarmTier)
        self.warm = warm_tier  # None means no warm tier

    def allocate_for_prefill(
        self,
        token_ids: list[int],
        model_hash: int = 0,
    ) -> tuple[BlockTable, PrefixMatch]:
        """Allocate blocks with hot + warm + SSD tier lookup (oMLX pattern).

        1. Check hot cache (UMA-resident FP16)
        2. For misses, check warm tier (4-bit quantized)
        3. For remaining misses, check SSD cache
        4. Allocate new blocks for remaining tokens
        """
        table, match = self.hot.allocate_for_prefill(token_ids, model_hash)

        # If there are unmatched tokens, check warm tier and SSD cache
        if not match.unmatched_token_ids:
            return table, match

        block_size = self.hot.block_size
        remaining = match.unmatched_token_ids

        # Initialize parent hash from hot tier's last matched block.
        # Must be computed before the warm/SSD branches so it's always defined.
        parent_hash = None
        if match.matched_blocks:
            parent_hash = match.matched_blocks[-1].block_hash

        # Check warm tier for each block-sized chunk.
        # Chain hashing: each block's hash depends on the parent hash,
        # so we must continue the chain from the hot tier's last matched
        # block (or None if there was no match).
        warm_promoted_blocks: list[KVBlock] = []
        warm_loaded = 0
        if self.warm:
            for i in range(0, len(remaining), block_size):
                chunk = remaining[i:i + block_size]
                if len(chunk) < block_size:
                    break
                h = compute_block_hash(parent_hash, chunk, (model_hash,))
                if self.warm.contains(h):
                    # Promote from warm tier back to hot
                    kv_data = self.warm.promote(h)
                    if kv_data is not None:
                        # Allocate a hot block for the promoted data
                        try:
                            new_block = self.hot.block_pool.allocate(1)[0]
                        except ValueError:
                            logger.warning(
                                "Warm tier promotion: no free blocks for hash 0x%x",
                                h,
                            )
                            # promote() already popped from the warm store.
                            # Re-insert to avoid data loss.
                            self.warm.demote(h, kv_data)
                            break
                        # Write KV data into hot cache tensors.
                        # Packed format: [2, num_heads, block_size, head_dim]
                        # where dim 0 has key at [0] and value at [1].
                        if self.hot._key_cache is not None:
                            if kv_data.ndim == 4 and kv_data.shape[0] == 2:
                                self.hot._key_cache[new_block.block_id] = kv_data[0]
                                if self.hot._value_cache is not None:
                                    self.hot._value_cache[new_block.block_id] = kv_data[1]
                            else:
                                self.hot._key_cache[new_block.block_id] = kv_data
                        # Register in prefix cache for future lookups
                        self.hot.block_pool.cache_block(new_block, h)
                        warm_promoted_blocks.append(new_block)
                        warm_loaded += 1
                        parent_hash = h  # chain: next block uses this as parent
                    else:
                        break
                else:
                    break

            if warm_loaded > 0:
                match.num_matched_tokens += warm_loaded * block_size
                remaining = remaining[warm_loaded * block_size:]
                match.unmatched_token_ids = remaining
                logger.debug(
                    f"Warm tier hit: {warm_loaded} blocks "
                    f"({warm_loaded * block_size} tokens)"
                )

        # If still unmatched tokens and we have SSD cache, check SSD
        ssd_promoted_blocks: list[KVBlock] = []
        if self.ssd and remaining:
            ssd_loaded = 0
            # parent_hash is already set: if warm promotion happened it points
            # to the last warm-promoted block; otherwise to the hot tier's
            # last matched block (or None).
            for i in range(0, len(remaining), block_size):
                chunk = remaining[i:i + block_size]
                if len(chunk) < block_size:
                    break
                h = compute_block_hash(parent_hash, chunk, (model_hash,))
                if self.ssd.contains(h):
                    # Load from SSD into hot cache
                    kv_data = self.ssd.load(h)
                    if kv_data is not None:
                        # Allocate a hot block and write kv_data back into it
                        try:
                            new_block = self.hot.block_pool.allocate(1)[0]
                        except ValueError:
                            logger.warning(
                                "SSD cache promotion: no free blocks for hash 0x%x",
                                h,
                            )
                            break
                        # Write KV data into hot cache tensors.
                        # Packed format: [2, num_heads, block_size, head_dim]
                        if self.hot._key_cache is not None:
                            if kv_data.ndim == 4 and kv_data.shape[0] == 2:
                                self.hot._key_cache[new_block.block_id] = kv_data[0]
                                if self.hot._value_cache is not None:
                                    self.hot._value_cache[new_block.block_id] = kv_data[1]
                            else:
                                self.hot._key_cache[new_block.block_id] = kv_data
                        # Register in prefix cache for future lookups
                        self.hot.block_pool.cache_block(new_block, h)
                        ssd_promoted_blocks.append(new_block)
                        ssd_loaded += 1
                        parent_hash = h  # chain: next block uses this as parent
                    else:
                        break
                else:
                    break

            if ssd_loaded > 0:
                match.num_matched_tokens += ssd_loaded * block_size
                match.unmatched_token_ids = remaining[ssd_loaded * block_size:]
                logger.debug(
                    f"SSD cache hit: {ssd_loaded} blocks "
                    f"({ssd_loaded * block_size} tokens)"
                )
        elif not remaining:
            match.unmatched_token_ids = []

        # Insert warm/SSD promoted blocks into the BlockTable.  They
        # logically sit between the hot-tier matched blocks (already in
        # the table) and the newly allocated blocks (also already in the
        # table from hot.allocate_for_prefill).  We splice them in at
        # the boundary between matched and new blocks.
        all_promoted = warm_promoted_blocks + ssd_promoted_blocks
        if all_promoted:
            hot_matched_count = len(match.matched_blocks)
            existing = table._blocks
            # Promoted blocks replace an equal number of new blocks that
            # were allocated for the same tokens.  Free the surplus.
            surplus = existing[hot_matched_count:hot_matched_count + len(all_promoted)]
            for block in surplus:
                block.ref_count = 1
            self.hot.block_pool.free(surplus)
            # Rebuild the internal block list with promoted blocks spliced in.
            table._blocks = (
                existing[:hot_matched_count]
                + all_promoted
                + existing[hot_matched_count + len(all_promoted):]
            )
            table.total_tokens = len(table._blocks) * block_size
            match.matched_blocks = match.matched_blocks + all_promoted

        return table, match

    def _extract_kv_for_block(self, block: KVBlock) -> Optional[bytes]:
        """Extract KV tensor data for a block from the hot cache.

        Serializes the key and value tensor slices for the given block_id.
        Returns None if the cache tensors aren't allocated or block_id is out of range.
        """
        try:
            key_cache = getattr(self.hot, '_key_cache', None)
            val_cache = getattr(self.hot, '_value_cache', None)
            if key_cache is None or val_cache is None:
                return None
            bid = block.block_id
            if bid >= key_cache.shape[0]:
                return None
            k_slice = key_cache[bid]
            v_slice = val_cache[bid]
            import numpy as np
            k_np = np.array(k_slice) if not isinstance(k_slice, np.ndarray) else k_slice
            v_np = np.array(v_slice) if not isinstance(v_slice, np.ndarray) else v_slice
            return k_np.tobytes() + v_np.tobytes()
        except Exception:
            logger.debug("KV block extraction from hot cache failed", exc_info=True)
            return None

    def free_request(self, table: BlockTable, request_id: str | None = None) -> None:
        """Free blocks from a completed request."""
        self.hot.free_request(table, request_id=request_id)

    def get_stats(self) -> dict:
        """Return combined stats from all tiers."""
        stats = {
            "hot_usage_pct": round(self.hot.usage * 100, 1),
            "hot_free_blocks": self.hot.num_free_blocks,
            "hot_block_size": self.hot.block_size,
        }
        if self.warm:
            stats["warm"] = self.warm.get_stats()
        if self.ssd:
            stats["ssd"] = self.ssd.get_stats()
        return stats


# ── Background SSD Flush Thread ──


import threading


class BackgroundSSDFlush:
    """Background thread for periodic async SSD writes.

    Periodically flushes warm-tier blocks to the SSD cold tier so that
    KV data survives process restarts. Runs as a daemon thread that
    wakes up at a configurable interval and writes any new blocks that
    haven't been persisted yet.
    """

    def __init__(
        self,
        ssd_store: SSDCacheStore,
        warm_tier: KVWarmTier,
        flush_interval_s: float = 60.0,
    ) -> None:
        self._ssd = ssd_store
        self._warm = warm_tier
        self._flush_interval = flush_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._flush_count: int = 0
        self._last_flush_time: float = 0.0

    def start(self) -> None:
        """Start the background flush thread (daemon)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="yunshu-ssd-flush",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Background SSD flush thread started (interval=%.0fs)",
            self._flush_interval,
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def flush_now(self) -> int:
        """Perform one flush cycle: write all warm blocks to SSD.

        Returns the number of blocks flushed.
        """
        flushed = 0
        # Access the warm tier's internal store
        for block_hash, entry in list(self._warm._store.items()):
            packed, scales = entry[0], entry[1]
            head_dim = entry[2] if len(entry) > 2 else 0
            if head_dim == 0:
                continue
            if not self._ssd.contains(block_hash):
                try:
                    import numpy as np
                    # Reconstruct full data from packed+scales for SSD storage
                    from .compression import dequantize_kv_4bit
                    kv_data = dequantize_kv_4bit(packed, scales, head_dim=head_dim)
                    self._ssd.store(block_hash, mx.array(kv_data), num_tokens=0)
                    flushed += 1
                except Exception as e:
                    logger.debug(
                        "Failed to flush block 0x%x to SSD: %s",
                        block_hash, e,
                    )
        self._flush_count += 1
        self._last_flush_time = time.monotonic()
        return flushed

    def _run(self) -> None:
        """Main loop for the background flush thread."""
        while not self._stop_event.wait(timeout=self._flush_interval):
            try:
                flushed = self.flush_now()
                if flushed > 0:
                    logger.debug("SSD flush: %d blocks persisted", flushed)
            except Exception as e:
                logger.warning("Background SSD flush error: %s", e)

    def get_stats(self) -> dict:
        """Return flush thread statistics."""
        return {
            "flush_count": self._flush_count,
            "last_flush_time": self._last_flush_time,
            "running": self._thread is not None and self._thread.is_alive(),
        }
