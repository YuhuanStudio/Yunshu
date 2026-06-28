from __future__ import annotations

"""Yunshu Tiered KV Cache Manager — Hot/Cold tier coordination.

Tier coordination:
- Hot tier: UMA-resident FP16 KV cache (fastest, limited by RAM)
- Warm tier: Quantized 4-bit in UMA (2x more blocks, slightly slower)
- Cold tier: SSD-backed KV cache (unlimited, slowest, for long context)

In Apple Silicon's UMA architecture, all tiers share the same physical memory.
The "hot" vs "warm" distinction is about quantization level, not memory location.
The SSD tier stores to NVMe for truly unlimited context.

Yunshu adds the hot/warm distinction for performance optimization on
Apple Silicon.
"""


import contextlib
import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
    Layout:
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
        # Lock protects index lookups + file reads/writes from concurrent
        # eviction.  Without it, _evict_lru() can unlink a block file
        # between load()'s index lookup and file read (TOCTOU race).
        self._lock = threading.Lock()

        # Deferred I/O from _evict_lru_locked() / _store_new_block():
        # populated under _lock, flushed by _flush_pending_io() after
        # lock release.
        self._pending_index_write: tuple[Path, list[dict]] | None = None
        self._pending_unlinks: list[Path] | None = None
        self._pending_block_writes: list[tuple[Path, bytes, int, int, int]] | None = None
        # ^ Each tuple: (path, raw_bytes, block_hash, block_index, header_size)
        # block_index is included so _flush_pending_io can skip the write
        # if the entry was already evicted before the write happened.

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
            # Use monotonic time for all loaded entries so LRU eviction
            # works correctly with new entries that also use monotonic time.
            # Entries loaded from disk are equally "old" from LRU perspective,
            # so they all get the same baseline timestamp (now minus 1 second
            # so they're evicted before freshly loaded/accessed entries).
            load_time = time.monotonic()
            for i, entry_data in enumerate(data.get("entries", [])):
                entry = SSDCacheEntry(
                    block_hash=entry_data["block_hash"],
                    block_index=entry_data["block_index"],
                    num_tokens=entry_data["num_tokens"],
                    size_bytes=entry_data["size_bytes"],
                    # Use monotonic baseline — old wall-clock values from disk
                    # would mix badly with monotonic timestamps from new code.
                    last_access=load_time - (len(data.get("entries", [])) - i) * 0.001,
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

    def _snapshot_index(self) -> list[dict]:
        """Snapshot current index entries for async persistence.

        Returns a serializable list of entry dicts.  Caller must hold
        ``_lock`` to ensure a consistent snapshot.
        """
        return [
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

    @staticmethod
    def _write_index_to_disk(index_path: Path, entries: list[dict]) -> None:
        """Write *entries* to *index_path* atomically via temp-file.

        This performs synchronous file I/O (json.dump + os.replace) and
        must **not** be called while holding ``_lock`` — the caller is
        responsible for releasing the lock first.
        """
        try:
            import tempfile
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

    def _save_index(self) -> None:
        """Persist cache index to disk (I/O outside lock).

        Takes a snapshot of the index under ``_lock``, then releases the
        lock before performing file I/O so that concurrent store / load /
        contains operations are not blocked by disk writes.
        """
        index_path = self._index_path()
        with self._lock:
            entries = self._snapshot_index()
        self._write_index_to_disk(index_path, entries)

    def _block_path(self, block_index: int) -> Path:
        return self.cache_dir / f"block_{block_index:08d}.bin"

    def store(self, block_hash: int, kv_data: mx.array, num_tokens: int) -> bool:
        """Store a KV block to SSD. Returns True if stored successfully."""
        with self._lock:
            if self._current_size_bytes >= self.max_size_bytes:
                self._evict_lru_locked()

            if block_hash in self._index:
                # Already stored — just update access time
                self._index[block_hash].last_access = time.monotonic()
                result = True
            else:
                # Serialize KV data to bytes with shape metadata
                result = self._store_new_block(block_hash, kv_data, num_tokens)
        # File I/O from any eviction deferred until after lock release.
        self._flush_pending_io()
        return result

    def _store_new_block(
        self, block_hash: int, kv_data: mx.array, num_tokens: int
    ) -> bool:
        """Prepare a brand-new block for SSD storage.

        Serializes KV data to bytes under the lock, but defers the
        actual file I/O to ``_flush_pending_io()`` which runs after
        the lock is released.  The index is tentatively updated so
        that concurrent lookups see the entry; the file will exist
        by the time any concurrent ``load()`` acquires the lock next.

        Uses 8-bit quantization (2x compression vs FP16) instead of raw
        FP16 to reduce SSD I/O bandwidth and disk usage. The warm tier's
        4-bit quantization is not reused here because it requires paired
        (packed, scales) storage which would complicate the simple binary
        format; 8-bit provides a good balance of compression ratio and
        implementation simplicity.

        Caller must hold ``_lock``.
        """
        if self._current_size_bytes >= self.max_size_bytes:
            return False  # Cannot free enough space

        # Step 1: Quantize to 8-bit and serialize (CPU-bound, fast).
        # 8-bit symmetric per-tensor: q = round(x / scale), stored as uint8
        # with scale = max(|x|) / 127. Compression: 2x vs FP16.
        try:
            import numpy as np
            # Cast to float32, NOT float16, before int8-quantizing. Default KV dtype
            # is bf16 (max ~3.4e38); any element with |x| > 65504 (attention sinks / outlier
            # channels routinely exceed fp16 max) became inf on the fp16 cast → scale =
            # max(inf)/127 = inf → round(x/inf) = 0 → the whole block stored as garbage. The
            # sibling ssd_kv_cache.py already casts to float32; this cold tier was never
            # swept. The int8 quant makes the wider intermediate free.
            data_mx = mx.array(kv_data).astype(mx.float32)
            numpy_data = np.array(data_mx)
            original_shape = numpy_data.shape
            # PER-SLICE (axis-0) int8 scales. The block is [2, …] (K=[0], V=[1])
            # and attention K/V magnitudes routinely differ 50-200×; a single shared scale
            # was dominated by the larger tensor and crushed the smaller one (~79% relative
            # error on the restored V tensor, measured). Quantize each axis-0 slice with its
            # own abs-max scale so both K and V round-trip cleanly.
            _n = original_shape[0] if numpy_data.ndim >= 1 and original_shape[0] > 0 else 1
            _flat = numpy_data.reshape(_n, -1)
            _scales: list[float] = []
            _qslices = []
            for _i in range(_n):
                _amax = float(np.max(np.abs(_flat[_i]))) if _flat[_i].size else 0.0
                if _amax == 0.0:
                    _amax = 1.0
                _sc = _amax / 127.0
                _scales.append(_sc)
                _qslices.append(np.clip(np.round(_flat[_i] / _sc), -127, 127).astype(np.int8))
            quantized = np.concatenate(_qslices).reshape(original_shape)
            raw_bytes = quantized.tobytes()
            shape = original_shape
        except Exception as e:
            logger.warning(f"Failed to serialize KV block: {e}")
            return False

        block_index = self._next_block_index
        self._next_block_index += 1

        # v3 multi-scale on-disk format. The leading MAGIC (an impossible ndim) unambiguously
        # distinguishes v3 from the legacy size-discriminated formats in load():
        # [MAGIC u32=0xFFFFFFFE][ndim u32][shape dims u32…][num_scales u32][scales f32…]
        # [block_hash u64][CRC32 u32][int8 data]
        # CRC32 covers the whole header + data. block_hash in the CRC prevents an index
        # mapping a wrong hash to a valid file.
        import zlib
        header_size_no_crc = 4 + 4 + 4 * len(shape) + 4 + 4 * len(_scales) + 8
        header_bytes = bytearray(header_size_no_crc)
        off = 0
        struct.pack_into("<I", header_bytes, off, 0xFFFFFFFE); off += 4
        struct.pack_into("<I", header_bytes, off, len(shape)); off += 4
        for dim in shape:
            struct.pack_into("<I", header_bytes, off, dim); off += 4
        struct.pack_into("<I", header_bytes, off, len(_scales)); off += 4
        for _sc in _scales:
            struct.pack_into("<f", header_bytes, off, _sc); off += 4
        struct.pack_into("<Q", header_bytes, off, block_hash & 0xFFFFFFFFFFFFFFFF); off += 8
        crc = zlib.crc32(bytes(header_bytes) + raw_bytes) & 0xFFFFFFFF
        header_size = header_size_no_crc + 4  # + CRC32
        file_payload = bytearray(header_size + len(raw_bytes))
        file_payload[:header_size_no_crc] = header_bytes
        struct.pack_into("<I", file_payload, header_size_no_crc, crc)
        file_payload[header_size:] = raw_bytes

        # Step 2: Tentatively update the in-memory index so concurrent
        # contains() / lookups see the entry immediately.
        entry = SSDCacheEntry(
            block_hash=block_hash,
            block_index=block_index,
            num_tokens=num_tokens,
            size_bytes=header_size + len(raw_bytes),
            last_access=time.monotonic(),
        )
        self._index[block_hash] = entry
        self._current_size_bytes += entry.size_bytes

        # Step 3: Defer the file write until after the lock is released.
        if self._pending_block_writes is None:
            self._pending_block_writes = []
        self._pending_block_writes.append((
            self._block_path(block_index),
            bytes(file_payload),
            block_hash,
            block_index,
            header_size,
        ))
        # Also snapshot the index for persistence after the write.
        self._pending_index_write = (self._index_path(), self._snapshot_index())
        return True

    def load(self, block_hash: int) -> mx.array | None:
        """Load a KV block from SSD. Returns None if not found.

        Holds _lock throughout index lookup + file read to prevent
        _evict_lru() from deleting the file between the two operations.
        """
        with self._lock:
            entry = self._index.get(block_hash)
            if entry is None:
                return None

            path = self._block_path(entry.block_index)
            if not path.exists():
                # Check if this block has a pending write — if so, the
                # index entry is valid but the file hasn't been flushed yet.
                # Do NOT delete the index entry in this case.
                if self._pending_block_writes is not None:
                    for _, _, phash, _, _ in self._pending_block_writes:
                        if phash == block_hash:
                            return None  # Pending write, don't delete index
                self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                del self._index[block_hash]
                return None

            try:
                with open(path, "rb") as f:
                    _first = struct.unpack("<I", f.read(4))[0]
                    if _first == 0xFFFFFFFE:
                        # ── v3 multi-scale format (per-axis-0 K/V int8 scales) ──
                        import zlib as _zl

                        import numpy as _np
                        ndim = struct.unpack("<I", f.read(4))[0]
                        shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
                        _num_scales = struct.unpack("<I", f.read(4))[0]
                        _scales = [struct.unpack("<f", f.read(4))[0] for _ in range(_num_scales)]
                        stored_block_hash = struct.unpack("<Q", f.read(8))[0]
                        stored_crc = struct.unpack("<I", f.read(4))[0]
                        raw_bytes = f.read()
                        _hdr = bytearray(4 + 4 + 4 * len(shape) + 4 + 4 * len(_scales) + 8)
                        _o = 0
                        struct.pack_into("<I", _hdr, _o, 0xFFFFFFFE); _o += 4
                        struct.pack_into("<I", _hdr, _o, len(shape)); _o += 4
                        for _d in shape:
                            struct.pack_into("<I", _hdr, _o, _d); _o += 4
                        struct.pack_into("<I", _hdr, _o, len(_scales)); _o += 4
                        for _s in _scales:
                            struct.pack_into("<f", _hdr, _o, _s); _o += 4
                        struct.pack_into("<Q", _hdr, _o, block_hash & 0xFFFFFFFFFFFFFFFF); _o += 8
                        _actual = _zl.crc32(bytes(_hdr) + raw_bytes) & 0xFFFFFFFF
                        if (stored_block_hash != (block_hash & 0xFFFFFFFFFFFFFFFF)
                                or stored_crc != _actual):
                            logger.warning(
                                "SSD v3 block 0x%x failed hash/CRC check, discarding", block_hash)
                            self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                            del self._index[block_hash]
                            return None
                        entry.last_access = time.monotonic()
                        _q = _np.frombuffer(raw_bytes, dtype=_np.int8).copy().reshape(shape)
                        _n = shape[0] if len(shape) >= 1 and shape[0] > 0 else 1
                        # Dequant in float32, NOT float16. The f32 scale applied in
                        # fp16 overflowed to inf when a slice's abs-max landed near fp16 max
                        # (127 * a scale that rounds up past 515.8 → >65504 → inf → NaN
                        # attention on restore). Same class fixed in ssd_kv_cache.py;
                        # un-swept here. Let MLX downcast the final array as needed.
                        _qf = _q.reshape(_n, -1).astype(_np.float32)
                        for _i in range(min(_n, len(_scales))):
                            _qf[_i] = _qf[_i] * _np.float32(_scales[_i])
                        return mx.array(_qf.reshape(shape))
                    # ── Legacy size-discriminated formats (v0/v1/v2) ──
                    ndim = _first
                    shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
                    # Detect format by file size:
                    # newest:  ndim + shape + scale + block_hash(u64) + CRC + int8 data
                    # prev:    ndim + shape + scale + CRC + int8 data (no block_hash)
                    # oldest:  ndim + shape + CRC + fp16 data
                    total_header_read = 4 + 4 * ndim
                    file_size = f.seek(0, 2)
                    num_elements = 1
                    for d in shape:
                        num_elements *= d
                    fp16_data_size = num_elements * 2
                    int8_data_size = num_elements * 1

                    # Compute expected total size for each format variant
                    newest_header_remain = 4 + 8 + 4  # scale + block_hash + CRC
                    newest_total = total_header_read + newest_header_remain + int8_data_size
                    prev_header_remain = 4 + 4  # scale + CRC
                    prev_total = total_header_read + prev_header_remain + int8_data_size
                    old_header_remain = 4  # CRC
                    old_total = total_header_read + old_header_remain + fp16_data_size

                    f.seek(total_header_read)
                    stored_block_hash: int | None = None
                    if newest_total == file_size:
                        # Newest format: scale + block_hash + CRC + int8
                        scale = struct.unpack("<f", f.read(4))[0]
                        stored_block_hash = struct.unpack("<Q", f.read(8))[0]
                        stored_crc = struct.unpack("<I", f.read(4))[0]
                        raw_bytes = f.read()
                    elif prev_total == file_size:
                        # Previous format: scale + CRC + int8 (no block_hash)
                        scale = struct.unpack("<f", f.read(4))[0]
                        stored_crc = struct.unpack("<I", f.read(4))[0]
                        raw_bytes = f.read()
                    elif old_total == file_size:
                        # Oldest format: CRC + fp16 (no scale, no block_hash)
                        stored_crc = struct.unpack("<I", f.read(4))[0]
                        raw_bytes = f.read()
                        scale = None  # Sentinel: raw FP16, no dequantization
                    else:
                        # Cannot determine format — default to newest
                        f.seek(total_header_read)
                        scale = struct.unpack("<f", f.read(4))[0]
                        stored_block_hash = struct.unpack("<Q", f.read(8))[0]
                        stored_crc = struct.unpack("<I", f.read(4))[0]
                        raw_bytes = f.read()

                import zlib
                # CRC covers header + data (not just data) to detect corruption
                if stored_block_hash is not None:
                    # Newest format: CRC includes ndim + shape + scale + block_hash
                    header_for_crc = bytearray(4 + 4 * len(shape) + 4 + 8)
                    off_h = 0
                    struct.pack_into("<I", header_for_crc, off_h, len(shape)); off_h += 4
                    for dim in shape:
                        struct.pack_into("<I", header_for_crc, off_h, dim); off_h += 4
                    struct.pack_into("<f", header_for_crc, off_h, scale); off_h += 4
                    struct.pack_into("<Q", header_for_crc, off_h, block_hash & 0xFFFFFFFFFFFFFFFF); off_h += 8
                    actual_crc = zlib.crc32(bytes(header_for_crc) + raw_bytes) & 0xFFFFFFFF
                elif scale is not None:
                    # Previous format: CRC includes ndim + shape + scale
                    header_for_crc = bytearray(4 + 4 * len(shape) + 4)
                    off_h = 0
                    struct.pack_into("<I", header_for_crc, off_h, len(shape)); off_h += 4
                    for dim in shape:
                        struct.pack_into("<I", header_for_crc, off_h, dim); off_h += 4
                    struct.pack_into("<f", header_for_crc, off_h, scale); off_h += 4
                    actual_crc = zlib.crc32(bytes(header_for_crc) + raw_bytes) & 0xFFFFFFFF
                else:
                    actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
                if stored_block_hash is not None and stored_block_hash != (block_hash & 0xFFFFFFFFFFFFFFFF):
                    # block_hash mismatch means index maps wrong hash to this file
                    logger.warning(
                        "block_hash mismatch for SSD block (index=0x%x stored=0x%x), discarding",
                        block_hash, stored_block_hash,
                    )
                    self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                    del self._index[block_hash]
                    return None
                if stored_crc != actual_crc:
                    logger.warning(
                        "CRC mismatch for SSD block 0x%x (stored=%08x actual=%08x), discarding",
                        block_hash, stored_crc, actual_crc,
                    )
                    self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                    del self._index[block_hash]
                    return None
                # Update last_access on successful read so LRU eviction
                # correctly tracks recency.  Without this, frequently-loaded
                # blocks appear "stale" and get prematurely evicted.
                entry.last_access = time.monotonic()
                import numpy as np
                if scale is not None:
                    # New 8-bit quantized format: dequantize int8 → FP16
                    quantized = np.frombuffer(raw_bytes, dtype=np.int8).copy().reshape(shape)
                    numpy_data = (quantized.astype(np.float16) * np.float16(scale))
                else:
                    # Old raw FP16 format (backward compat)
                    numpy_data = np.frombuffer(raw_bytes, dtype=np.float16).copy().reshape(shape)
                return mx.array(numpy_data)
            except FileNotFoundError:
                # Race: file was deleted after existence check (e.g. external cleanup)
                logger.debug("SSD block file vanished for hash 0x%x", block_hash)
                self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                self._index.pop(block_hash, None)
                return None
            except (ValueError, OSError, EOFError, KeyError) as e:
                # Was broad `except Exception` which masked
                # programmer bugs (AttributeError, TypeError) as "corrupt
                # block" and silently dropped index entries → SSD tier
                # quietly leaked all blocks. Now narrow to actual I/O /
                # deserialization errors that legitimately indicate corruption.
                logger.warning(
                    "Corrupt SSD cache block 0x%x, removing from index: %s",
                    block_hash, e,
                )
                self._current_size_bytes = max(0, self._current_size_bytes - entry.size_bytes)
                self._index.pop(block_hash, None)
                return None

    def contains(self, block_hash: int) -> bool:
        with self._lock:
            entry = self._index.get(block_hash)
            if entry is None:
                return False
            return self._block_path(entry.block_index).exists()

    def _evict_lru(self) -> None:
        """Evict least recently used blocks to free space. Acquires _lock."""
        with self._lock:
            self._evict_lru_locked()
        # File I/O (index persist + block file deletion) outside the lock.
        self._flush_pending_io()

    def _evict_lru_locked(self) -> None:
        """Evict least recently used blocks to free space. Caller must hold _lock.

        Updates the in-memory index under the lock, snapshots it, then
        performs file I/O (index persist + block file deletion) after
        releasing the lock so that concurrent operations are not blocked.
        """
        if not self._index:
            return

        # Sort by access time, evict oldest 10%. Skip blocks with active refs.
        # Also skip blocks that have pending writes — evicting those would
        # create orphan files and permanent index inconsistency.
        pending_hashes = set()
        if self._pending_block_writes:
            for _, _, phash, _, _ in self._pending_block_writes:
                pending_hashes.add(phash)
        sorted_entries = sorted(
            [e for e in self._index.values()
             if e.ref_count == 0 and e.block_hash not in pending_hashes],
            key=lambda e: e.last_access,
        )
        if not sorted_entries:
            # All entries have active refs — nothing eligible for eviction.
            return
        to_evict = max(1, len(sorted_entries) // 10)

        # Collect entries to evict and update the in-memory index first.
        # If the process crashes between index save and file deletion,
        # the orphaned files are harmless.  If we deleted files first and
        # crashed before saving the index, the index would reference
        # deleted blocks — a crash-consistency corruption.
        paths_to_unlink: list[Path] = []
        for i in range(to_evict):
            entry = sorted_entries[i]
            paths_to_unlink.append(self._block_path(entry.block_index))
            del self._index[entry.block_hash]
            self._current_size_bytes -= entry.size_bytes

        self._current_size_bytes = max(0, self._current_size_bytes)
        # Snapshot the index while we still hold the lock, but defer I/O.
        index_path = self._index_path()
        entries_snapshot = self._snapshot_index()

        # --- Lock released by caller after return ---
        # The caller (_evict_lru or store) holds _lock around this method.
        # We cannot release _lock inside this method because the caller
        # acquired it.  Instead, we write the snapshot *after* the caller
        # releases _lock by recording what needs to happen.
        self._pending_index_write = (index_path, entries_snapshot)
        # CRITICAL: was `= paths_to_unlink` which OVERWROTE
        # pending unlinks if two evictions queued before _flush ran. Use
        # extend to preserve all queued unlinks.
        if not isinstance(getattr(self, '_pending_unlinks', None), list):
            self._pending_unlinks = []
        self._pending_unlinks.extend(paths_to_unlink)

    def _flush_pending_io(self) -> None:
        """Perform deferred I/O (block writes + index write + block file deletion).

        Must be called **after** releasing ``_lock``.  Safe to call even
        when there is nothing pending (no-op).
        """
        # Swap pending fields under the lock into locals so that
        # _evict_lru_locked() / _store_new_block() can safely repopulate
        # them while we do I/O.
        with self._lock:
            pending_write = self._pending_index_write
            pending_unlinks = self._pending_unlinks
            pending_block_writes = self._pending_block_writes
            self._pending_index_write = None
            self._pending_unlinks = None
            self._pending_block_writes = None

        # Write deferred block data files first — the index snapshot already
        # includes the entries, so on crash the index file will reference
        # these blocks.  If the process crashes mid-write, the block file
        # may be truncated/missing, which load() handles gracefully.
        if pending_block_writes:
            for path, payload, block_hash, block_index, _header_size in pending_block_writes:
                try:
                    with open(path, "wb") as f:
                        f.write(payload)
                except OSError as e:
                    logger.warning(
                        "Failed to write SSD cache block %d: %s — rolling back index",
                        block_index, e,
                    )
                    # Roll back the in-memory index entry so future lookups
                    # don't reference a missing file.  The space accounting
                    # must also be reverted.
                    with self._lock:
                        entry = self._index.pop(block_hash, None)
                        if entry is not None:
                            self._current_size_bytes = max(
                                0, self._current_size_bytes - entry.size_bytes
                            )

        if pending_write is not None:
            index_path, entries = pending_write
            self._write_index_to_disk(index_path, entries)
        if pending_unlinks:
            for path in pending_unlinks:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)

    def get_stats(self) -> dict:
        with self._lock:
            num_entries = len(self._index)
            size_bytes = self._current_size_bytes
        return {
            "num_entries": num_entries,
            "size_bytes": size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "size_gb": round(size_bytes / 1024**3, 2),
            "max_size_gb": round(self.max_size_bytes / 1024**3, 2),
            "utilization_pct": (
                size_bytes / self.max_size_bytes * 100
                if self.max_size_bytes > 0 else 0
            ),
        }


# ── Tiered Cache Manager ──


class TieredKVCacheManager:
    """Manages hot/warm/cold KV cache tiers.

    Coordination strategy:
    - New requests: check hot (UMA) cache first → warm tier → SSD cache
    - Hot cache miss but warm hit: promote from warm to hot
    - Hot cache miss but SSD hit: load from SSD into hot cache
    - Hot cache pressure: evict LRU blocks to warm tier, then SSD
    - Request completion: free hot blocks, persist to SSD if caching enabled

    This adds a warm tier (quantized) for Apple Silicon optimization.
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

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the hot-tier manager.

        CRITICAL: the engine consumes the KV manager polymorphically —
        the scheduler reads ``.block_size``, ``.block_pool``, ``.num_free_blocks``
        etc. directly. TieredKVCacheManager only overrode allocate_for_prefill /
        free_request, so any other access raised
        "'TieredKVCacheManager' object has no attribute 'block_size'" and the
        engine-loop ``add_request`` insert failed for EVERY request → empty
        output. That made the entire 4-tier KV path (YUNSHU_SSD_CACHE=1 /
        YUNSHU_KV_OFFLOAD=1 under the continuous-batching loop) unusable. This
        makes Tiered a true drop-in wrapper: anything it doesn't override falls
        through to the hot manager (block_size/block_pool/free counts all live
        there). ``name == 'hot'`` is guarded to avoid recursion before __init__
        finishes setting self.hot.
        """
        if name in ("hot", "ssd", "warm"):
            raise AttributeError(name)
        return getattr(self.hot, name)

    def allocate_for_prefill(
        self,
        token_ids: list[int],
        model_hash: int = 0,
    ) -> tuple[BlockTable, PrefixMatch]:
        """Allocate blocks with hot + warm + SSD tier lookup.

        1. Check hot cache (UMA-resident FP16)
        2. For misses, check warm tier (4-bit quantized)
        3. For remaining misses, check SSD cache
        4. Allocate new blocks for remaining tokens
        """
        table, match = self.hot.allocate_for_prefill(token_ids, model_hash)

        # If there are unmatched tokens, check warm tier and SSD cache
        if not match.unmatched_token_ids:
            return table, match

        # Hold hot manager lock for the entire warm/SSD promotion section
        # to prevent concurrent evict_for_memory / free_request from
        # mutating _key_cache / _value_cache / table._blocks.
        with self.hot._lock:
            return self._allocate_prefill_promote(table, match, model_hash)

    def _allocate_prefill_promote(self, table, match, model_hash):
        """Promote warm/SSD blocks into hot tier. Caller holds hot._lock."""
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
                # Pre-check: skip promote() if no hot blocks available, to
                # avoid popping the warm entry and then failing to re-home it.
                if self.hot.block_pool.get_free_block_count() <= 0:
                    logger.debug(
                        "Warm promotion skipped: no free hot blocks for 0x%x", h,
                    )
                    break
                # Direct promote() without preceding contains() to avoid
                # TOCTOU race: a concurrent eviction between contains() and
                # promote() would cause us to break instead of falling
                # through to SSD.  promote() is atomic — it returns None
                # if the entry has been evicted.
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
                        # Re-insert to avoid data loss.  If re-insertion also
                        # fails, fall back to SSD (if available) before logging
                        # the loss rather than silently discarding.
                        try:
                            self.warm.demote(h, kv_data, num_tokens=block_size)
                        except Exception:
                            ssd_ok = False
                            if self.ssd is not None:
                                try:
                                    ssd_ok = self.ssd.store(h, kv_data, num_tokens=block_size)
                                except Exception:
                                    logger.debug(
                                        "SSD fallback also failed for hash 0x%x",
                                        h, exc_info=True,
                                    )
                            if ssd_ok:
                                logger.info(
                                    "Warm re-insertion failed, KV data saved to "
                                    "SSD fallback for hash 0x%x", h,
                                )
                            else:
                                logger.error(
                                    "Warm tier promotion AND re-insertion AND SSD "
                                    "fallback all failed for hash 0x%x — KV data LOST",
                                    h, exc_info=True,
                                )
                        break
                    # Write KV data into hot cache tensors.
                    # Packed format: [2, num_heads, block_size, head_dim]
                    # where dim 0 has key at [0] and value at [1].
                    try:
                        if self.hot._key_cache is not None:
                            # In-place write of the block slice. NEVER rebind
                            # _key_cache to the slice — that would replace the
                            # whole [num_blocks, ...] tensor with one block.
                            if kv_data.ndim == 4 and kv_data.shape[0] == 2:
                                self.hot._key_cache[new_block.block_id] = kv_data[0]
                                if self.hot._value_cache is not None:
                                    self.hot._value_cache[new_block.block_id] = kv_data[1]
                            else:
                                self.hot._key_cache[new_block.block_id] = kv_data
                            if isinstance(self.hot._key_cache, mx.array):
                                mx.eval(self.hot._key_cache)
                                if self.hot._value_cache is not None:
                                    mx.eval(self.hot._value_cache)
                    except Exception:
                        logger.debug(
                            "Warm tier KV data write failed for block %d",
                            new_block.block_id, exc_info=True,
                        )
                        # Re-insert into warm tier to avoid data loss, then
                        # free the allocated hot block.
                        self.warm.demote(h, kv_data, num_tokens=block_size)
                        self.hot.block_pool.free([new_block])
                        break
                    if self.hot._key_cache is None:
                        # FAIL-CLOSED: hot KV tensors unbound → the block holds
                        # UNINITIALIZED KV. The old code skipped the write-back silently yet
                        # still cache_block'd + counted it → an empty block registered as a
                        # prefix-cache HIT = silent KV corruption / wrong tokens once this
                        # dormant path is wired. Re-insert + stop instead of a bogus hit.
                        self.warm.demote(h, kv_data, num_tokens=block_size)
                        self.hot.block_pool.free([new_block])
                        break
                    # Register in prefix cache for future lookups
                    self.hot.block_pool.cache_block(new_block, h)
                    warm_promoted_blocks.append(new_block)
                    warm_loaded += 1
                    parent_hash = h  # chain: next block uses this as parent
                else:
                    # Warm miss or entry evicted concurrently — fall through
                    # to SSD lookup rather than breaking the chain entirely.
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
                # Direct load() without preceding contains() to avoid TOCTOU
                # race: a concurrent eviction between contains() and load()
                # would return None and break the chain.  load() is atomic —
                # it returns None if the entry was evicted or on disk error.
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
                    try:
                        if self.hot._key_cache is not None:
                            # In-place write of the block slice. NEVER rebind
                            # _key_cache to the slice — that would replace the
                            # whole [num_blocks, ...] tensor with one block.
                            if kv_data.ndim == 4 and kv_data.shape[0] == 2:
                                self.hot._key_cache[new_block.block_id] = kv_data[0]
                                if self.hot._value_cache is not None:
                                    self.hot._value_cache[new_block.block_id] = kv_data[1]
                            else:
                                self.hot._key_cache[new_block.block_id] = kv_data
                            if isinstance(self.hot._key_cache, mx.array):
                                mx.eval(self.hot._key_cache)
                                if self.hot._value_cache is not None:
                                    mx.eval(self.hot._value_cache)
                    except Exception:
                        logger.debug(
                            "SSD cache KV data write failed for block %d",
                            new_block.block_id, exc_info=True,
                        )
                        # Re-store to SSD so the data is not lost, then free
                        # the allocated hot block.
                        self.ssd.store(h, kv_data, num_tokens=block_size)
                        self.hot.block_pool.free([new_block])
                        break
                    if self.hot._key_cache is None:
                        # FAIL-CLOSED (SSD sibling of the warm path): an unbound
                        # hot KV tensor means the block holds UNINITIALIZED KV; don't
                        # register it as a prefix-cache hit. Re-store to SSD + stop.
                        self.ssd.store(h, kv_data, num_tokens=block_size)
                        self.hot.block_pool.free([new_block])
                        break
                    # Register in prefix cache for future lookups
                    self.hot.block_pool.cache_block(new_block, h)
                    ssd_promoted_blocks.append(new_block)
                    ssd_loaded += 1
                    parent_hash = h  # chain: next block uses this as parent
                else:
                    # SSD miss or entry evicted concurrently — stop chain.
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
            # Must hold block_pool._lock for the entire sequence of
            # eviction + ref_count reset + free to prevent a concurrent
            # allocate() from grabbing a surplus block between the
            # eviction and the free.  block_pool.free() acquires _lock
            # internally, so we call the unlocked path directly.
            pool = self.hot.block_pool
            with pool._lock:
                for block in surplus:
                    if block.block_hash is not None:
                        pool._evict_cached_block_unlocked(block)
                    block.ref_count = 1
                # Inline free() logic (decrement ref_count, return to
                # free queue) without re-acquiring _lock.  Using the
                # public free() here would deadlock because threading.Lock
                # is not reentrant and we already hold _lock.
                freed = []
                seen_ids: set[int] = set()
                for block in surplus:
                    if block.block_id in seen_ids:
                        continue
                    seen_ids.add(block.block_id)
                    if block.ref_count <= 0:
                        continue
                    block.ref_count -= 1
                    if block.ref_count == 0 and not block.is_null:
                        if block.block_hash is not None:
                            block.cache_only = True
                        freed.append(block)
                pool.free_queue.append_n(freed)
            # Rebuild the internal block list with promoted blocks spliced in.
            table._blocks = (
                existing[:hot_matched_count]
                + all_promoted
                + existing[hot_matched_count + len(all_promoted):]
            )
            len(table._blocks)
            # Compute total_tokens from actual token counts rather than
            # block count * block_size, which overcounts when matched blocks
            # are not fully aligned to block boundaries.
            matched_tok = match.num_matched_tokens
            unmatched_tok = len(match.unmatched_token_ids) if match.unmatched_token_ids else 0
            actual_tokens = matched_tok + unmatched_tok
            table.total_tokens = actual_tokens
            remainder = actual_tokens % block_size
            if actual_tokens == 0:
                table._last_block_occupancy = 0
            else:
                table._last_block_occupancy = remainder if remainder != 0 else block_size
            match.matched_blocks = match.matched_blocks + all_promoted

        return table, match

    def _extract_kv_for_block(self, block: KVBlock):
        """Extract KV tensor data for a block from the hot cache.

        Returns a single mx.array shaped [2, ...] with K stacked over V (NOT
        raw bytes). Every offload consumer — warm.demote() → quantize_kv_4bit(),
        ssd_store.store(), and the promote write-back which checks kv_data.shape[0]==2 —
        expects an ARRAY. The old `k.tobytes()+v.tobytes()` made hot→WARM offload
        ALWAYS raise (quantize_kv_4bit(bytes) → caught as a failed block, so NOTHING
        ever left the hot tier) and hot→SSD persist a 1-D uint8 garbage tensor with a
        valid CRC (silently corrupt KV on restore).
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
            import mlx.core as mx
            return mx.stack([key_cache[bid], val_cache[bid]], axis=0)
        except Exception:
            logger.debug("KV block extraction from hot cache failed", exc_info=True)
            return None

    def free_request(self, table: BlockTable, request_id: str | None = None) -> None:
        """Free blocks from a completed request."""
        self.hot.free_request(table, request_id=request_id)

    def get_stats(self) -> dict:
        """Return combined stats from all tiers."""
        total_blocks = self.hot.block_pool.num_blocks - 1  # exclude null block
        free_blocks = self.hot.num_free_blocks
        stats = {
            "hot_usage_pct": round(self.hot.usage * 100, 1),
            "hot_total_blocks": total_blocks,
            "hot_blocks_in_use": total_blocks - free_blocks,
            "hot_free_blocks": free_blocks,
            "hot_block_size": self.hot.block_size,
        }
        if self.warm:
            stats["warm"] = self.warm.get_stats()
        if self.ssd:
            stats["ssd"] = self.ssd.get_stats()
        return stats


# ── Background SSD Flush Thread ──


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
        self._stats_lock = threading.Lock()

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
        # Snapshot entries under the warm tier lock
        with self._warm._lock:
            entries = list(self._warm._store.items())
        for block_hash, entry in entries:
            packed, scales = entry[0], entry[1]
            head_dim = entry[2] if len(entry) > 2 else 0
            # Recover num_tokens from the 4-tuple stored by demote().
            # Falls back to 0 for entries written by older code (3-tuple).
            num_tokens = entry[3] if len(entry) > 3 else 0
            if head_dim == 0:
                continue
            if not self._ssd.contains(block_hash):
                try:
                    from .compression import dequantize_kv_4bit
                    kv_data = dequantize_kv_4bit(packed, scales, head_dim=head_dim)
                    self._ssd.store(block_hash, mx.array(kv_data), num_tokens=num_tokens)
                    flushed += 1
                except Exception as e:
                    logger.debug(
                        "Failed to flush block 0x%x to SSD: %s",
                        block_hash, e,
                    )
        with self._stats_lock:
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
        with self._stats_lock:
            return {
                "flush_count": self._flush_count,
                "last_flush_time": self._last_flush_time,
                "running": self._thread is not None and self._thread.is_alive(),
            }
