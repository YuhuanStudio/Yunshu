from __future__ import annotations

"""Yunshu SSD-Tier KV Cache — persistent KV block storage for prefix reuse.

Features:
- Background writer thread for non-blocking saves
- safetensors-based serialization (thread-safe, no Metal API in writer)
- Hot RAM cache + SSD index for fast lookup
- LRU eviction from hot → SSD, SSD → delete
- Block-level hashing integration with KVPrefixCache
- Dynamic disk budget awareness
- SQLite-backed metadata for crash consistency

Backend selection via YUNSHU_SSD_BACKEND env var:
  "sqlite" (default) — crash-consistent WAL-mode SQLite (yunshu_kv.SSDSQLiteStore)
  "json" — legacy JSON index (backward compat, no crash safety)

Architecture:
  KVPrefixCache
    → SSDKVCache (this module)
      → hot_cache: OrderedDict[bytes, list] (recent blocks in RAM)
      → _index: dict[bytes, _BlockMeta] (all known blocks, on disk or hot)
      → _sqlite_store: SSDSQLiteStore (crash-consistent metadata, when backend=sqlite)
      → _write_queue: Queue (background writer thread)

Thread safety:
  - Tensor bytes extracted on inference thread (Metal-safe)
  - Background writer does only file I/O (no Metal API calls)
  - Index mutations protected by threading.Lock
"""

import contextlib
import json
import logging
import os
import struct
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = "1"
_READABLE_VERSIONS = frozenset({"1"})

# MLX dtype to safetensors dtype string mapping
_DTYPE_MAP = {
    "float16": "F16",
    "mlx.core.float16": "F16",
    "float32": "F32",
    "mlx.core.float32": "F32",
    "bfloat16": "BF16",
    "mlx.core.bfloat16": "BF16",
    "int8": "I8",
    "mlx.core.int8": "I8",
    "int32": "I32",
    "mlx.core.int32": "I32",
    "uint8": "U8",
    "mlx.core.uint8": "U8",
    "uint16": "U16",
    "mlx.core.uint16": "U16",
    "uint32": "U32",
    "mlx.core.uint32": "U32",
    "bool_": "BOOL",
    "mlx.core.bool_": "BOOL",
}


@dataclass
class _BlockMeta:
    """Metadata for a cached block (in-memory index entry)."""
    block_hash: bytes
    file_path: str
    token_count: int
    model_name: str
    created_at: float
    file_size: int = 0
    last_accessed: float = 0.0


@dataclass
class SSDCacheStats:
    """Statistics for the SSD-tier KV cache."""
    hot_cache_entries: int = 0
    hot_cache_bytes: int = 0
    disk_entries: int = 0
    disk_bytes: int = 0
    total_entries: int = 0
    max_size_bytes: int = 0
    writes_completed: int = 0
    reads_completed: int = 0
    evictions: int = 0


def _extract_tensor_bytes_quantized(arr) -> tuple[bytes, str, list[int], float]:
    """Extract raw bytes from an mx.array with 8-bit quantization for SSD storage.

    Uses per-tensor symmetric int8 quantization (2x compression vs FP16).
    Returns (quantized_bytes, dtype_str, shape, scale_factor).

    Runs on the inference thread (Metal-safe for eval'd arrays).
    """
    import mlx.core as mx
    import numpy as np

    # numpy has no bfloat16, so np.array(bf16_mx, dtype=f32)
    # raised "Item size 2 ... does not match" and the SSD spill silently failed
    # for every (bf16) KV block → nothing was ever persisted. Cast to float32
    # INSIDE mlx first (unconditionally — KV is bf16/fp16, and str(dtype) is
    # "mlx.core.bfloat16" so a name check missed it), then hand numpy a real f32
    # buffer. We quantize to int8 next anyway, so the cast costs nothing.
    arr = arr.astype(mx.float32)
    mx.eval(arr)
    numpy_data = np.array(arr)
    shape = list(numpy_data.shape)

    abs_max = float(np.max(np.abs(numpy_data)))
    if abs_max == 0:
        abs_max = 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(numpy_data / scale), -127, 127).astype(np.int8)

    raw = bytes(quantized)
    return raw, "I8", shape, scale


def _extract_tensor_bytes(arr) -> tuple[bytes, str, list[int]]:
    """Extract raw bytes from an mx.array for safetensors serialization.

    Runs on the inference thread (Metal-safe for eval'd arrays).
    """
    import mlx.core as mx

    dtype_name = str(arr.dtype)
    st_dtype = _DTYPE_MAP.get(dtype_name, "F16")
    shape = list(arr.shape)

    # Ensure array is evaluated before extracting bytes
    mx.eval(arr)

    if dtype_name == "bfloat16":
        raw = bytes(memoryview(arr.astype(mx.uint16)))
    else:
        try:
            raw = bytes(memoryview(arr))
        except (TypeError, BufferError):
            raw = bytes(arr)

    return raw, st_dtype, shape


def _write_safetensors(path: str, tensors: dict[str, tuple[bytes, str, list[int]]],
                       metadata: dict[str, str] | None = None) -> int:
    """Write safetensors file without using mlx API (thread-safe).

    Format: [8-byte header_size LE uint64][header JSON padded to 8 bytes][tensor data]
    """
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for name, (raw_bytes, dtype, shape) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw_bytes)],
        }
        offset += len(raw_bytes)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    padding = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * padding

    header_size = len(header_json)
    total_size = 8 + header_size + offset

    tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_size))
            f.write(header_json)
            for name in tensors:
                f.write(tensors[name][0])
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        logger.debug("safetensors write failed", exc_info=True)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    return total_size


def _read_safetensors_metadata(path: str) -> dict[str, Any]:
    """Read safetensors header without loading tensor data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size <= 0 or header_size > 100 * 1024 * 1024:
            raise ValueError(
                f"Corrupt safetensors file {path}: header_size={header_size} "
                f"(expected 1..100MB)"
            )
        header = json.loads(f.read(header_size))
    return header


class SSDKVCache:
    """Persistent SSD-tier KV cache for block-level prefix cache persistence.

    Integrates with KVPrefixCache to provide:
    - Background non-blocking save of KV blocks to disk
    - Fast hot cache for recently accessed blocks
    - LRU eviction from hot → SSD → delete
    - Dynamic disk budget management

    Thread safety:
    - All public methods are thread-safe (Lock-protected)
    - Background writer thread handles all file I/O
    """

    def __init__(
        self,
        cache_dir: str = "~/.cache/yunshu/kv-ssd",
        max_size_bytes: int = 10 * 1024 ** 3,  # 10 GB default
        hot_cache_size: int = 100,
        writer_queue_size: int = 64,
        backend: str | None = None,
    ):
        self._cache_dir = Path(os.path.expanduser(cache_dir))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Create hex-bucket subdirectories
        for h in "0123456789abcdef":
            (self._cache_dir / h).mkdir(exist_ok=True)

        self._max_size_bytes = max_size_bytes
        self._hot_cache_size = hot_cache_size
        self._lock = threading.Lock()

        # Hot cache: block_hash_hex → (cache_data, token_count)
        self._hot_cache: OrderedDict[str, tuple[list, int]] = OrderedDict()

        # On-disk index: block_hash_hex → _BlockMeta
        self._index: dict[str, _BlockMeta] = {}

        # Backend selection: "sqlite" (default, crash-consistent) or "json" (legacy)
        self._backend = (
            backend
            or os.environ.get("YUNSHU_SSD_BACKEND", "sqlite")
        ).lower()
        if self._backend not in ("sqlite", "json"):
            logger.warning("Unknown YUNSHU_SSD_BACKEND=%r, falling back to sqlite", self._backend)
            self._backend = "sqlite"

        # SQLite metadata store for crash consistency
        self._sqlite_store: SSDSQLiteStore | None = None
        self._db = None  # kept for backward compat alias
        if self._backend == "sqlite":
            from yunshu_kv.ssd_sqlite_store import SSDSQLiteStore
            db_path = self._cache_dir / "index.db"
            self._sqlite_store = SSDSQLiteStore(db_path)
            # Attempt JSON → SQLite migration for backward compat
            json_index_path = self._cache_dir / "index.json"
            if json_index_path.exists() and db_path.exists():
                migrated = self._sqlite_store.import_json_index(json_index_path)
                if migrated > 0:
                    logger.info("SSD KV cache: migrated %d entries from JSON to SQLite", migrated)

        # Background writer
        self._write_queue: list[tuple] = []
        self._writer_thread: threading.Thread | None = None
        self._writer_stop = threading.Event()
        self._writer_lock = threading.Lock()
        self._writer_queue_size = writer_queue_size

        # Stats
        self._writes_completed = 0
        self._reads_completed = 0
        self._evictions = 0

        # Recover existing index from disk
        self._recover_index()

    def _start_writer(self) -> None:
        """Start the background writer thread if not running."""
        with self._writer_lock:
            if self._writer_thread is not None and self._writer_thread.is_alive():
                return
            self._writer_stop.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="ssd-kv-writer"
            )
            self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Background writer loop — processes save/delete operations."""
        while not self._writer_stop.is_set():
            item = None
            with self._writer_lock:
                if self._write_queue:
                    item = self._write_queue.pop(0)

            if item is None:
                self._writer_stop.wait(timeout=0.5)
                continue

            self._write_one_item(item)

    def _flush_writer(self) -> None:
        """Flush all pending writes.

        Signals the background writer to stop, then drains any remaining
        items.  Uses _drain_pending_writes_locked() which handles lock
        ordering internally to prevent AB/BA deadlock.
        """
        # Signal writer to stop and wait for it
        self._writer_stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        self._writer_thread = None
        self._writer_stop.clear()
        # Drain remaining items.  _drain_pending_writes_locked handles
        # its own locking (snapshots queue under _writer_lock, processes
        # outside any lock) so we do NOT wrap it in _writer_lock here.
        self._drain_pending_writes_locked()

    def _recover_index(self) -> None:
        """Recover block index from SQLite store, or scan cache directory."""
        # Try SQLite store first (fast, crash-consistent)
        if self._sqlite_store is not None:
            try:
                # Run recovery to clean up orphaned entries
                self._sqlite_store.recover()
                count = self._recover_from_sqlite_store()
                if count > 0:
                    logger.info(f"SSD KV cache: recovered {count} blocks from SQLite store")
                    return
            except Exception:
                logger.debug("SQLite store recovery failed, falling back to scan", exc_info=True)

        # Fallback: scan safetensors headers
        count = 0
        for bucket in self._cache_dir.iterdir():
            if not bucket.is_dir() or len(bucket.name) != 1:
                continue
            for f in bucket.iterdir():
                # Clean up leftover .tmp files from interrupted writes
                if f.name.endswith(".tmp"):
                    try:
                        os.unlink(f)
                        logger.debug("Cleaned up leftover .tmp file: %s", f.name)
                    except OSError:
                        pass
                    continue
                if not f.name.endswith(".safetensors"):
                    continue
                block_hash_hex = f.stem
                # Skip zero-byte files (incomplete writes from a crash)
                try:
                    if f.stat().st_size == 0:
                        logger.debug("Skipping zero-byte safetensors file: %s", f.name)
                        os.unlink(f)
                        continue
                except OSError:
                    continue
                try:
                    header = _read_safetensors_metadata(str(f))
                    meta = header.get("__metadata__", {})
                    version = meta.get("yunshu_cache_version", "unknown")
                    if version not in _READABLE_VERSIONS:
                        continue
                    self._index[block_hash_hex] = _BlockMeta(
                        block_hash=bytes.fromhex(block_hash_hex),
                        file_path=str(f),
                        token_count=int(meta.get("token_count", "0")),
                        model_name=meta.get("model_name", ""),
                        created_at=float(meta.get("created_at", "0")),
                        file_size=f.stat().st_size,
                    )
                    # Populate SQLite store if available (migration from file scan)
                    if self._sqlite_store is not None:
                        self._sqlite_store.put(
                            block_hash_hex, str(f),
                            int(meta.get("token_count", "0")),
                            f.stat().st_size,
                        )
                    count += 1
                except Exception:
                    logger.debug("SSD block index recovery failed", exc_info=True)
        if count > 0:
            logger.info(f"SSD KV cache: recovered {count} blocks from disk scan")

    def _recover_from_sqlite_store(self) -> int:
        """Recover block metadata from SSDSQLiteStore."""
        if self._sqlite_store is None:
            return 0
        entries = self._sqlite_store.list_all()
        count = 0
        stale = []
        for entry in entries:
            bh_hex = entry["block_hash"]
            fpath = entry["block_path"]
            # Verify the file actually exists on disk — if the process
            # crashed after SQLite insert but before the safetensors
            # write completed, the entry is stale and must be pruned.
            if not os.path.isfile(fpath):
                stale.append(bh_hex)
                logger.debug(
                    "SSD recovery: pruning stale entry %s (file missing)",
                    bh_hex[:16],
                )
                continue
            self._index[bh_hex] = _BlockMeta(
                block_hash=bytes.fromhex(bh_hex),
                file_path=fpath,
                token_count=entry["num_tokens"],
                model_name="",  # model_name not in SSDSQLiteStore schema
                created_at=entry["created_at"],
                file_size=entry["size_bytes"],
                last_accessed=entry["last_accessed"],
            )
            count += 1
        # Prune stale entries from SQLite store
        for bh_hex in stale:
            try:
                self._sqlite_store.delete(bh_hex)
            except Exception:
                logger.debug("stale entry deletion failed for %s", bh_hex[:16], exc_info=True)
        if stale:
            logger.info("SSD recovery: pruned %d stale entries (files missing)", len(stale))
        return count

    def _sqlite_upsert(self, hex_hash: str, meta: _BlockMeta) -> None:
        """Insert or update a block entry in the metadata store."""
        if self._sqlite_store is not None:
            self._sqlite_store.put(
                hex_hash, meta.file_path, meta.token_count, meta.file_size,
            )

    def _sqlite_delete(self, hex_hash: str) -> None:
        """Delete a block entry from the metadata store."""
        if self._sqlite_store is not None:
            self._sqlite_store.delete(hex_hash)

    def _block_path(self, block_hash: bytes) -> str:
        """Get file path for a block hash."""
        hex_hash = block_hash.hex()
        return str(self._cache_dir / hex_hash[0] / f"{hex_hash}.safetensors")

    def save_block(
        self,
        block_hash: bytes,
        cache_data: list,
        token_count: int,
        model_name: str = "",
    ) -> None:
        """Save a KV block to the SSD cache (non-blocking).

        Extracts tensor bytes on the calling thread (Metal-safe),
        then enqueues for background writing.

        Args:
            block_hash: Block content hash (from KVPrefixCache block hashing)
            cache_data: Per-layer KV cache data list
            token_count: Number of tokens in this block
            model_name: Model name for isolation
        """
        # Extract tensor bytes on the inference thread with 8-bit quantization
        # to reduce SSD I/O bandwidth and disk usage (2x compression vs FP16).
        # Each entry: (quantized_bytes, "I8", shape, scale_factor).
        tensors_quantized: dict[str, tuple[bytes, str, list[int], float]] = {}
        tensors_raw: dict[str, tuple[bytes, str, list[int]]] = {}
        for i, layer_data in enumerate(cache_data):
            if isinstance(layer_data, (list, tuple)):
                for k, tensor in enumerate(layer_data):
                    if tensor is not None and hasattr(tensor, 'shape'):
                        key = f"layer_{i}_state_{k}"
                        tensors_quantized[key] = _extract_tensor_bytes_quantized(tensor)
            elif hasattr(layer_data, 'keys') and hasattr(layer_data, 'values'):
                tensors_quantized[f"layer_{i}_keys"] = _extract_tensor_bytes_quantized(layer_data.keys)
                tensors_quantized[f"layer_{i}_values"] = _extract_tensor_bytes_quantized(layer_data.values)
            elif hasattr(layer_data, 'shape'):
                # Raw mx.array layer (no keys/values wrapper)
                tensors_quantized[f"layer_{i}_state_0"] = _extract_tensor_bytes_quantized(layer_data)

        # Build the safetensors-compatible dict: (bytes, dtype, shape)
        # Store scale factors in metadata for dequantization on load.
        scale_factors: dict[str, float] = {}
        for key, (raw_bytes, dtype_str, shape, scale) in tensors_quantized.items():
            tensors_raw[key] = (raw_bytes, dtype_str, shape)
            scale_factors[key] = scale

        if not tensors_raw:
            return

        meta = {
            "yunshu_cache_version": _CACHE_FORMAT_VERSION,
            "token_count": str(token_count),
            "num_layers": str(len(cache_data)),
            "model_name": model_name,
            "created_at": str(time.time()),
            "quantization": "int8",
            "scale_factors": json.dumps(scale_factors),
        }

        hex_hash = block_hash.hex()
        file_path = self._block_path(block_hash)

        # Add to hot cache and in-memory index.
        # NOTE: SQLite upsert is deferred until the file is written to disk
        # (in _writer_loop / _process_pending_writes). This prevents the
        # SQLite store from claiming a block exists on disk if the write
        # fails (e.g., disk full). The in-memory index is safe because it
        # only means "data is available via hot cache".
        with self._lock:
            self._hot_cache[hex_hash] = (cache_data, token_count)
            self._hot_cache.move_to_end(hex_hash)
            self._evict_hot_if_full()
            self._index[hex_hash] = _BlockMeta(
                block_hash=block_hash,
                file_path=file_path,
                token_count=token_count,
                model_name=model_name,
                created_at=time.time(),
                file_size=0,  # Updated after successful write
            )

        # Enqueue for background writing (no lock held here).
        # NOTE: We never silently drop queued saves — dropping would leave
        # the index/SQLite claiming the block exists on disk when it was
        # never written (data loss).  Instead we block the caller until
        # the queue drains below capacity.
        with self._writer_lock:
            # Drain queue synchronously if at capacity to prevent unbounded growth
            if self._writer_queue_size > 0 and len(self._write_queue) >= self._writer_queue_size:
                self._drain_pending_writes_locked()
            # Remove any existing save for this hash (replace, don't duplicate)
            self._write_queue = [
                item for item in self._write_queue
                if not (item[0] == "save" and item[1] == hex_hash)
            ]
            self._write_queue.append(("save", hex_hash, tensors_raw, meta, file_path))
            queue_len = len(self._write_queue)

        # For small caches, write synchronously to avoid thread management overhead
        # Background writer is started only when queue exceeds threshold.
        # Snapshot queue_len under _writer_lock to avoid racing with the
        # background writer thread which pops from _write_queue under the
        # same lock.
        if queue_len <= 2:
            self._process_pending_writes()
        else:
            self._start_writer()

    def _process_pending_writes(self) -> None:
        """Process pending writes synchronously (acquires _writer_lock)."""
        while True:
            with self._writer_lock:
                if not self._write_queue:
                    break
                item = self._write_queue.pop(0)
            self._write_one_item(item)

    def _drain_pending_writes_locked(self) -> None:
        """Process pending writes synchronously (caller holds _writer_lock).

        Used by save_block and _flush_writer which already hold _writer_lock.
        Calling _process_pending_writes() from those sites would deadlock
        because _process_pending_writes acquires _writer_lock and
        threading.Lock is not reentrant.

        IMPORTANT: _write_one_item() acquires _lock internally.  To avoid
        AB/BA deadlock (caller holds _writer_lock, _write_one_item wants
        _lock; a concurrent save_block holds _lock and wants _writer_lock),
        we must NOT hold _writer_lock while calling _write_one_item().
        We snapshot and clear the queue under _writer_lock, then drain it
        with no locks held, and re-acquire _writer_lock only if we need to
        re-check the queue after draining.
        """
        while True:
            with self._writer_lock:
                if not self._write_queue:
                    return
                # Snapshot the entire queue and clear it atomically so
                # new items appended during drain go to a fresh list.
                batch = list(self._write_queue)
                self._write_queue.clear()
            # Process batch without holding any lock — _write_one_item
            # acquires _lock internally for index updates, which is safe
            # because we don't hold _writer_lock here.
            for item in batch:
                self._write_one_item(item)

    def _write_one_item(self, item: tuple) -> None:
        """Process a single write/delete item (no lock held by caller)."""
        try:
            op = item[0]
            if op == "save":
                _, block_hash_hex, tensors_raw, meta_dict, file_path = item
                # Check if the entry is still in the index before writing.
                # A concurrent delete_block() may have removed it, and writing
                # the file would leave an orphan on disk that's never cleaned up.
                with self._lock:
                    if block_hash_hex not in self._index:
                        logger.debug(
                            "Skipping SSD write for %s — entry removed from index",
                            block_hash_hex[:16],
                        )
                        return
                file_size = _write_safetensors(file_path, tensors_raw, meta_dict)
                with self._lock:
                    if block_hash_hex in self._index:
                        self._index[block_hash_hex].file_size = file_size
                        self._sqlite_upsert(block_hash_hex, self._index[block_hash_hex])
                        self._writes_completed += 1
                        # enforce the disk budget after each successful write.
                        self._enforce_disk_budget_locked(skip_hash=block_hash_hex)
                    else:
                        # Entry was deleted between the check above and here.
                        # Clean up the file we just wrote to avoid orphan.
                        logger.debug(
                            "SSD block %s deleted during write — cleaning up file",
                            block_hash_hex[:16],
                        )
                        with contextlib.suppress(OSError):
                            os.unlink(file_path)
            elif op == "delete":
                with contextlib.suppress(OSError):
                    os.unlink(item[1])
        except Exception:
            logger.debug("SSD write error", exc_info=True)

    def load_block(self, block_hash: bytes) -> list | None:
        """Load a KV block from hot cache or SSD.

        Returns:
            Cache data list, or None if not found.
        """
        hex_hash = block_hash.hex()

        with self._lock:
            # Check hot cache first
            if hex_hash in self._hot_cache:
                self._hot_cache.move_to_end(hex_hash)
                self._reads_completed += 1
                # Return a deep copy so callers can't mutate the hot
                # cache entry (e.g., modifying layer offset/trim).
                # A shallow list() copy is insufficient because callers
                # may mutate inner numpy/mx arrays in-place.
                import copy
                return copy.deepcopy(self._hot_cache[hex_hash][0])

            # Check disk index — capture meta while holding the lock
            # to prevent a concurrent delete_block from racing with us.
            meta = self._index.get(hex_hash)
            if meta is None:
                return None
            if meta.file_size == 0:
                return None

        # Load from disk (blocking, runs on inference thread)
        if meta is None:
            return None

        try:
            import mlx.core as mx
            import numpy as np
            data, header = mx.load(meta.file_path, return_metadata=True)

            # Check if data was stored with int8 quantization
            is_quantized = header.get("quantization") == "int8"
            scale_factors = {}
            if is_quantized:
                try:
                    scale_factors = json.loads(header.get("scale_factors", "{}"))
                except (json.JSONDecodeError, TypeError):
                    scale_factors = {}

            # Reconstruct cache data from safetensors keys
            num_layers = int(header.get("num_layers", "0"))
            cache_data = [None] * num_layers

            for i in range(num_layers):
                keys_key = f"layer_{i}_keys"
                vals_key = f"layer_{i}_values"
                if keys_key in data and vals_key in data:
                    k_data = data[keys_key]
                    v_data = data[vals_key]
                    if is_quantized:
                        k_scale = scale_factors.get(keys_key)
                        v_scale = scale_factors.get(vals_key)
                        if k_scale is None or v_scale is None:
                            logger.warning(
                                "SSD block %s missing scale factor for layer %d "
                                "(keys_found=%s, vals_found=%s) — treating as corrupt",
                                hex_hash[:16], i,
                                keys_key in scale_factors,
                                vals_key in scale_factors,
                            )
                            return None
                        # dequantize in float32, NOT float16. The default KV
                        # dtype is bf16 (max ~3.4e38); an int8*scale product can exceed
                        # fp16's 65504 max for large-model outlier channels / attention
                        # sinks → np.float16 silently overflows to inf → NaN attention on
                        # restore. float32 holds the full bf16 range losslessly.
                        k_data = mx.array(
                            np.array(k_data, dtype=np.float32)
                            * np.float32(k_scale)
                        )
                        v_data = mx.array(
                            np.array(v_data, dtype=np.float32)
                            * np.float32(v_scale)
                        )
                    cache_data[i] = (k_data, v_data)
                else:
                    # Collect state_N entries
                    states = []
                    k = 0
                    while f"layer_{i}_state_{k}" in data:
                        tensor = data[f"layer_{i}_state_{k}"]
                        if is_quantized:
                            s_key = f"layer_{i}_state_{k}"
                            s_scale = scale_factors.get(s_key)
                            if s_scale is None:
                                logger.warning(
                                    "SSD block %s missing scale factor for %s "
                                    "— treating as corrupt",
                                    hex_hash[:16], s_key,
                                )
                                return None
                            tensor = mx.array(  # float32, not fp16 (overflow→inf)
                                np.array(tensor, dtype=np.float32)
                                * np.float32(s_scale)
                            )
                        states.append(tensor)
                        k += 1
                    if states:
                        cache_data[i] = states

            # Promote to hot cache — re-validate that the entry wasn't
            # deleted or replaced by a concurrent delete_block() while we
            # read from disk (TOCTOU guard: verify file_path still matches).
            token_count = int(header.get("token_count", "0"))
            with self._lock:
                meta_now = self._index.get(hex_hash)
                if meta_now is None or meta_now.file_path != meta.file_path:
                    return None
                self._hot_cache[hex_hash] = (cache_data, token_count)
                self._hot_cache.move_to_end(hex_hash)
                self._evict_hot_if_full()
                self._reads_completed += 1

            return cache_data

        except Exception:
            logger.debug(f"SSD KV load failed for {hex_hash[:16]}", exc_info=True)
            # Prune stale entries: either file_size > 0 (corrupted/missing file)
            # or file_size == 0 but old enough that the writer should have finished
            # (prevents phantom entries where has_block() returns True but
            # load_block() always returns None).
            #
            # CRITICAL: Only prune if the current index entry still points to
            # the SAME file that failed to load.  A concurrent save_block()
            # may have replaced the entry with a different file_path — pruning
            # that would destroy the new valid entry (data loss).
            with self._lock:
                meta_now = self._index.get(hex_hash)
                if meta_now is not None and meta_now.file_path == meta.file_path:
                    if meta_now.file_size > 0:
                        # File was written but is corrupted — safe to prune
                        self._index.pop(hex_hash, None)
                        self._hot_cache.pop(hex_hash, None)
                        self._sqlite_delete(hex_hash)
                        try:
                            if meta_now.file_path:
                                os.unlink(meta_now.file_path)
                        except OSError:
                            pass
                    elif meta_now.created_at > 0 and (time.time() - meta_now.created_at) > 30:
                        # file_size == 0 but entry is >30s old — writer failed/lost
                        self._index.pop(hex_hash, None)
                        self._hot_cache.pop(hex_hash, None)
                        self._sqlite_delete(hex_hash)
            return None

    def has_block(self, block_hash: bytes) -> bool:
        """Check if a block exists in hot cache or on disk."""
        hex_hash = block_hash.hex()
        with self._lock:
            if hex_hash in self._hot_cache:
                return True
            meta = self._index.get(hex_hash)
            # Phantom entries (file_size == 0) mean the writer hasn't finished
            return meta is not None and meta.file_size > 0

    def delete_block(self, block_hash: bytes) -> None:
        """Delete a block from both hot cache and disk.

        Lock ordering: acquires ``_lock`` to remove from index/hot cache,
        then acquires ``_writer_lock`` to cancel pending writes.  Never
        holds both simultaneously to avoid AB/BA deadlock with
        ``_process_pending_writes`` (which acquires ``_writer_lock`` then
        ``_lock``).

        A concurrent ``save_block()`` could re-insert the entry between
        the two critical sections.  This is benign: the re-inserted entry
        will either be overwritten on the next save cycle or cleaned up
        by the next disk budget check.
        """
        hex_hash = block_hash.hex()
        file_path_to_delete: str | None = None
        with self._lock:
            self._hot_cache.pop(hex_hash, None)
            meta = self._index.pop(hex_hash, None)
            self._sqlite_delete(hex_hash)
            if meta is not None and meta.file_path:
                file_path_to_delete = meta.file_path

        # Mutate write queue under writer_lock ONLY (no _lock held).
        # This avoids the AB/BA deadlock with _process_pending_writes
        # which acquires _writer_lock → _lock.
        if file_path_to_delete is not None:
            with self._writer_lock:
                # Cancel any pending save for this hash to prevent orphaned files
                self._write_queue = [
                    item for item in self._write_queue
                    if not (item[0] == "save" and item[1] == hex_hash)
                ]
                self._write_queue.append(("delete", file_path_to_delete))

            self._process_pending_writes()

    def clear(self) -> int:
        """Clear all cached blocks from memory and disk. Returns count cleared."""
        with self._lock:
            count = len(self._index)
            self._hot_cache.clear()
            paths = [m.file_path for m in self._index.values()]
            self._index.clear()
            # Clear SQLite store
            if self._sqlite_store is not None:
                # Delete all entries individually for correctness
                try:
                    for entry in self._sqlite_store.list_all():
                        self._sqlite_store.delete(entry["block_hash"])
                except Exception:
                    logger.debug("SQLite store clear failed", exc_info=True)

        for p in paths:
            with contextlib.suppress(OSError):
                os.unlink(p)

        return count

    def _evict_hot_if_full(self) -> None:
        """Evict oldest hot cache entries when over capacity."""
        while len(self._hot_cache) > self._hot_cache_size:
            hex_hash, _ = self._hot_cache.popitem(last=False)
            # Data stays on disk — just evict from RAM

    def _enforce_disk_budget_locked(self, skip_hash: str | None = None) -> None:
        """Evict LRU on-disk blocks (delete their files + index/sqlite entries) until the
        total on-disk size is within _max_size_bytes. Caller MUST hold self._lock.

        The disk budget (max_size_bytes / YUNSHU_SSD_CACHE_MAX_GB) was accepted,
        stored, and reported in get_stats but NEVER enforced — save_block wrote
        unconditionally and the only "eviction" dropped RAM-hot entries to disk (never
        deleting files), so the SSD KV grew without bound across requests and model reloads
        until the disk filled. Now enforced after each successful write."""
        if self._max_size_bytes <= 0:
            return
        total = sum(m.file_size for m in self._index.values())
        if total <= self._max_size_bytes:
            return
        # Oldest activity first; a never-read entry falls back to its write time so the
        # block just written (skip_hash, created_at=now, last_accessed=0) is evicted LAST.
        order = sorted(
            (h for h, m in self._index.items() if m.file_size > 0 and h != skip_hash),
            key=lambda h: max(self._index[h].last_accessed, self._index[h].created_at),
        )
        for h in order:
            if total <= self._max_size_bytes:
                break
            meta = self._index.pop(h, None)
            if meta is None:
                continue
            total -= meta.file_size
            self._evictions += 1
            self._hot_cache.pop(h, None)  # drop any RAM copy too
            with contextlib.suppress(OSError):
                os.unlink(meta.file_path)
            with contextlib.suppress(Exception):
                self._sqlite_delete(h)

    def get_stats(self) -> SSDCacheStats:
        """Return cache statistics."""
        with self._lock:
            hot_bytes = 0
            disk_bytes = 0
            hot_set = set(self._hot_cache.keys())
            for hex_hash, meta in self._index.items():
                if hex_hash in hot_set:
                    hot_bytes += meta.file_size
                else:
                    disk_bytes += meta.file_size

            return SSDCacheStats(
                hot_cache_entries=len(self._hot_cache),
                hot_cache_bytes=hot_bytes,
                disk_entries=len(self._index) - len(self._hot_cache),
                disk_bytes=disk_bytes,
                total_entries=len(self._index),
                max_size_bytes=self._max_size_bytes,
                writes_completed=self._writes_completed,
                reads_completed=self._reads_completed,
                evictions=self._evictions,
            )

    def close(self) -> None:
        """Flush and stop the background writer."""
        self._flush_writer()
        # Close SQLite store
        if self._sqlite_store is not None:
            self._sqlite_store.close()
        # Ensure thread reference is cleared
        self._writer_thread = None
