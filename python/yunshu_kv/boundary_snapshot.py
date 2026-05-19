from __future__ import annotations
"""Yunshu Boundary Snapshot SSD Store.

Stores non-sliceable cache layer snapshots (ArraysCache, RotatingKVCache) to
SSD during prefill, freeing GPU memory immediately. At request completion the
snapshots are loaded back for final storage.

Based on oMLX's boundary_snapshot_store.py pattern:
- Async writes: tensors serialized on Metal thread, buffered for read-back
- Background writer flushes to disk via thread
- Ephemeral files cleaned up on request completion/abort
"""

import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAGIC = b"YNSHBS01"  # Yunshu Boundary Snapshot v1
_HEADER_VERSION = 1
_MAX_PENDING_WRITES = 128


class BoundarySnapshotSSDStore:
    """Temporary SSD storage for non-sliceable cache layer snapshots.

    Stores ArraysCache/RotatingKVCache boundary snapshots to SSD during
    prefill to avoid GPU memory accumulation. Files are ephemeral and
    cleaned up when the request completes or aborts.

    Args:
        base_dir: Parent directory for SSD cache.
                  Snapshots stored under base_dir/_boundary_snapshots/.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir / "_boundary_snapshots"
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._pending_writes: dict[str, bytes] = {}
        self._pending_lock = threading.Lock()

        # Background writer
        self._write_queue: list[tuple[str, bytes]] = []
        self._write_lock = threading.Lock()
        self._writer_thread: threading.Thread | None = None
        self._shutdown = False

    def start(self) -> None:
        self._shutdown = False
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="boundary-writer",
        )
        self._writer_thread.start()

    def stop(self) -> None:
        # Flush any remaining writes before signalling shutdown so that
        # in-flight data is not silently dropped.
        self._flush_pending()
        self._shutdown = True
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

    def save(
        self,
        request_id: str,
        layer_index: int,
        cache_state: dict,
    ) -> Path:
        """Save a boundary snapshot for a non-sliceable layer.

        Serializes the cache state to bytes on the calling thread (Metal-safe),
        then queues for async disk write.

        Args:
            request_id: Unique request identifier.
            layer_index: Layer index in the model.
            cache_state: Dict from mlx_cache.extract_cache_state().

        Returns:
            Path to the snapshot file (may not be written yet).
        """
        key = f"{request_id}_L{layer_index}"
        data = self._serialize(cache_state)

        with self._pending_lock:
            self._pending_writes[key] = data

        # Queue for background write
        filename = f"{key}.bin"
        filepath = self._base_dir / filename

        with self._write_lock:
            self._write_queue.append((str(filepath), data))
            # Block if too many pending writes
            if len(self._write_queue) > _MAX_PENDING_WRITES:
                self._flush_pending_unlocked()

        return filepath

    def load(
        self,
        request_id: str,
        layer_index: int,
    ) -> dict | None:
        """Load a boundary snapshot from SSD.

        Checks in-memory buffer first, then disk.

        Args:
            request_id: Request identifier.
            layer_index: Layer index.

        Returns:
            Deserialized cache state dict, or None if not found.
        """
        key = f"{request_id}_L{layer_index}"

        # Check pending writes first (may not be flushed yet)
        with self._pending_lock:
            data = self._pending_writes.get(key)
            if data is not None:
                return self._deserialize(data)

        # Check disk
        filename = f"{key}.bin"
        filepath = self._base_dir / filename
        if filepath.exists():
            try:
                data = filepath.read_bytes()
                return self._deserialize(data)
            except Exception as e:
                logger.warning(f"Failed to load boundary snapshot {filepath}: {e}")
                return None

        return None

    def cleanup(self, request_id: str) -> None:
        """Remove all snapshots for a request (completion or abort)."""
        prefix = f"{request_id}_L"
        with self._pending_lock:
            keys_to_remove = [k for k in self._pending_writes if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._pending_writes[k]

        # Clean disk files
        try:
            for f in self._base_dir.glob(f"{prefix}*.bin"):
                f.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Error cleaning boundary snapshots for {request_id}: {e}")

    def _serialize(self, cache_state: dict) -> bytes:
        """Serialize cache state to binary format.

        Format:
            MAGIC (8 bytes)
            version (4 bytes, uint32 LE)
            num_entries (4 bytes, uint32 LE)
            for each entry:
                key_len (4 bytes) + key (UTF-8)
                dtype_len (4 bytes) + dtype string (UTF-8)
                shape_len (4 bytes) + shape (JSON)
                data_len (4 bytes) + data (raw bytes)
        """
        parts = [_MAGIC]
        entries = []

        for key, value in cache_state.items():
            if key == "cache_type":
                # Serialize CacheType enum as string
                ct = value
                type_name = ct.name if hasattr(ct, "name") else str(ct)
                entries.append((key, "string", [], type_name.encode("utf-8")))
            elif isinstance(value, np.ndarray):
                entries.append((key, str(value.dtype), list(value.shape), value.tobytes()))
            elif hasattr(value, "shape"):
                # MLX array
                arr = np.array(value, copy=False)
                entries.append((key, str(arr.dtype), list(arr.shape), arr.tobytes()))
            elif isinstance(value, bool):
                entries.append((key, "bool", [], struct.pack("<?", value)))
            elif isinstance(value, float):
                entries.append((key, "float64", [], struct.pack("<d", value)))
            elif isinstance(value, int):
                entries.append((key, "int64", [], struct.pack("<q", value)))
            elif isinstance(value, str):
                entries.append((key, "string", [], value.encode("utf-8")))

        parts.append(struct.pack("<II", _HEADER_VERSION, len(entries)))

        for key, dtype, shape, raw_data in entries:
            key_bytes = key.encode("utf-8")
            dtype_bytes = dtype.encode("utf-8")
            shape_json = json.dumps(shape).encode("utf-8")
            parts.append(struct.pack("<I", len(key_bytes)))
            parts.append(key_bytes)
            parts.append(struct.pack("<I", len(dtype_bytes)))
            parts.append(dtype_bytes)
            parts.append(struct.pack("<I", len(shape_json)))
            parts.append(shape_json)
            parts.append(struct.pack("<I", len(raw_data)))
            parts.append(raw_data)

        return b"".join(parts)

    def _deserialize(self, data: bytes) -> dict:
        """Deserialize binary format back to cache state dict."""
        offset = 0

        magic = data[offset:offset + 8]
        if magic != _MAGIC:
            raise ValueError(f"Invalid boundary snapshot magic: {magic}")
        offset += 8

        version, num_entries = struct.unpack_from("<II", data, offset)
        offset += 8

        result = {}
        for _ in range(num_entries):
            key_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            key = data[offset:offset + key_len].decode("utf-8")
            offset += key_len

            dtype_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            dtype = data[offset:offset + dtype_len].decode("utf-8")
            offset += dtype_len

            shape_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            shape = json.loads(data[offset:offset + shape_len].decode("utf-8"))
            offset += shape_len

            data_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            raw = data[offset:offset + data_len]
            offset += data_len

            if dtype == "string":
                result[key] = raw.decode("utf-8")
            elif dtype == "bool":
                result[key] = bool(struct.unpack("<?", raw)[0])
            elif dtype == "int64":
                result[key] = struct.unpack("<q", raw)[0]
            elif dtype == "float64":
                result[key] = struct.unpack("<d", raw)[0]
            elif dtype == "number":
                # Legacy format: disambiguate by data length
                if data_len == 8:
                    result[key] = struct.unpack("<q", raw)[0]
                else:
                    result[key] = struct.unpack("<d", raw)[0]
            elif isinstance(shape, list):
                # ndarray entry (including 0-D arrays with shape=[])
                np_dtype = np.dtype(dtype)
                if shape:
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
                else:
                    # 0-D scalar array
                    arr = np.frombuffer(raw, dtype=np_dtype).copy()
                result[key] = arr

        return result

    def _writer_loop(self) -> None:
        """Background thread that flushes pending writes to disk."""
        while not self._shutdown:
            items = []
            with self._write_lock:
                items = self._write_queue[:]
                self._write_queue.clear()

            if items:
                for filepath_str, data in items:
                    try:
                        # Write to temp file first, then atomically rename to
                        # prevent torn reads from a concurrent load().
                        tmp_path = filepath_str + ".tmp"
                        Path(tmp_path).write_bytes(data)
                        os.replace(tmp_path, filepath_str)
                    except Exception as e:
                        logger.warning(f"Boundary snapshot write failed for {filepath_str}: {e}")
            else:
                time.sleep(0.05)

    def _flush_pending_unlocked(self) -> None:
        """Flush all pending writes synchronously. Caller MUST hold _write_lock."""
        items = self._write_queue[:]
        self._write_queue.clear()

        for filepath_str, data in items:
            try:
                # Write to temp file first, then atomically rename to
                # prevent torn reads from a concurrent load().
                tmp_path = filepath_str + ".tmp"
                Path(tmp_path).write_bytes(data)
                os.replace(tmp_path, filepath_str)
            except Exception as e:
                logger.warning(f"Boundary snapshot flush failed: {e}")

    def _flush_pending(self) -> None:
        """Flush all pending writes synchronously (acquires lock)."""
        with self._write_lock:
            self._flush_pending_unlocked()
