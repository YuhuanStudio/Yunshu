"""Parallel row reads for the qwen4_exp external PLE table (SSD-resident n-gram embedding).

Upstream ``QuantizedMMapNGramEmbedding._read_rows`` gathers each missing row with a NumPy fancy index on
a memmap, one (shard, tensor) at a time: a token touches 16 head rows x 3 tensors = 48 serial page
faults (~150 us each on the SSD, ~7 ms per token measured on M5 Max). Rows are random, so a repeated
prompt hides this through the row LRU while fresh text pays it on every step.

This replaces ``_read_rows`` with ``pread`` calls issued from a thread pool (the GIL is released
inside the syscall), so the faults overlap: for the ``safetensors_ranges`` layout three reads per row
(weight, scales, biases sit in different tensors), for the ``interleaved_q4`` layout (one record per
row, ``YUNSHU_PLE_STORAGE=ssd-rows``) one. The bytes read, the returned arrays, the row cache and every
counter are identical to upstream's; only the order in which the reads happen differs.
"""

# Patches upstream (MIT): mlx-vlm qwen4_exp/ple_storage.py QuantizedMMapNGramEmbedding._read_rows
# (tracked in vendor.json; Blaizzy/mlx-vlm).
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_STATE: dict = {"installed": False, "original": None, "threads": 0}
_LOCK = threading.Lock()


def _fd(table, path: str) -> int:
    fds = getattr(table, "_yunshu_fds", None)
    if fds is None:
        fds = table._yunshu_fds = {}
    fd = fds.get(path)
    if fd is None:
        fd = fds[path] = os.open(path, os.O_RDONLY)
    return fd


def _pool(table, threads: int) -> ThreadPoolExecutor:
    pool = getattr(table, "_yunshu_pool", None)
    if pool is None:
        pool = table._yunshu_pool = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="ple-read"
        )
    return pool


def _read_chunk(chunk) -> None:
    for fd, offset, view in chunk:
        n = os.preadv(fd, [view], offset)
        if n != len(view):
            raise OSError(f"short PLE row read: {n} of {len(view)} bytes at {offset}")


def parallel_read_rows(table, row_ids, threads: int, dtypes: dict):
    """Drop-in for ``_read_rows`` (ranges layout). ``dtypes`` maps the layout's dtype names to numpy."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.ndim != 1:
        raise ValueError("row_ids must be one-dimensional")
    names = table._tensor_names
    n = len(row_ids)
    layout = dict(table._tensor_layout)
    first = table._shards[0][2]
    out = {
        name: np.empty((n, first[name].shape[1]), dtype=dtypes[layout[name]])
        for name in names
    }
    missing_positions: list[int] = []
    missing_ids: list[int] = []
    for position, row_id in enumerate(row_ids.tolist()):
        cached = table._cache.get(row_id)
        if cached is None:
            missing_positions.append(position)
            missing_ids.append(row_id)
            continue
        table._hits += 1
        table._cache.move_to_end(row_id)
        for index, name in enumerate(names):
            out[name][position] = cached[index]

    if missing_ids:
        shards = table._shards
        starts = np.asarray([s[0] for s in shards], dtype=np.int64)
        flat = []
        for position, row_id in zip(missing_positions, missing_ids, strict=True):
            shard_index = int(np.searchsorted(starts, row_id, side="right")) - 1
            start, end, arrays = shards[shard_index]
            if shard_index < 0 or not (start <= row_id < end):
                raise IndexError("n-gram row outside mmap table")
            local = row_id - start
            for name in names:
                mm = arrays[name]
                row_bytes = mm.shape[1] * mm.dtype.itemsize
                view = memoryview(out[name][position].view(np.uint8))
                flat.append(
                    (
                        _fd(table, str(mm.filename)),
                        int(mm.offset) + local * row_bytes,
                        view,
                    )
                )
        lanes = max(1, min(threads, len(flat)))
        chunks = [flat[i::lanes] for i in range(lanes)]
        futures = [
            _pool(table, threads).submit(_read_chunk, chunk) for chunk in chunks[1:]
        ]
        _read_chunk(chunks[0])
        for future in futures:
            future.result()
        table._bytes += sum(out[name][0].nbytes for name in names) * len(missing_ids)

    table._misses += len(missing_ids)
    for position, row_id in zip(missing_positions, missing_ids, strict=True):
        if table.cache_rows:
            table._cache[row_id] = tuple(
                np.array(out[name][position], copy=True) for name in names
            )
            table._cache.move_to_end(row_id)
            while len(table._cache) > table.cache_rows:
                table._cache.popitem(last=False)
    return tuple(out[name] for name in names)


def interleaved_columns(table, dtypes: dict, raw: np.ndarray) -> list:
    """Split ``raw`` [n, row_stride] interleaved records into the table's tensors, as upstream's
    ``_read_rows`` does for the ``interleaved_q4`` layout."""
    packed = table.row_width // 2
    columns, offset = [], 0
    for name, dtype_name in table._tensor_layout:
        width = (
            packed
            if name == "weight"
            else table.row_width // table._group_size * dtypes[dtype_name].itemsize
        )
        columns.append(
            raw[:, offset : offset + width]
            .reshape(len(raw), -1)
            .view(dtypes[dtype_name])
        )
        offset += width
    return columns


def parallel_read_rows_interleaved(table, row_ids, threads: int, dtypes: dict):
    """``_read_rows`` for the ``interleaved_q4`` layout: one ``pread`` of a row's whole record
    (weight, scales and biases together) instead of three, from a thread pool. Same bytes, counters
    and row cache as upstream's fancy index on the memmap."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.ndim != 1:
        raise ValueError("row_ids must be one-dimensional")
    store = table._interleaved
    stride = int(store.shape[1])
    n = len(row_ids)
    cached_rows: dict[int, tuple] = {}
    missing_positions: list[int] = []
    missing_ids: list[int] = []
    for position, row_id in enumerate(row_ids.tolist()):
        cached = table._cache.get(row_id)
        if cached is None:
            missing_positions.append(position)
            missing_ids.append(row_id)
            continue
        table._hits += 1
        table._cache.move_to_end(row_id)
        cached_rows[position] = cached
    raw = np.empty((len(missing_ids), stride), dtype=np.uint8)
    if missing_ids:
        fd = _fd(table, str(store.filename))
        base = int(store.offset)
        flat = [
            (fd, base + row_id * stride, memoryview(raw[i]))
            for i, row_id in enumerate(missing_ids)
        ]
        lanes = max(1, min(threads, len(flat)))
        chunks = [flat[i::lanes] for i in range(lanes)]
        futures = [
            _pool(table, threads).submit(_read_chunk, chunk) for chunk in chunks[1:]
        ]
        _read_chunk(chunks[0])
        for future in futures:
            future.result()
        table._bytes += raw.nbytes
    table._misses += len(missing_ids)
    loaded = interleaved_columns(table, dtypes, raw)
    out = [np.empty((n, column.shape[1]), dtype=column.dtype) for column in loaded]
    for k, column in enumerate(loaded):
        if missing_positions:
            out[k][missing_positions] = column
        for position, row in cached_rows.items():
            out[k][position] = row[k]
    if table.cache_rows:
        for i, row_id in enumerate(missing_ids):
            table._cache[row_id] = tuple(np.array(c[i], copy=True) for c in loaded)
            table._cache.move_to_end(row_id)
            while len(table._cache) > table.cache_rows:
                table._cache.popitem(last=False)
    return tuple(out)


def install(threads: int) -> bool:
    """Patch upstream's ``_read_rows`` to read in parallel; 0 restores upstream. True if active."""
    try:
        from .qwen4_family import qwen4_first_module

        ple_storage = qwen4_first_module("ple_storage")
    except Exception:  # noqa: BLE001 - mlx_vlm without a Qwen4 module
        return False
    if ple_storage is None:
        return False
    cls = ple_storage.QuantizedMMapNGramEmbedding
    with _LOCK:
        if _STATE["original"] is None:
            _STATE["original"] = cls._read_rows
        original = _STATE["original"]
        if threads <= 0:
            cls._read_rows = original
            _STATE.update(installed=False, threads=0)
            return False
        dtypes = ple_storage._DTYPES

        def _read_rows(self, row_ids):
            if not len(row_ids):
                return original(self, row_ids)
            if self._interleaved is not None:
                return parallel_read_rows_interleaved(self, row_ids, threads, dtypes)
            if not self._shards:
                return original(self, row_ids)
            return parallel_read_rows(self, row_ids, threads, dtypes)

        cls._read_rows = _read_rows
        _STATE.update(installed=True, threads=threads)
        return True


def active_threads() -> int:
    return int(_STATE["threads"]) if _STATE["installed"] else 0
