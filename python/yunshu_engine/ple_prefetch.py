"""Background prefetch of the next prompt span's PLE rows (SSD-resident n-gram table).

A Flash-Next prompt span reads its n-gram embedding rows (``ple_ssd``, ``ple_parallel``) on the MLX
thread while the graph is built, before any GPU work of that span is dispatched, so each span's
SSD reads sit in front of its own compute. The driver knows the whole prompt, so the rows of span
k+1 can be read on a background thread while span k runs on the GPU (the driver evaluates each
span before building the next one; see ``round_driver.driver``).

The prefetch only ever holds rows keyed by row id, and a row id always maps to the same bytes, so a
prefetched row is byte-identical to a synchronous read. A wrong prediction (for instance a restored
prompt whose context window differs) costs a synchronous read, never a different output. Counters
(``_misses``, ``_bytes``) are updated on the MLX thread only and match a run without prefetch.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import settings

logger = logging.getLogger(__name__)

MAX_PENDING = 4  # spans read ahead at most, over all prompts (the bounded buffer)


class PrefetchCancelledError(Exception):
    """A background row read was cancelled before it finished."""


@dataclass(frozen=True)
class NGramParams:
    """The arrays ``Qwen4ExpNGramEmbedding.__call__`` derives its row ids from."""

    ngram_size: int
    heads_per_ngram: int
    context_len: int
    eos: int
    multipliers: np.ndarray  # int64 [ngram_size]
    sizes: np.ndarray  # int64 [ngram_heads]
    offsets: np.ndarray  # int64 [ngram_heads]


def params_from_module(mod) -> NGramParams:
    """Host copies of an ``Qwen4ExpNGramEmbedding``'s id parameters (call on the MLX thread)."""
    return NGramParams(
        ngram_size=int(mod.ngram_size),
        heads_per_ngram=int(mod.heads_per_ngram),
        context_len=int(mod.context_len),
        eos=int(mod.eos_token_id),
        multipliers=np.array(mod.layer_multipliers, dtype=np.int64),
        sizes=np.array(mod.ngram_heads_vocab_sizes, dtype=np.int64),
        offsets=np.array(mod.ngram_heads_offsets, dtype=np.int64),
    )


def _shift_right_ignore_eos(tokens: np.ndarray, shift: int, eos: int) -> np.ndarray:
    """1-D twin of ``Qwen4ExpNGramEmbedding._shift_right_ignore_eos``."""
    if shift == 0:
        return tokens
    pos = np.arange(len(tokens), dtype=np.int64)
    eos_pos = np.where(tokens == eos, pos, -1)
    prev_eos = np.concatenate([[-1], np.maximum.accumulate(eos_pos)[:-1]])
    pos_in_segment = pos - (prev_eos + 1)
    source = pos - shift
    shifted = tokens[np.maximum(source, 0)]
    valid = (pos_in_segment >= shift) & (source >= 0)
    return np.where(valid, shifted, eos)


def span_row_ids(history: np.ndarray, params: NGramParams) -> np.ndarray:
    """Row ids ``[T, ngram_heads]`` of a span; ``history`` is its ``context_len`` previous tokens
    followed by the span's ``T`` tokens (the model's ``token_history``)."""
    history = np.asarray(history, dtype=np.int64)
    span = len(history) - params.context_len
    shifted = [
        _shift_right_ignore_eos(history, s, params.eos)
        for s in range(params.ngram_size)
    ]
    blocks = []
    for ngram in range(2, params.ngram_size + 1):
        start = (ngram - 2) * params.heads_per_ngram
        end = start + params.heads_per_ngram
        mixed = shifted[0] * params.multipliers[0]
        for position in range(1, ngram):
            mixed = np.bitwise_xor(
                mixed, shifted[position] * params.multipliers[position]
            )
        sizes = params.sizes[start:end]
        offsets = params.offsets[start:end]
        blocks.append(mixed[:, None] % sizes[None] + offsets[None])
    return np.concatenate(blocks, axis=1)[-span:]


class _RowReader:
    """Reads table rows with ``pread`` and no shared state (safe on a background thread)."""

    def __init__(self, table):
        self._table = table
        self._lock = threading.Lock()
        self._fds: dict[str, int] = {}
        self._module = sys.modules[type(table).__module__]  # ple_storage (``_DTYPES``)

    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = self._fds[path] = os.open(path, os.O_RDONLY)
            return fd

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()

    def read(
        self,
        row_ids: np.ndarray,
        cancel: threading.Event,
        pool=None,
        slices: int = 1,
        gate=None,
    ) -> tuple:
        """All rows of ``row_ids``; with ``pool`` the id range is split in ``slices`` parallel parts
        (``pread`` drops the GIL), as ``ple_parallel`` reads them on the foreground path."""
        table = self._table
        if table._interleaved is not None:
            return self._read_interleaved(row_ids, cancel, pool, slices, gate)
        names = table._tensor_names
        layout = dict(table._tensor_layout)
        dtypes = self._module._DTYPES
        shards = table._shards
        first = shards[0][2]
        out = {
            name: np.empty(
                (len(row_ids), first[name].shape[1]), dtype=dtypes[layout[name]]
            )
            for name in names
        }
        starts = np.asarray([s[0] for s in shards], dtype=np.int64)

        def part(lo: int, hi: int) -> None:
            self._fill(out, names, shards, starts, row_ids, lo, hi, cancel, gate)

        n = len(row_ids)
        if pool is None or slices <= 1 or n < 2 * slices:
            part(0, n)
        else:
            bounds = np.linspace(0, n, slices + 1, dtype=np.int64).tolist()
            futures = [
                pool.submit(part, a, b)
                for a, b in zip(bounds, bounds[1:], strict=False)
            ]
            for f in futures:
                f.result()
        return tuple(out[name] for name in names)

    def _read_interleaved(self, row_ids, cancel, pool, slices, gate) -> tuple:
        """``read`` for the ``interleaved_q4`` layout: one ``pread`` per row record."""
        from .ple_parallel import interleaved_columns

        store = self._table._interleaved
        stride = int(store.shape[1])
        raw = np.empty((len(row_ids), stride), dtype=np.uint8)
        fd = self._fd(str(store.filename))
        base = int(store.offset)

        def part(lo: int, hi: int) -> None:
            for position in range(lo, hi):
                if position % 16 == 0:
                    if gate is not None:
                        gate()
                    if cancel.is_set():
                        raise PrefetchCancelledError()
                offset = base + int(row_ids[position]) * stride
                n = os.preadv(fd, [memoryview(raw[position])], offset)
                if n != stride:
                    raise OSError(
                        f"short PLE row read: {n} of {stride} bytes at {offset}"
                    )

        n = len(row_ids)
        if pool is None or slices <= 1 or n < 2 * slices:
            part(0, n)
        else:
            bounds = np.linspace(0, n, slices + 1, dtype=np.int64).tolist()
            futures = [
                pool.submit(part, a, b)
                for a, b in zip(bounds, bounds[1:], strict=False)
            ]
            for f in futures:
                f.result()
        return tuple(interleaved_columns(self._table, self._module._DTYPES, raw))

    def _fill(
        self, out, names, shards, starts, row_ids, lo, hi, cancel, gate=None
    ) -> None:
        for position in range(lo, hi):
            row_id = int(row_ids[position])
            if position % 16 == 0:
                if gate is not None:
                    gate()
                if cancel.is_set():
                    raise PrefetchCancelledError()
            shard_index = int(np.searchsorted(starts, row_id, side="right")) - 1
            start, end, arrays = shards[shard_index]
            if shard_index < 0 or not (start <= row_id < end):
                raise IndexError("n-gram row outside mmap table")
            local = row_id - start
            for name in names:
                mm = arrays[name]
                row_bytes = mm.shape[1] * mm.dtype.itemsize
                view = memoryview(out[name][position].view(np.uint8))
                offset = int(mm.offset) + local * row_bytes
                n = os.preadv(self._fd(str(mm.filename)), [view], offset)
                if n != len(view):
                    raise OSError(
                        f"short PLE row read: {n} of {len(view)} bytes at {offset}"
                    )


@dataclass
class _Entry:
    key: tuple  # (prompt length, span start, span end)
    row_ids: np.ndarray  # sorted, unique
    future: Future
    cancel: threading.Event


class PLEPrefetch:
    """Reads upcoming spans' rows on a background thread; serves them to ``_read_rows``.

    The driver calls ``begin`` for the span it is about to build (that span's entry, if any, answers
    the table's reads) and ``lookahead`` for the spans that follow; the reads then run while the
    GPU evaluates the step."""

    def __init__(self, table, params: NGramParams):
        self._table = table
        self._params = params
        self._reader = _RowReader(table)
        self._threads = max(1, int(settings.get("YUNSHU_PLE_READ_THREADS")) or 1)
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple-prefetch"
        )
        self._workers = ThreadPoolExecutor(
            max_workers=self._threads, thread_name_prefix="ple-prefetch-io"
        )
        self._pending: deque[_Entry] = deque()
        self._current: _Entry | None = None
        # row reads the MLX thread is blocked on; other spans' background reads wait for them
        self._foreground = 0
        self._idle = threading.Condition()
        self._blocked_s, self._asked, self._served = 0.0, 0, 0
        self.before = (0.0, 0)
        self._original = table._read_rows
        self._closed = False
        table._read_rows = (
            self._read_rows
        )  # instance attribute: shadows the class method
        table._yunshu_prefetch = self

    # ── producer (MLX thread) ─────────────────────────────────────────────
    def submit_span(self, ids: list[int], start: int, end: int) -> bool:
        """Start reading the rows of prompt tokens ``ids[start:end]``; False if closed or already
        pending. A full buffer drops the oldest entry."""
        if self._closed:
            return False
        key = (len(ids), start, end)
        if any(e.key == key for e in self._pending):
            return False
        ctx = self._params.context_len
        prev = list(ids[max(0, start - ctx) : start])
        prev = [self._params.eos] * (ctx - len(prev)) + prev
        history = np.asarray(prev + list(ids[start:end]), dtype=np.int64)
        row_ids = np.unique(span_row_ids(history, self._params))
        if row_ids.size and (row_ids[0] < 0 or row_ids[-1] >= self._table.row_count):
            raise IndexError("n-gram row outside mmap table")
        while len(self._pending) >= MAX_PENDING:
            self._drop(self._pending.popleft())
        entry = _Entry(key, row_ids, Future(), threading.Event())
        self._pool.submit(self._run, entry)
        self._pending.append(entry)
        return True

    def begin(self, n_ids: int, start: int, end: int) -> None:
        """The span ``start:end`` of an ``n_ids`` prompt is about to be built: its entry (if any)
        serves the table's reads; earlier spans of that prompt are dropped (other prompts' entries
        wait for their own turn and are evicted oldest-first by ``submit_span``)."""
        self._current = None
        # what was read since the previous span (decode steps of the last request) stays visible
        self.before = (self._blocked_s, self._asked)
        self._blocked_s, self._asked, self._served = 0.0, 0, 0
        for entry in list(self._pending):
            if entry.key == (n_ids, start, end):
                self._current = entry
                self._pending.remove(entry)
            elif entry.key[0] == n_ids and entry.key[1] < start:
                self._pending.remove(entry)
                self._drop(entry)

    def _run(self, entry: _Entry) -> None:
        if not entry.future.set_running_or_notify_cancel():
            return
        try:
            entry.future.set_result(
                self._reader.read(
                    entry.row_ids,
                    entry.cancel,
                    self._workers,
                    self._threads,
                    gate=lambda: self._yield(entry),
                )
            )
        except BaseException as exc:  # noqa: BLE001 - delivered to the consumer
            entry.future.set_exception(exc)

    def _yield(self, entry: _Entry) -> None:
        """A background read waits while the MLX thread reads rows it needs now (a decode step's
        rows, or a span with no prefetch); the span being built never waits on itself."""
        with self._idle:
            while (
                self._foreground
                and entry is not self._current
                and not entry.cancel.is_set()
            ):
                self._idle.wait(0.05)

    @staticmethod
    def _drop(entry: _Entry) -> None:
        entry.cancel.set()

    # ── consumer (MLX thread): the table's ``_read_rows`` ────────────────
    def take_stats(self) -> tuple[float, int, int]:
        """(seconds the MLX thread was blocked in row reads, rows asked, rows served by a prefetch)
        since the last call."""
        out = (self._blocked_s, self._asked, self._served)
        self._blocked_s, self._asked, self._served = 0.0, 0, 0
        return out

    def _read_rows(self, row_ids):
        t0 = time.perf_counter()
        with self._idle:
            self._foreground += 1
        try:
            return self._read_rows_inner(row_ids)
        finally:
            with self._idle:
                self._foreground -= 1
                self._idle.notify_all()
            self._blocked_s += time.perf_counter() - t0

    def _read_rows_inner(self, row_ids):
        row_ids = np.asarray(row_ids, dtype=np.int64)
        self._asked += len(row_ids)
        entry = self._current
        if not row_ids.size or entry is None:
            return self._original(row_ids)
        table = self._table
        hit = np.isin(row_ids, entry.row_ids)
        if not hit.any():
            return self._original(row_ids)
        arrays = entry.future.result()  # fail closed: a read error reaches the caller
        pos = np.searchsorted(entry.row_ids, row_ids[hit])
        out = [np.empty((len(row_ids), a.shape[1]), dtype=a.dtype) for a in arrays]
        for k, a in enumerate(arrays):
            out[k][hit] = a[pos]
        missing = np.flatnonzero(~hit)
        if missing.size:
            read = self._original(row_ids[missing])
            for k in range(len(out)):
                out[k][missing] = read[k]
        # counters as upstream keeps them: every served row is a miss (the table has no row cache here)
        served_count = len(row_ids) - missing.size
        self._served += served_count
        table._misses += served_count
        table._bytes += served_count * sum(a[0].nbytes for a in out)
        return tuple(out)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def cancel(self) -> None:
        """Stop every pending read (a running one exits at its next check)."""
        while self._pending:
            self._drop(self._pending.popleft())
        self._current = None

    def close(self) -> None:
        if self._closed:
            return
        self.cancel()
        self._closed = True
        self._pool.shutdown(wait=True)
        self._workers.shutdown(wait=True)
        self._reader.close()
        if self._table.__dict__.get("_read_rows") == self._read_rows:
            del self._table.__dict__["_read_rows"]
        self._table.__dict__.pop("_yunshu_prefetch", None)


def _ple_module(lm):
    for layer in lm.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is not None:
            return ple.ple_embedding
    return None


def attach(lm) -> PLEPrefetch | None:
    """The prefetch for ``lm``'s PLE table, or None when it does not apply (memory table, a
    row cache, no PLE layer). Call on the MLX thread."""
    mod = _ple_module(lm)
    if mod is None:
        return None
    table = mod.ngram_embedding
    if (
        not getattr(table, "_shards", None)
        and getattr(table, "_interleaved", None) is None
    ) or table.cache_rows:
        return None
    existing: PLEPrefetch | None = table.__dict__.get("_yunshu_prefetch")
    if existing is not None:
        return existing
    return PLEPrefetch(table, params_from_module(mod))
