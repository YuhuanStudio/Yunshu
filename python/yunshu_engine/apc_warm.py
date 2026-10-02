"""WARM tier of the VLM runner's prefix cache: a compact in-RAM form of an exact checkpoint.

HOT holds ready-to-use arrays (``YunshuAPCManager._exact_cache``). WARM holds the same checkpoint in
a smaller form that has to be decoded before use; on unified memory this is a format difference, not
another memory pool. Hierarchy: HOT -> WARM -> SSD -> recompute. A demoted checkpoint moves HOT to
WARM; a hit decodes it and promotes it back to HOT.

Modes (``YUNSHU_VLM_APC_WARM``):

``lossless``  every array is compressed (zstd, optionally after a byte-plane shuffle of 2-byte
              floats); decode is bit-exact, so a WARM hit equals a HOT hit token for token. Compression
              runs on a worker thread; the checkpoint stays promotable (as HOT) while it runs.
``int8`` / ``int4``  attention K/V are affine-quantized per group (``mx.quantize``); recurrent state
              stays exact. LOSSY: a hit restores the dequantized K/V, so the output may differ from a
              cold prefill. The SSD tier keeps exact states in this mode (an entry leaves HOT for the
              SSD as exact bytes at the same moment it enters WARM), and a WARM entry that is evicted
              is dropped, never written to SSD.

Everything that touches MLX arrays runs on the caller's thread (the runner's single GPU thread); the
worker only compresses NumPy buffers.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

MODES = ("off", "lossless", "int8", "int4")
CHUNK = 4 << 20
ZSTD_LEVEL = 1
# One entry compresses at a time: its HOT arrays stay pinned until the encode finishes, so this
# bounds the transient memory above the HOT + WARM budgets to one checkpoint.
_MAX_INFLIGHT = 1

_NP_DTYPE = {
    "mlx.core.bfloat16": (np.uint16, mx.bfloat16),
    "mlx.core.float16": (np.float16, None),
    "mlx.core.float32": (np.float32, None),
    "mlx.core.int8": (np.int8, None),
    "mlx.core.uint8": (np.uint8, None),
    "mlx.core.int32": (np.int32, None),
    "mlx.core.uint32": (np.uint32, None),
    "mlx.core.int64": (np.int64, None),
}


class WarmCorruptError(Exception):
    """A WARM entry failed its integrity check (the lookup falls through to SSD / recompute)."""


@dataclass
class _Blob:
    """One array, compressed in 4 MiB chunks."""

    shape: tuple
    dtype: str
    chunks: list[bytes]
    raw_bytes: int
    planes: bool

    @property
    def nbytes(self) -> int:
        return sum(len(c) for c in self.chunks)


@dataclass
class _Quant:
    """One attention K or V array, affine-quantized (lossy)."""

    shape: tuple
    dtype: Any
    codes: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group: int

    @property
    def nbytes(self) -> int:
        return int(self.codes.nbytes + self.scales.nbytes + self.biases.nbytes)


@dataclass
class _Layer:
    cls: type
    meta: Any
    is_tuple: bool
    items: list  # _Blob | _Quant | mx.array | None


@dataclass
class WarmEntry:
    token_ids: tuple
    extra_hash: int
    layers: list[_Layer]
    nbytes: int
    raw_nbytes: int
    lossy: bool
    born: int = 0
    head: bool = False


@dataclass
class _Inflight:
    entry_tokens: tuple
    extra_hash: int
    prompt_cache: (
        list  # the HOT arrays, kept until the encode finishes (promotable as-is)
    )
    raw_nbytes: int
    born: int
    head: bool
    future: Future
    cancelled: bool = False


@dataclass
class WarmStats:
    demotions: int = 0
    rejected: int = 0
    hits: int = 0
    hit_tokens: int = 0
    corrupt: int = 0
    evicted_to_ssd: int = 0
    dropped: int = 0
    encode_s: float = 0.0
    decode_s: float = 0.0
    raw_bytes_in: int = 0
    stored_bytes_in: int = 0
    inflight_hits: int = 0


def _layer_arrays(c: Any):
    """(is_tuple, [arrays]) of one cache layer through the state contract, or None."""
    if not (hasattr(c, "state") and hasattr(c, "meta_state")):
        return None
    if callable(getattr(c, "is_single_row", None)) or hasattr(c, "caches"):
        return None  # batched / composite layers are not stored by the exact APC either
    st = c.state
    if isinstance(st, tuple):
        return True, list(st)
    if isinstance(st, list):
        return False, list(st)
    return None


def prompt_cache_nbytes(prompt_cache) -> int:
    total = 0
    for c in prompt_cache:
        got = _layer_arrays(c)
        if got is None:
            continue
        total += sum(a.nbytes for a in got[1] if a is not None)
    return total


def _is_kv(c: Any) -> bool:
    return type(c).__name__ == "KVCache"


class WarmTier:
    """Budgeted store of compact checkpoints. All methods run on the caller's (GPU) thread except
    the compression jobs on the internal worker; call :meth:`drain` before reading."""

    def __init__(
        self,
        mode: str,
        budget_bytes: int,
        *,
        threads: int = 8,
        level: int = ZSTD_LEVEL,
        shuffle: bool = True,
        group: int = 32,
    ):
        if mode not in MODES or mode == "off":
            raise ValueError(
                f"WarmTier mode must be lossless / int8 / int4, got {mode!r}"
            )
        self.mode = mode
        self.lossy = mode != "lossless"
        self.budget_bytes = int(budget_bytes)
        self.level = level
        self.shuffle = shuffle
        self.group = group
        self.bits = {"int8": 8, "int4": 4}.get(mode, 0)
        self.stats = WarmStats()
        self.entries: collections.OrderedDict[int, WarmEntry] = (
            collections.OrderedDict()
        )
        self.bytes = 0
        self._inflight: collections.OrderedDict[int, _Inflight] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self._overflow: list[tuple[int, WarmEntry]] = []
        self._failed: dict[int, _Inflight] = {}
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="apc-warm")
        self._chunk_pool = ThreadPoolExecutor(
            max(1, threads), thread_name_prefix="apc-warm-c"
        )
        self._zstd: Any = None
        if not self.lossy:
            import zstandard  # the "compression" extra; the lossy modes do not need it

            self._zstd = zstandard

    # ── encode ─────────────────────────────────────────────────────────
    def _compress_buffer(self, buf: np.ndarray, plane: bool) -> list[bytes]:
        if plane:
            u = buf.reshape(-1, 2)
            buf = np.concatenate([u[:, 0], u[:, 1]])
        parts = [buf[i : i + CHUNK] for i in range(0, buf.size, CHUNK)]
        return list(self._chunk_pool.map(self._compress_chunk, parts))

    def _compress_chunk(self, part: np.ndarray) -> bytes:
        # a ZstdCompressor is not thread-safe: one per chunk
        zc = self._zstd.ZstdCompressor(level=self.level, write_checksum=True)
        return bytes(zc.compress(part.tobytes()))

    def _decompress_chunk(self, chunk: bytes) -> bytes:
        return bytes(self._zstd.ZstdDecompressor().decompress(chunk))

    def _encode_lossless(self, staged: list) -> tuple[list[_Layer], int]:
        """``staged``: per layer (cls, meta, is_tuple, [(shape, dtype_name, evaluated array)])."""
        layers, total = [], 0
        for cls, meta, is_tuple, arrs in staged:
            items: list = []
            for item in arrs:
                if item is None:
                    items.append(None)
                    continue
                shape, dname, arr = item
                np_dt = _NP_DTYPE[dname][0]
                buf = np.array(arr, copy=False).reshape(-1)
                if buf.dtype != np_dt:
                    buf = buf.view(np_dt)
                raw = buf.view(np.uint8)
                plane = self.shuffle and buf.dtype.itemsize == 2
                blob = _Blob(
                    shape, dname, self._compress_buffer(raw, plane), raw.size, plane
                )
                total += blob.nbytes
                items.append(blob)
            layers.append(_Layer(cls, meta, is_tuple, items))
        return layers, total

    def _stage(self, prompt_cache) -> list | None:
        """Evaluate the arrays (as uint16 views for bf16) on the caller's thread; None when a
        layer cannot be stored."""
        staged, to_eval = [], []
        for c in prompt_cache:
            got = _layer_arrays(c)
            if got is None:
                return None
            is_tuple, arrs = got
            rows: list[Any] = []
            for a in arrs:
                if a is None:
                    rows.append(None)
                    continue
                dname = str(a.dtype)
                if dname not in _NP_DTYPE:
                    return None
                view = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
                to_eval.append(view)
                rows.append((tuple(a.shape), dname, view))
            staged.append((type(c), c.meta_state, is_tuple, rows))
        if to_eval:
            mx.eval(*to_eval)
        return staged

    def _quantize(self, prompt_cache) -> tuple[list[_Layer], int] | None:
        layers, total, to_eval = [], 0, []
        for c in prompt_cache:
            got = _layer_arrays(c)
            if got is None:
                return None
            is_tuple, arrs = got
            items: list = []
            for a in arrs:
                if a is None:
                    items.append(None)
                elif _is_kv(c) and a.ndim == 4 and a.shape[-1] % self.group == 0:
                    codes, scales, biases = mx.quantize(
                        a, group_size=self.group, bits=self.bits
                    )
                    q = _Quant(
                        tuple(a.shape),
                        a.dtype,
                        codes,
                        scales,
                        biases,
                        self.bits,
                        self.group,
                    )
                    to_eval += [codes, scales, biases]
                    items.append(q)
                    total += q.nbytes
                else:
                    cp = mx.contiguous(mx.array(a))
                    to_eval.append(cp)
                    items.append(cp)
                    total += cp.nbytes
            layers.append(_Layer(type(c), c.meta_state, is_tuple, items))
        if to_eval:
            mx.eval(*to_eval)
        return layers, total

    def demote(self, key: int, entry, *, born: int = 0, head: bool = False) -> bool:
        """Take an evicted HOT entry (``APCExactCacheEntry``). False when WARM cannot hold it (the
        caller then spills it to SSD as it did before WARM existed)."""
        raw = prompt_cache_nbytes(entry.prompt_cache)
        if raw <= 0 or raw > self.budget_bytes * 4:
            self.stats.rejected += 1
            return False
        t0 = time.perf_counter()
        if self.lossy:
            got = self._quantize(entry.prompt_cache)
            if got is None:
                self.stats.rejected += 1
                return False
            layers, nbytes = got
            if nbytes > self.budget_bytes:
                self.stats.rejected += 1
                return False
            self.stats.encode_s += time.perf_counter() - t0
            self._insert(
                key,
                WarmEntry(
                    entry.token_ids,
                    entry.extra_hash,
                    layers,
                    nbytes,
                    raw,
                    True,
                    born,
                    head,
                ),
                raw,
            )
            self.stats.demotions += 1
            return True
        if len(self._inflight) >= _MAX_INFLIGHT:
            self.stats.rejected += 1
            return False
        staged = self._stage(entry.prompt_cache)
        if staged is None:
            self.stats.rejected += 1
            return False
        fut = self._worker.submit(self._encode_lossless, staged)
        with self._lock:
            self._inflight[key] = _Inflight(
                entry.token_ids,
                entry.extra_hash,
                entry.prompt_cache,
                raw,
                born,
                head,
                fut,
            )
        self.stats.demotions += 1
        return True

    # ── bookkeeping ────────────────────────────────────────────────────
    def _insert(
        self, key: int, ent: WarmEntry, raw: int
    ) -> list[tuple[int, WarmEntry]]:
        """Insert, evict the LRU overflow; returns the evicted (key, entry) list."""
        evicted: list[tuple[int, WarmEntry]] = []
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        self.entries[key] = ent
        self.bytes += ent.nbytes
        self.stats.raw_bytes_in += raw
        self.stats.stored_bytes_in += ent.nbytes
        while self.bytes > self.budget_bytes and len(self.entries) > 1:
            k, e = self.entries.popitem(last=False)
            self.bytes -= e.nbytes
            evicted.append((k, e))
        if self.bytes > self.budget_bytes:  # a single entry above the budget
            k, e = self.entries.popitem(last=False)
            self.bytes -= e.nbytes
            evicted.append((k, e))
        self._overflow.extend(evicted)
        return evicted

    def drain(self) -> list[tuple[int, WarmEntry]]:
        """Move finished encodes into WARM. Returns the entries evicted by budget (the caller
        decides whether they go to SSD)."""
        with self._lock:
            done = [k for k, f in self._inflight.items() if f.future.done()]
            items = [(k, self._inflight.pop(k)) for k in done]
        for key, inf in items:
            if inf.cancelled:
                continue
            try:
                layers, nbytes = inf.future.result()
            except Exception as e:
                logger.warning("APC warm: encode failed (%s)", e)
                self.stats.rejected += 1
                self._failed[key] = inf
                continue
            ent = WarmEntry(
                inf.entry_tokens,
                inf.extra_hash,
                layers,
                nbytes,
                inf.raw_nbytes,
                False,
                inf.born,
                inf.head,
            )
            self._insert(key, ent, inf.raw_nbytes)
        out, self._overflow = self._overflow, []
        return out

    def failed(self) -> list[tuple[int, _Inflight]]:
        got = list(self._failed.items())
        self._failed = {}
        return got

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Block until every encode finished (tests / shutdown)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                pending = [f.future for f in self._inflight.values()]
            if all(f.done() for f in pending):
                return
            time.sleep(0.005)

    # ── lookup / decode ────────────────────────────────────────────────
    def find(self, tokens: tuple, extra_hash: int, max_len: int, min_len: int = 0):
        """Longest stored prefix of ``tokens``: (key, length, where) with where in
        ``warm`` / ``inflight``; None when nothing is longer than ``min_len``."""
        best = None
        with self._lock:
            infl = [
                (k, i.entry_tokens, i.extra_hash, "inflight")
                for k, i in self._inflight.items()
            ]
        for key, stored, extra, where in infl + [
            (k, e.token_ids, e.extra_hash, "warm") for k, e in self.entries.items()
        ]:
            n = len(stored)
            if extra != extra_hash or n > max_len or n <= min_len:
                continue
            if best is not None and n <= best[1]:
                continue
            if tokens[:n] == stored:
                best = (key, n, where)
        return best

    def take_inflight(self, key: int):
        """The HOT arrays of an entry still being compressed (the encode result is discarded)."""
        with self._lock:
            inf = self._inflight.pop(key, None)
        if inf is None:
            return None
        inf.cancelled = True
        self.stats.inflight_hits += 1
        return inf

    def take(self, key: int):
        """Decode and remove a WARM entry -> (prompt_cache, entry); raises :class:`WarmCorruptError` (the
        entry is dropped) when it fails verification; None if absent."""
        ent = self.entries.pop(key, None)
        if ent is None:
            return None
        self.bytes -= ent.nbytes
        t0 = time.perf_counter()
        try:
            cache = self._decode(ent)
        except Exception as e:
            self.stats.corrupt += 1
            logger.warning(
                "APC warm: entry unusable (%s: %s), dropped", type(e).__name__, e
            )
            raise WarmCorruptError(str(e)) from e
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.hits += 1
        self.stats.hit_tokens += len(ent.token_ids)
        return cache, ent

    def _decode_blob(self, b: _Blob) -> mx.array:
        """Chunks decompress in parallel straight into one buffer; the byte planes are put back
        together on the GPU (a NumPy interleave runs at about 2 GB/s on one core)."""
        n = b.raw_bytes
        buf = np.empty(n, dtype=np.uint8)

        def task(i: int) -> int:
            dec = self._decompress_chunk(b.chunks[i])
            if i * CHUNK + len(dec) > n:
                raise WarmCorruptError("chunk overruns the array")
            buf[i * CHUNK : i * CHUNK + len(dec)] = np.frombuffer(dec, dtype=np.uint8)
            return len(dec)

        lens = list(self._chunk_pool.map(task, range(len(b.chunks))))
        if sum(lens) != n or any(x != CHUNK for x in lens[:-1]):
            raise WarmCorruptError("size mismatch")
        np_dt, mx_dt = _NP_DTYPE[b.dtype]
        if b.planes:
            half = n // 2
            a = mx.array(buf)
            u = (a[half:].astype(mx.uint16) << 8) | a[:half].astype(mx.uint16)
            return u.reshape(b.shape).view(
                mx_dt or getattr(mx, b.dtype.rsplit(".", 1)[-1])
            )
        arr = mx.array(buf.view(np_dt).reshape(b.shape))
        return arr.view(mx_dt) if mx_dt is not None else arr

    def _decode(self, ent: WarmEntry) -> list:
        out, to_eval = [], []
        for ly in ent.layers:
            items: list[Any] = []
            for it in ly.items:
                if it is None:
                    items.append(None)
                elif isinstance(it, _Blob):
                    a = self._decode_blob(it)
                    to_eval.append(a)
                    items.append(a)
                elif isinstance(it, _Quant):
                    a = mx.dequantize(
                        it.codes,
                        it.scales,
                        it.biases,
                        group_size=it.group,
                        bits=it.bits,
                    ).astype(it.dtype)
                    to_eval.append(a)
                    items.append(a)
                else:
                    items.append(mx.contiguous(mx.array(it)))
                    to_eval.append(items[-1])
            state = tuple(items) if ly.is_tuple else items
            from_state = getattr(ly.cls, "from_state", None)
            c: Any
            if callable(from_state):
                c = from_state(state, ly.meta)
            else:
                c = object.__new__(ly.cls)
                c.state = state
                c.meta_state = ly.meta
            out.append(c)
        if to_eval:
            mx.eval(*to_eval)
        return out

    def decode_entry(self, ent: WarmEntry) -> list:
        """Decode without removing (shutdown spill)."""
        return self._decode(ent)

    # ── maintenance ────────────────────────────────────────────────────
    def supersede(self, tokens: tuple, extra_hash: int, gen: int, keep: set) -> int:
        """Drop earlier-request entries (older ``born``) that are strict prefixes of ``tokens``."""
        dropped = 0
        with self._lock:
            for k, i in list(self._inflight.items()):
                n = len(i.entry_tokens)
                if (
                    i.extra_hash == extra_hash
                    and n < len(tokens)
                    and k not in keep
                    and i.born < gen
                    and tokens[:n] == i.entry_tokens
                ):
                    i.cancelled = True
                    del self._inflight[k]
                    dropped += 1
        for k, e in list(self.entries.items()):
            n = len(e.token_ids)
            if (
                e.extra_hash == extra_hash
                and n < len(tokens)
                and k not in keep
                and e.born < gen
                and tokens[:n] == e.token_ids
            ):
                del self.entries[k]
                self.bytes -= e.nbytes
                dropped += 1
        return dropped

    def snapshot(self) -> dict:
        s = self.stats
        return {
            "warm_mode": self.mode,
            "warm_entries": len(self.entries),
            "warm_inflight": len(self._inflight),
            "warm_bytes": self.bytes,
            "warm_max_bytes": self.budget_bytes,
            "warm_demotions": s.demotions,
            "warm_rejected": s.rejected,
            "warm_hits": s.hits,
            "warm_hit_tokens": s.hit_tokens,
            "warm_corrupt": s.corrupt,
            "warm_evicted_to_ssd": s.evicted_to_ssd,
            "warm_dropped": s.dropped,
            "warm_ratio": (s.raw_bytes_in / s.stored_bytes_in)
            if s.stored_bytes_in
            else 0.0,
        }

    def close(self) -> None:
        self._worker.shutdown(wait=True, cancel_futures=False)
        self._chunk_pool.shutdown(wait=True)


__all__ = ["MODES", "WarmCorruptError", "WarmEntry", "WarmTier", "prompt_cache_nbytes"]
