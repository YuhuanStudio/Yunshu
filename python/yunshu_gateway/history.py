"""Server-side history ring behind ``GET /v1/yunshu/history``.

A sampler wakes every ``YUNSHU_HISTORY_INTERVAL_S`` (5 s) and appends one row of cheap numbers to
a preallocated columnar ring so the console's charts survive a reload. Memory is fixed at start:
``capacity * (8 + 4 * len(FIELDS))`` bytes (8,640 slots * 56 B = 0.46 MiB for 12 h at 5 s) and
never grows. The sampler runs as an asyncio task on the gateway loop and only reads counters:
MLX allocator counters, ``proc_pid_rusage``, the request registry (its own short lock, not any
lock the generation thread takes). It never calls into an engine, never touches the GPU thread
and never evaluates an array.
"""

from __future__ import annotations

import array
import asyncio
import contextlib
import logging
import math
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Float32 columns after the float64 timestamp ``t``. NaN = not available (reported as null).
FIELDS = (
    "active_gb",
    "cache_gb",
    "footprint_gb",
    "pressure",
    "requests_active",
    "queued",
    "decode_tps",
    "prefill_tps",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "prompt_tokens_per_s",
    "completion_tokens_per_s",
)
_NAN = float("nan")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


class HistoryRing:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._t = array.array("d", [_NAN]) * self.capacity
        self._cols = {f: array.array("f", [_NAN]) * self.capacity for f in FIELDS}
        self._n = 0  # rows ever appended
        self._lock = (
            threading.Lock()
        )  # the sampler and readers only; nothing else takes it

    @property
    def nbytes(self) -> int:
        return self.capacity * (8 + 4 * len(FIELDS))

    def __len__(self) -> int:
        return min(self._n, self.capacity)

    def append(self, t: float, row: dict[str, float | None]) -> None:
        with self._lock:
            i = self._n % self.capacity
            self._t[i] = t
            for f in FIELDS:
                v = row.get(f)
                self._cols[f][i] = _NAN if v is None else float(v)
            self._n += 1

    def read(self, since: float | None = None, step: float | None = None) -> dict:
        """Columnar rows in time order, newer than ``since`` (epoch s); with ``step`` > 0 each
        ``step``-second bucket is the mean of its available samples."""
        with self._lock:
            n = len(self)
            start = self._n - n
            idx = [(start + k) % self.capacity for k in range(n)]
            t = [self._t[i] for i in idx]
            cols = {f: [self._cols[f][i] for i in idx] for f in FIELDS}
        keep = [k for k, tv in enumerate(t) if since is None or tv > since]
        if step and step > 0 and keep:
            buckets: dict[int, list[int]] = {}
            for k in keep:
                buckets.setdefault(int(t[k] // step), []).append(k)
            out_t: list[float] = []
            out_c: dict[str, list[float | None]] = {f: [] for f in FIELDS}
            for b in sorted(buckets):
                ks = buckets[b]
                out_t.append(round(sum(t[k] for k in ks) / len(ks), 3))
                for f in FIELDS:
                    vals = [cols[f][k] for k in ks if not math.isnan(cols[f][k])]
                    out_c[f].append(round(sum(vals) / len(vals), 3) if vals else None)
            return {"t": out_t, **out_c}
        return {
            "t": [round(t[k], 3) for k in keep],
            **{
                f: [
                    None if math.isnan(cols[f][k]) else round(cols[f][k], 3)
                    for k in keep
                ]
                for f in FIELDS
            },
        }


def sample_row() -> dict[str, float | None]:
    """One row of numbers from counters only (no engine calls)."""
    from . import memory_ledger as ml
    from .x_yunshu import registry

    row: dict[str, float | None] = {}
    mlx = ml.mlx_counters()
    row["active_gb"] = ml.gb(mlx["active"])
    row["cache_gb"] = ml.gb(mlx["cache"])
    foot = ml.process_footprint()["footprint"]
    row["footprint_gb"] = ml.gb(foot)
    total = ml.host()["total_gb"]
    row["pressure"] = (
        round(mlx["active"] / (total * 1e9), 4)
        if mlx["active"] is not None and total
        else None
    )
    infos = registry.active()
    queued = 0
    decode = 0.0
    any_decode = False
    prefill = 0.0
    any_prefill = False
    now = time.perf_counter()
    for info in infos:
        st = info.stats
        if st is None:
            queued += 1
            continue
        phase = st.phase
        if phase == "queued":
            queued += 1
        elif phase == "prefill" and st.t_admit and st.prefill_done > 0:
            prefill += st.prefill_done / max(now - st.t_admit, 1e-6)
            any_prefill = True
        elif phase == "decode" and st.t_first and st.t_last > st.t_first:
            decode += (st.generated - 1) / (st.t_last - st.t_first)
            any_decode = True
    row["requests_active"] = float(len(infos))
    row["queued"] = float(queued)
    row["decode_tps"] = round(decode, 1) if any_decode else None
    row["prefill_tps"] = round(prefill, 1) if any_prefill else None
    window = registry.recent(60.0)
    ttfts = [e["ttft_ms"] for e in window if e.get("ttft_ms") is not None]
    row["ttft_p50_ms"] = percentile(ttfts, 0.5)
    row["ttft_p95_ms"] = percentile(ttfts, 0.95)
    row["prompt_tokens_per_s"] = (
        round(sum(e["prompt_tokens"] for e in window) / 60.0, 2) if window else None
    )
    row["completion_tokens_per_s"] = (
        round(sum(e["completion_tokens"] for e in window) / 60.0, 2) if window else None
    )
    return row


class Sampler:
    def __init__(self, interval_s: float, hours: float):
        self.interval_s = float(interval_s)
        self.ring = HistoryRing(math.ceil(hours * 3600.0 / self.interval_s))
        self.errors = 0
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None

    def sample_once(self, now: float | None = None) -> None:
        """Append one row; a failing collector costs that sample, never the loop."""
        t = time.time() if now is None else now
        try:
            self.ring.append(t, sample_row())
        except Exception as exc:
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.debug("history sample failed", exc_info=True)

    async def _run(self) -> None:
        while True:
            self.sample_once()
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name="yunshu-history"
            )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def payload(self, since: float | None, step: float | None) -> dict[str, Any]:
        return {
            "object": "yunshu.history",
            "interval_s": self.interval_s,
            "ring": {
                "capacity": self.ring.capacity,
                "rows": len(self.ring),
                "bytes": self.ring.nbytes,
            },
            "errors": self.errors,
            "last_error": self.last_error,
            "fields": list(FIELDS),
            "step_s": step if step and step > 0 else self.interval_s,
            "series": self.ring.read(since, step),
        }


_SAMPLER: Sampler | None = None


def get() -> Sampler | None:
    return _SAMPLER


def start_from_settings() -> Sampler | None:
    """Start the process-wide sampler (idempotent); None when it is switched off."""
    global _SAMPLER
    from yunshu_engine import settings

    interval = float(settings.get("YUNSHU_HISTORY_INTERVAL_S") or 0.0)
    hours = float(settings.get("YUNSHU_HISTORY_HOURS") or 0.0)
    if interval <= 0 or hours <= 0:
        return None
    if _SAMPLER is None:
        _SAMPLER = Sampler(interval, hours)
    _SAMPLER.start()
    return _SAMPLER


async def stop() -> None:
    global _SAMPLER
    s, _SAMPLER = _SAMPLER, None
    if s is not None:
        await s.stop()
