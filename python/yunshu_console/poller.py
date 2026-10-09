"""The console process watches the engine: one cheap read a second, recorded.

Each tick reads ``GET /v1/yunshu/status`` (live counters; the engine builds it in well under a
millisecond) and ``GET /v1/yunshu/requests/recent?after_seq=`` (finished requests since the last
cursor), and about every 5 s the host telemetry. Nothing else is asked of the engine, and nothing
runs on its generation thread: every read is an ordinary HTTP request the engine answers on its
event loop, the same ones the web console makes.

What is recorded:

* a metrics row per tick (``FIELDS``), only while the engine answers; a span with no rows *is* the
  outage, and ``store.read`` reports it as a gap instead of interpolating;
* the finished requests, by cursor ``(boot_id, seq)``: the first read after the console starts
  takes everything still in the engine's ring, later reads only what is new, and a changed
  ``boot_id`` (the engine restarted) starts the sequence over, so none is skipped or repeated
  (rows are keyed by ``(t, request_id)``, so even a replay writes nothing twice);
* events: engine unreachable / reachable again (with how long it was away), engine restarted,
  a model loaded / unloaded, a model that failed to load.

Request rows carry metadata only. The engine's finished-request entries hold no prompt or output
text in the first place, and :func:`yunshu_console.store.request_row` drops anything not on its
whitelist.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import math
import time
from typing import Any

import httpx

from .store import HistoryStore

logger = logging.getLogger(__name__)

FIELDS = (
    "active_gb",
    "cache_gb",
    "peak_gb",
    "pressure",
    "rss_gb",
    "requests_active",
    "queued",
    "decode_tps",
    "prefill_tps",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "prompt_tokens_per_s",
    "completion_tokens_per_s",
    "cached_tokens_per_s",
    # host telemetry (only while the engine's YUNSHU_TELEMETRY is on; null otherwise)
    "gpu_w",
    "package_w",
    "gpu_mhz",
    "gpu_active",
    "die_c",
)
GIB = 1024**3
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def memory_gib(memory: dict, name: str) -> float | None:
    """Binary GB of ``active`` / ``cache`` / ``peak`` from /status: exact bytes when the engine
    sends them, else its decimal ``*_gb`` converted once."""
    b = _num(memory.get(f"{name}_bytes"))
    if b is not None:
        return round(b / GIB, 3)
    g = _num(memory.get(f"{name}_gb"))
    return None if g is None else round(g * 1e9 / GIB, 3)


def row_from_status(status: dict, window: list[dict]) -> dict[str, float | None]:
    """One metrics row from a /status payload and the requests finished in the last 60 s."""
    row: dict[str, float | None] = dict.fromkeys(FIELDS)
    memory = status.get("memory") or {}
    row["active_gb"] = memory_gib(memory, "active")
    row["cache_gb"] = memory_gib(memory, "cache")
    row["peak_gb"] = memory_gib(memory, "peak")
    row["pressure"] = _num(memory.get("pressure"))
    reqs = status.get("requests") or {}
    row["requests_active"] = _num(reqs.get("active"))
    row["queued"] = _num(reqs.get("queued"))
    items = reqs.get("items") or []
    live = _num((status.get("throughput") or {}).get("live_decode_tps"))
    decode = [
        _num(i.get("tokens_per_second"))
        for i in items
        if i.get("phase") == "decode" and _num(i.get("tokens_per_second")) is not None
    ]
    prefill = [
        _num(i.get("tokens_per_second"))
        for i in items
        if i.get("phase") == "prefill" and _num(i.get("tokens_per_second")) is not None
    ]
    row["decode_tps"] = (
        live if live is not None else (round(sum(decode), 1) if decode else None)
    )  # type: ignore[arg-type]
    row["prefill_tps"] = round(sum(prefill), 1) if prefill else None  # type: ignore[arg-type]
    ttfts = [t for e in window if (t := _num(e.get("ttft_ms"))) is not None]
    row["ttft_p50_ms"] = percentile(ttfts, 0.5)
    row["ttft_p95_ms"] = percentile(ttfts, 0.95)
    if window:
        row["prompt_tokens_per_s"] = round(
            sum(_num(e.get("prompt_tokens")) or 0 for e in window) / 60, 2
        )
        row["completion_tokens_per_s"] = round(
            sum(_num(e.get("completion_tokens")) or 0 for e in window) / 60, 2
        )
        row["cached_tokens_per_s"] = round(
            sum(_num(e.get("cached_tokens")) or 0 for e in window) / 60, 2
        )
    return row


def telemetry_fields(host: dict) -> dict[str, float | None]:
    t = (host or {}).get("telemetry") or {}
    watts, gpu, temp = (
        t.get("watts") or {},
        t.get("gpu") or {},
        t.get("temperature") or {},
    )
    return {
        "gpu_w": _num(watts.get("gpu")),
        "package_w": _num(watts.get("package")),
        "gpu_mhz": _num(gpu.get("frequency_mhz")),
        "gpu_active": _num(gpu.get("active_ratio")),
        "die_c": _num(temp.get("die_max_c")),
    }


class Poller:
    """Reads the engine on a fixed cadence and records it. ``tick`` is one read (the tests call it
    directly with a fake transport); ``run`` loops until cancelled."""

    def __init__(
        self,
        engine_url: str,
        store: HistoryStore | None,
        *,
        token: str | None = None,
        interval_s: float = 1.0,
        client: httpx.AsyncClient | None = None,
        clock=time.time,
    ) -> None:
        self.engine_url = engine_url.rstrip("/")
        self.store = store
        self.interval_s = max(0.25, float(interval_s))
        self.token = token
        self._clock = clock
        self._client = client or httpx.AsyncClient(
            base_url=self.engine_url, timeout=httpx.Timeout(2.0, connect=1.5)
        )
        self._owns_client = client is None
        # connection state
        self.up: bool | None = None  # None until the first answer or failure
        self.since: float | None = None  # when the current state began
        self.last_ok: float | None = None
        self.last_error: str | None = None
        self.error_status: int | None = None
        self.engine: dict[str, Any] = {}
        # request cursor (boot_id, seq)
        self.boot_id: str | None = None
        self.seq: int | None = None
        self.requests_recorded = 0
        self._window: collections.deque[dict] = collections.deque()
        self._host: dict[str, float | None] = dict.fromkeys(FIELDS[-5:])
        self._host_at = 0.0
        self._models: dict[str, bool] = {}
        self._load_error: str | None = None
        self._pid: int | None = None
        self._uptime: float | None = None
        self.ticks = 0
        self.errors = 0

    # ── http ────────────────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _get(self, path: str, **params: Any) -> Any:
        response = await self._client.get(
            path, params=params or None, headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

    # ── one read ────────────────────────────────────────────────────────
    async def tick(self) -> None:
        now = self._clock()
        self.ticks += 1
        try:
            status = await self._get("/v1/yunshu/status")
        except httpx.HTTPStatusError as exc:
            self._down(
                now, f"HTTP {exc.response.status_code}", exc.response.status_code
            )
            return
        except (httpx.HTTPError, ValueError, OSError) as exc:
            self._down(now, type(exc).__name__, None)
            return
        self._reachable(now, status)
        await self._take_requests(now)
        self._trim_window(now)
        row = row_from_status(status, list(self._window))
        row["rss_gb"] = self._rss_gb(status)
        if now - self._host_at >= 5.0:
            self._host_at = now
            with contextlib.suppress(httpx.HTTPError, ValueError, OSError):
                self._host = telemetry_fields(await self._get("/v1/yunshu/host"))
        row.update(self._host)
        if self.store is not None:
            self.store.add_sample(now, row)

    def _down(self, now: float, reason: str, status: int | None) -> None:
        self.errors += 1
        self.last_error, self.error_status = reason, status
        if self.up is not False:
            was_up = self.up
            self.up, self.since = False, now
            self.engine = {**self.engine}
            if self.store is not None:
                self.store.add_event(
                    now,
                    "engine_unreachable" if status is None else "engine_error",
                    {"reason": reason, "after_up": bool(was_up)},
                )
            logger.info(
                "engine %s: %s", "unreachable" if status is None else "error", reason
            )

    def _reachable(self, now: float, status: dict) -> None:
        uptime = _num(status.get("uptime_s"))
        pid = status.get("pid") if isinstance(status.get("pid"), int) else None
        restarted = (
            self._uptime is not None
            and uptime is not None
            and uptime + 2.0 < self._uptime
        ) or (self._pid is not None and pid is not None and pid != self._pid)
        if self.up is False and self.since is not None and self.store is not None:
            self.store.add_event(
                now, "engine_reachable", {"down_s": round(now - self.since, 1)}
            )
        if self.up is not True:
            self.since = now
        if restarted and self.store is not None:
            self.store.add_event(now, "engine_restarted", {"pid": pid})
        self.up, self.last_ok, self.last_error, self.error_status = (
            True,
            now,
            None,
            None,
        )
        self._uptime, self._pid = uptime, pid
        models = {
            str(m.get("id")): bool(m.get("loaded"))
            for m in status.get("models") or []
            if isinstance(m, dict) and m.get("id")
        }
        if self.store is not None:
            for mid, loaded in models.items():
                if loaded and not self._models.get(mid):
                    self.store.add_event(now, "model_loaded", {"model": mid})
            for mid, was in self._models.items():
                if was and not models.get(mid):
                    self.store.add_event(now, "model_unloaded", {"model": mid})
            err = status.get("load_error")
            err = err if isinstance(err, str) and err else None
            if err and err != self._load_error:
                self.store.add_event(now, "load_error", {"message": err[:240]})
            self._load_error = err
        else:
            self._load_error = status.get("load_error") or None
        self._models = models
        self.engine = {
            "version": status.get("version"),
            "state": status.get("state"),
            "uptime_s": uptime,
            "pid": pid,
            "load_error": self._load_error,
            "models": sorted(m for m, loaded in models.items() if loaded),
        }

    async def _take_requests(self, now: float) -> None:
        try:
            page = await self._get(
                "/v1/yunshu/requests/recent",
                limit=512,
                **({"after_seq": self.seq} if self.seq is not None else {}),
            )
        except (httpx.HTTPError, ValueError, OSError):
            return
        boot, latest = page.get("boot_id"), page.get("latest_seq")
        if self.boot_id is not None and boot != self.boot_id:
            # the engine restarted: the sequence started over; read everything that is in its ring
            self.boot_id, self.seq = boot, None
            try:
                page = await self._get("/v1/yunshu/requests/recent", limit=512)
            except (httpx.HTTPError, ValueError, OSError):
                return
            latest = page.get("latest_seq")
        self.boot_id = boot
        for entry in reversed(page.get("data") or []):  # oldest first
            if self.store is not None:
                self.store.add_request(entry)
            self.requests_recorded += 1
            if isinstance(entry.get("t"), (int, float)):
                self._window.append(entry)
        if isinstance(latest, int):
            self.seq = latest

    def _trim_window(self, now: float) -> None:
        while self._window and self._window[0].get("t", now) < now - 60:
            self._window.popleft()

    def _rss_gb(self, status: dict) -> float | None:
        """RSS of the engine process when it runs on this machine (psutil, one syscall)."""
        pid = status.get("pid")
        host = httpx.URL(self.engine_url).host
        if not isinstance(pid, int) or host not in LOCAL_HOSTS:
            return None
        try:
            import psutil

            return round(psutil.Process(pid).memory_info().rss / GIB, 3)
        except Exception:
            return None

    # ── loop ────────────────────────────────────────────────────────────
    async def run(self) -> None:
        while True:
            started = time.perf_counter()
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # a bug in one tick must not end the recorder
                self.errors += 1
                logger.exception("console poll failed")
            await asyncio.sleep(
                max(0.0, self.interval_s - (time.perf_counter() - started))
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def state(self) -> dict[str, Any]:
        return {
            "up": self.up,
            "since": self.since,
            "last_ok": self.last_ok,
            "last_error": self.last_error,
            "error_status": self.error_status,
            "engine_url": self.engine_url,
            "engine": self.engine,
            "poll_s": self.interval_s,
            "requests_recorded": self.requests_recorded,
            "cursor": {"boot_id": self.boot_id, "seq": self.seq},
        }
