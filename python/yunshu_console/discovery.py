"""Engine discovery for ``yunshu console --engine auto``.

The console probes a candidate list (the configured engine URL plus 127.0.0.1 ports) with
``GET /health`` every couple of seconds, asynchronously and with a short timeout, so one console can
stay open while engines start and stop on other ports. ``YUNSHU_CONSOLE_DISCOVER_PORTS=all`` (the
default) takes the ports from every TCP socket listening on this Mac (``lsof``, refreshed every
``PORT_REFRESH_S``); a port that answered as something else is not probed again for ``NON_ENGINE_RECHECK_S``.
An explicit list (``8000,18990-18999``) probes exactly those ports.

A candidate counts only when its ``/health`` answers like a Yunshu engine (``status: "ok"``, an
``engine`` object, a ``server_state`` and a numeric ``uptime_seconds``: the combination the gateway's
``/health`` returns); any other server on a probed port is ignored.

Selection: keep the engine being followed while it is healthy (sticky); when it disappears (two
missed probes in a row, so one slow answer does not flip it) follow the most recently started live
engine (smallest ``uptime_seconds``); an engine pinned by the user wins while it is live. With no
live engine ``selected`` is None ("waiting for an engine"); ``current_url`` then stays on the last
one followed, so the proxy answers ``engine_unreachable`` and the poller records the gap.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PORTS = "all"
MISS_LIMIT = 2
PORT_REFRESH_S = 10.0
NON_ENGINE_RECHECK_S = 60.0


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


SERVER_PORTS: set[int] = (
    set()
)  # ports a `yunshu serve` process itself listens on (last refresh)


def listening_ports() -> list[int]:
    """TCP ports listening on this machine (lsof; the current user's processes), without the ports of
    console processes: a console proxies /health, so it would look like the engine behind it.
    Also records in ``SERVER_PORTS`` which ports belong to a ``yunshu serve`` process, so a proxy in
    front of an engine (same start time) can be told apart from the engine."""
    ports: dict[int, int] = {}
    pid = 0
    for line in _run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpn"]).splitlines():
        if line.startswith("p"):
            pid = int(line[1:]) if line[1:].isdigit() else 0
        elif m := re.match(r"^n.*:(\d+)$", line):
            ports.setdefault(int(m.group(1)), pid)
    pids = sorted({p for p in ports.values() if p})
    consoles: set[int] = set()
    servers: set[int] = set()
    if pids:
        for line in _run(
            ["ps", "-o", "pid=,command=", "-p", ",".join(map(str, pids))]
        ).splitlines():
            pid_s, _, command = line.strip().partition(" ")
            if "yunshu_console" in command or " console " in f"{command} ":
                consoles.add(int(pid_s))
            elif re.search(r"yunshu serve\b|yunshu_gateway", command):
                servers.add(int(pid_s))
    SERVER_PORTS.clear()
    SERVER_PORTS.update(p for p, owner in ports.items() if owner in servers)
    return sorted(
        p for p, owner in ports.items() if 1 <= p <= 65535 and owner not in consoles
    )


def parse_ports(spec: str | None) -> list[int]:
    """``"8000,18990-18999"`` -> ports in order, de-duplicated; malformed parts are skipped."""
    out: list[int] = []
    for part in str(spec or "").replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                rng = range(min(lo, hi), max(lo, hi) + 1)
            else:
                rng = range(int(part), int(part) + 1)
        except ValueError:
            continue
        for p in rng:
            if 1 <= p <= 65535 and p not in out:
                out.append(p)
    return out


def is_yunshu_health(body: Any) -> bool:
    """True when a /health body has the shape only a Yunshu gateway returns."""
    if not isinstance(body, dict) or body.get("status") != "ok":
        return False
    up = body.get("uptime_seconds")
    return (
        isinstance(body.get("engine"), dict)
        and isinstance(body.get("server_state"), str)
        and isinstance(up, (int, float))
        and not isinstance(up, bool)
    )


def candidate_urls(
    ports: list[int], extra: str | None = None, skip_port: int | None = None
) -> list[str]:
    urls: list[str] = []
    if extra and extra != "auto":
        urls.append(extra.rstrip("/"))
    for p in ports:
        if p == skip_port:
            continue
        u = f"http://127.0.0.1:{p}"
        if u not in urls:
            urls.append(u)
    return urls


class Discovery:
    def __init__(
        self,
        candidates: list[str],
        *,
        default_url: str = "http://127.0.0.1:8000",
        token: str | None = None,
        interval_s: float = 2.0,
        timeout_s: float = 0.8,
        client: httpx.AsyncClient | None = None,
        on_change: Callable[[str | None, str], None] | None = None,
        clock=time.time,
        pause=None,
        port_source: Callable[[], list[int]] | None = None,
        skip_port: int | None = None,
    ) -> None:
        self.pause = pause
        self.fixed = [c.rstrip("/") for c in candidates]
        self.candidates = list(self.fixed)
        self.port_source = port_source  # set: candidates follow the listening ports
        self.skip_port = skip_port
        self._ports_at = float("-inf")
        self._not_yunshu: dict[str, float] = {}
        self.interval_s = interval_s
        self.token = token
        self._clock = clock
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 0.5)),
            follow_redirects=False,
        )
        self._owns_client = client is None
        self.on_change = on_change
        self.engines: dict[str, dict] = {}  # url -> info, live ones only
        self._misses: dict[str, int] = {}
        self._models: dict[str, tuple[float, str | None, list[str]]] = {}
        self.pinned: str | None = None
        self.selected: str | None = None
        self.current_url = default_url.rstrip("/")  # last followed engine
        self.scans = 0

    # ── probing ────────────────────────────────────────────────────────
    async def _probe(self, url: str) -> dict | None:
        try:
            r = await self._client.get(url + "/health")
            if r.status_code != 200:
                return None
            body = r.json()
        except (httpx.HTTPError, ValueError, OSError):
            return None
        if not is_yunshu_health(body):
            self._not_yunshu[url] = self._clock()
            return None
        now = self._clock()
        uptime = float(body["uptime_seconds"])
        info = {
            "url": url,
            "port": httpx.URL(url).port,
            "state": body.get("server_state"),
            "loaded": bool(body["engine"].get("loaded")),
            "uptime_seconds": uptime,
            "started_at": round(now - uptime, 1),
            "model": None,
            "models": [],
        }
        return info

    async def _fill_models(self, info: dict) -> None:
        """Model ids from GET /v1/models, at most every 10 s per engine (auth failures leave it empty)."""
        url = info["url"]
        at, model, models = self._models.get(url, (0.0, None, []))
        if self._clock() - at >= 10.0 or (not models and self._clock() - at >= 2.0):
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            try:
                r = await self._client.get(url + "/v1/models", headers=headers)
                ids = (
                    [
                        str(m["id"])
                        for m in (r.json().get("data") or [])
                        if isinstance(m, dict) and m.get("id")
                    ]
                    if r.status_code == 200
                    else []
                )
            except (httpx.HTTPError, ValueError, OSError, AttributeError):
                ids = []
            models = ids
            model = ids[0] if ids else None
            self._models[url] = (self._clock(), model, models)
        info["model"], info["models"] = model, list(models)

    async def _refresh_candidates(self) -> None:
        if self.port_source is None or self._clock() - self._ports_at < PORT_REFRESH_S:
            return
        self._ports_at = self._clock()
        ports = await asyncio.to_thread(self.port_source)
        self.candidates = candidate_urls(ports, None, skip_port=self.skip_port)
        for u in self.fixed:
            if u not in self.candidates:
                self.candidates.insert(0, u)
        # an engine that stopped listening is dropped by the miss count below
        for u in self.engines:
            if u not in self.candidates:
                self.candidates.append(u)

    def _probe_now(self, url: str) -> bool:
        at = self._not_yunshu.get(url)
        return at is None or self._clock() - at >= NON_ENGINE_RECHECK_S

    async def scan(self) -> None:
        """One probe round over every candidate (concurrent), then re-select."""
        self.scans += 1
        await self._refresh_candidates()
        urls = [u for u in self.candidates if u in self.engines or self._probe_now(u)]
        results = await asyncio.gather(*(self._probe(u) for u in urls))
        for url, info in zip(urls, results, strict=True):
            if info is not None:
                self._misses[url] = 0
                previous = self.engines.get(url)
                # a restart on the same port shows as a smaller uptime: keep the new start time
                self.engines[url] = info
                if previous is None:
                    self._models.pop(url, None)
            elif url in self.engines:
                self._misses[url] = self._misses.get(url, 0) + 1
                if self._misses[url] >= MISS_LIMIT:
                    del self.engines[url]
                    self._models.pop(url, None)
        for info in self.engines.values():
            await self._fill_models(info)
        self._select()

    # ── selection ──────────────────────────────────────────────────────
    def _dedupe(self) -> None:
        """One engine reached through several ports (a recording proxy, a tunnel) answers /health
        with the same start time: keep the port a `yunshu serve` process listens on, else the lowest."""
        if self.port_source is None:
            return
        groups: list[list[str]] = []
        for url in sorted(self.engines, key=lambda u: self.engines[u]["started_at"]):
            info = self.engines[url]
            head = self.engines[groups[-1][0]] if groups else None
            if (
                head is not None
                and abs(info["started_at"] - head["started_at"]) <= 2.0
                and info.get("model") == head.get("model")
            ):
                groups[-1].append(url)
            else:
                groups.append([url])
        for urls in groups:
            if len(urls) < 2:
                continue
            keep = min(
                urls,
                key=lambda u: (
                    self.engines[u]["port"] not in SERVER_PORTS,
                    self.engines[u]["port"] or 0,
                ),
            )
            for u in urls:
                if u != keep:
                    self.engines.pop(u, None)

    def _select(self) -> None:
        self._dedupe()
        live = self.engines
        if self.pinned and self.pinned in live:
            choice: str | None = self.pinned
        elif self.selected in live:
            choice = self.selected  # sticky
        elif live:
            choice = min(live, key=lambda u: (live[u]["uptime_seconds"], u))
        else:
            choice = None
        self.selected = choice
        if choice is not None:
            changed = choice != self.current_url
            old = self.current_url
            self.current_url = choice
            if changed and self.on_change is not None:
                self.on_change(old, choice)

    def pin(self, url: str | None) -> bool:
        """Pin a live engine (``None`` unpins). False when the url is not a live engine."""
        if url is None:
            self.pinned = None
        else:
            url = url.rstrip("/")
            if url not in self.engines:
                return False
            self.pinned = url
        self._select()
        return True

    # ── view ───────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        data = []
        for url, info in sorted(
            self.engines.items(), key=lambda kv: (kv[1]["port"] or 0, kv[0])
        ):
            data.append(
                {
                    **info,
                    "selected": url == self.selected,
                    "pinned": url == self.pinned,
                }
            )
        return {
            "object": "yunshu.console.engines",
            "mode": "auto",
            "selected": self.selected,
            "pinned": self.pinned,
            "waiting": self.selected is None,
            "current_url": self.current_url,
            "candidates": len(self.candidates),
            "data": data,
        }

    # ── loop ───────────────────────────────────────────────────────────
    async def run(self) -> None:
        while True:
            started = time.perf_counter()
            try:
                if self.pause is None or not self.pause.paused:
                    await self.scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("engine discovery failed")
            await asyncio.sleep(
                max(0.0, self.interval_s - (time.perf_counter() - started))
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
