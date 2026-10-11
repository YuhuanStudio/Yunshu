"""Persistent per-request event log of the production server (opt-in, local, numbers only).

One JSON line per finished generation request: timestamps, route and dialect, model id,
token counts, TTFT, decode speed, speculative mode and acceptance, cache tier, finish reason,
concurrency at start and end, the arm label (for a future A/B) and the engine build id.

Privacy is structural, not a filter: :func:`event_from_stats` builds the event from a fixed
whitelist of numeric fields and short enumerated / sanitised labels. Prompts, outputs, token
ids, headers and client identity never reach this module's inputs, and a free-text value
cannot pass the label sanitiser (spaces, quotes and anything longer than 96 chars are dropped).

The file is append-only JSONL with size-based rotation: ``serve_log.jsonl`` plus ``keep``
rotated files, each at most ``max_bytes``, so the directory is bounded by
``max_bytes * (keep + 1)`` (its own small cap; the APC disk budget is not touched).
Off by default (``YUNSHU_SERVE_LOG``). Logging never raises into a request.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import heapq
import json
import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from yunshu_engine import paths, settings

logger = logging.getLogger(__name__)

FILE_NAME = "serve_log.jsonl"
SCHEMA = 1
_LABEL_OK = re.compile(r"^[A-Za-z0-9._:/+\-]{1,96}$")
_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

_arm_override: str | None = None
_build_id: str | None = None


def label(value: Any) -> str | None:
    """``value`` when it is a short identifier-like string, else None (never free text)."""
    if isinstance(value, str) and _LABEL_OK.match(value):
        return value
    return None


def request_label(value: Any) -> str | None:
    return value if isinstance(value, str) and _REQUEST_ID_OK.fullmatch(value) else None


def set_arm(arm: str | None) -> None:
    """Set the arm label for requests finishing from now on (the future switchback driver
    calls this at block boundaries); None clears it back to ``YUNSHU_ARM``."""
    global _arm_override
    _arm_override = label(arm) if arm else None


def current_arm() -> str | None:
    return _arm_override or label(settings.get("YUNSHU_ARM"))


def build_id() -> str:
    """Engine build: ``<version>+<git short sha>`` for a checkout, else the version."""
    global _build_id
    if _build_id is None:
        from yunshu_engine.version import yunshu_version

        sha = None
        with contextlib.suppress(Exception):
            out = subprocess.run(
                ["git", "rev-parse", "--short=10", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if out.returncode == 0:
                sha = label(out.stdout.strip())
        ver = label(yunshu_version()) or "unknown"
        _build_id = f"{ver}+{sha}" if sha else ver
    return _build_id


def _num(value: Any, nd: int | None = None) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(value, nd) if nd is not None and isinstance(value, float) else value


def ctx_bucket(prompt_tokens: int | None) -> str | None:
    """Context-size bucket (``<=1k`` ... ``>64k``): a workload class that carries no content."""
    if prompt_tokens is None:
        return None
    for edge, name in ((1024, "<=1k"), (4096, "<=4k"), (16384, "<=16k")):
        if prompt_tokens <= edge:
            return name
    return "<=64k" if prompt_tokens <= 65536 else ">64k"


def event_from_stats(stats: dict, ctx: dict) -> dict:
    """The log line for one request: a whitelist projection of ``x_yunshu`` ``stats`` and the
    request context ``ctx`` (route, dialect, model, finish_reason, t_start/t_end wall time,
    concurrency_start/end, stream, arm, build). Unknown keys in either are ignored."""
    from .metadata_projection import extras

    spec = stats.get("speculative") or {}
    cache = stats.get("cache") or {}
    prompt = _num(stats.get("prompt_tokens"))
    return {
        **extras(stats),
        "schema": SCHEMA,
        "status": _num(ctx.get("status")),
        "prefill_tps": _num(stats.get("prefill_tps")),
        "request_id": request_label(ctx.get("request_id")),
        "t_start": _num(ctx.get("t_start"), 3),
        "t_end": _num(ctx.get("t_end"), 3),
        "route": label(ctx.get("route")),
        "dialect": label(ctx.get("dialect")),
        "stream": bool(ctx.get("stream")),
        "model": label(ctx.get("model")),
        "prompt_tokens": prompt,
        "completion_tokens": _num(stats.get("completion_tokens")),
        "cached_tokens": _num(stats.get("cached_tokens")),
        "ctx_bucket": ctx_bucket(int(prompt)) if prompt is not None else None,
        "queue_wait_ms": _num(stats.get("queue_wait_ms")),
        "ttft_ms": _num(stats.get("ttft_ms")),
        "prefill_ms": _num(stats.get("prefill_ms")),
        "decode_ms": _num(stats.get("decode_ms")),
        "decode_tps": _num(stats.get("decode_tps")),
        "spec_mode": label(spec.get("mode")),
        "spec_drafted": _num(spec.get("drafted")),
        "spec_accepted": _num(spec.get("accepted")),
        "spec_acceptance": _num(spec.get("acceptance_rate")),
        "cache_tier": label(cache.get("tier")),
        "finish_reason": label(ctx.get("finish_reason")),
        "cancelled": bool(stats.get("cancelled")),
        "concurrency_start": _num(ctx.get("concurrency_start")),
        "concurrency_end": _num(ctx.get("concurrency_end")),
        "arm": label(ctx.get("arm")),
        "build": label(ctx.get("build")),
    }


def history_row(raw: dict) -> dict:
    """Project untrusted on-disk metadata; never forward arbitrary JSON keys."""
    from .metadata_projection import extras

    safe = event_from_stats({}, {})
    for key in safe:
        if key in ("schema", "stream", "cancelled"):
            safe[key] = bool(raw.get(key)) if key != "schema" else SCHEMA
        elif key == "request_id":
            safe[key] = request_label(raw.get(key))
        elif key in (
            "model",
            "route",
            "dialect",
            "spec_mode",
            "cache_tier",
            "finish_reason",
            "arm",
            "build",
        ):
            safe[key] = label(raw.get(key))
        elif key not in (
            "ctx_bucket",
            "speculative",
            "structured_output",
            "latency",
            "energy",
            "cache",
            "reasons",
        ):
            safe[key] = _num(raw.get(key))
    rid = (
        safe["request_id"]
        or "legacy_"
        + hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest()[:24]
    )
    extra = extras(raw)
    if not raw.get("speculative"):
        extra.pop("speculative")
    if not raw.get("cache"):
        extra.pop("cache")
    return {
        "t": safe["t_end"],
        "request_id": rid,
        "model": safe["model"],
        "path": safe["route"],
        "stream": safe["stream"],
        "status": safe["status"],
        **{
            k: safe[k]
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "ttft_ms",
                "decode_tps",
            )
        },
        "prefill_tps": safe["prefill_tps"],
        "latency": None,
        "energy": None,
        "speculative": {
            "mode": safe["spec_mode"],
            "drafted": safe["spec_drafted"],
            "accepted": safe["spec_accepted"],
            "acceptance_rate": safe["spec_acceptance"],
        }
        if safe["spec_mode"]
        else None,
        "cache": {
            "tier": safe["cache_tier"],
            "cached_tokens": safe["cached_tokens"],
            "reload_ms": None,
        },
        **extra,
    }


class ServeLog:
    """Append-only JSONL with rotation; thread-safe; never raises from :meth:`append`."""

    def __init__(self, directory: Path, max_bytes: int, keep: int):
        self.dir = Path(directory)
        self.path = self.dir / FILE_NAME
        self.max_bytes = max(int(max_bytes), 1024)
        self.keep = max(int(keep), 0)
        self._lock = threading.Lock()
        self._warned = False

    @property
    def cap_bytes(self) -> int:
        return self.max_bytes * (self.keep + 1)

    def _rotated(self, i: int) -> Path:
        return self.dir / f"{FILE_NAME}.{i}"

    def _rotate(self) -> None:
        if self.keep == 0:
            self.path.unlink(missing_ok=True)
            return
        self._rotated(self.keep).unlink(missing_ok=True)
        for i in range(self.keep - 1, 0, -1):
            src = self._rotated(i)
            if src.exists():
                src.replace(self._rotated(i + 1))
        self.path.replace(self._rotated(1))

    def append(self, event: dict) -> bool:
        data = (
            json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        try:
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                size = self.path.stat().st_size if self.path.exists() else 0
                if size and size + len(data) > self.max_bytes:
                    self._rotate()
                with open(self.path, "ab") as f:
                    f.write(data)
            return True
        except OSError as exc:
            if not self._warned:
                self._warned = True
                logger.warning("serve log write failed (%s); events are dropped", exc)
            return False

    def files(self) -> list[Path]:
        out = [self._rotated(i) for i in range(self.keep, 0, -1)] + [self.path]
        return [p for p in out if p.exists()]

    def history(self, *, limit: int, before: str | None, retention_days: int) -> dict:
        """Cursor page over rotated files, with a second structural privacy boundary.

        Only limit+1 candidates remain in memory. Concurrent append/rotation may
        omit newly rotated rows; the cursor prevents repeats of already read rows.
        """
        boundary = None
        if before:
            try:
                value = json.loads(
                    base64.urlsafe_b64decode(before + "=" * (-len(before) % 4))
                )
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or _num(value[0]) is None
                    or not request_label(value[1])
                ):
                    raise ValueError("invalid history cursor")
                boundary = (value[0], value[1])
            except (ValueError, TypeError):
                raise ValueError("invalid history cursor") from None
        cutoff = time.time() - retention_days * 86400 if retention_days else 0
        picked: list[tuple[tuple, int, dict]] = []
        serial = 0
        for path in self.files():
            with contextlib.suppress(OSError):
                with path.open() as source:
                    for line in source:
                        try:
                            raw = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(raw, dict):
                            continue
                        stamp = _num(raw.get("t_end"))
                        if stamp is None or stamp < cutoff:
                            continue
                        row = history_row(raw)
                        key = (stamp, row["request_id"])
                        if boundary is not None and key >= boundary:
                            continue
                        serial += 1
                        heapq.heappush(picked, (key, serial, row))
                        if len(picked) > limit + 1:
                            heapq.heappop(picked)
        ordered = sorted(picked, reverse=True)
        data = [item[2] for item in ordered[:limit]]
        cursor = None
        if len(ordered) > limit:
            cursor = (
                base64.urlsafe_b64encode(json.dumps(ordered[limit - 1][0]).encode())
                .decode()
                .rstrip("=")
            )
        return {
            "object": "list",
            "enabled": True,
            "data": data,
            "count": len(data),
            "next_cursor": cursor,
        }

    def read(self) -> list[dict]:
        """Every parseable event, oldest first; a torn line is skipped, not guessed."""
        rows: list[dict] = []
        for p in self.files():
            with contextlib.suppress(OSError):
                for ln in p.read_text().splitlines():
                    with contextlib.suppress(ValueError):
                        row = json.loads(ln)
                        if isinstance(row, dict):
                            rows.append(row)
        return rows


_instance: ServeLog | None = None
_instance_key: tuple | None = None
_instance_lock = threading.Lock()


def get_log() -> ServeLog | None:
    """The configured log, or None while ``YUNSHU_SERVE_LOG`` is off."""
    global _instance, _instance_key
    if not settings.get_bool("YUNSHU_SERVE_LOG"):
        return None
    configured = settings.get("YUNSHU_SERVE_LOG_DIR")
    directory = Path(configured).expanduser() if configured else paths.home() / "logs"
    max_mb = settings.get_float("YUNSHU_SERVE_LOG_MAX_MB") or 4.0
    keep = settings.get_int("YUNSHU_SERVE_LOG_KEEP") or 0
    key = (str(directory), max_mb, keep)
    with _instance_lock:
        if _instance is None or _instance_key != key:
            _instance = ServeLog(directory, int(max_mb * 1024 * 1024), keep)
            _instance_key = key
        return _instance


def record(info: Any, stats: dict, concurrency_end: int | None = None) -> None:
    """Log one finished request (called from ``x_yunshu.record_done``); a no-op when off."""
    log = get_log()
    if log is None:
        return
    from .x_yunshu import dialect

    st = getattr(info, "stats", None)
    gen = getattr(info, "gen", None)
    path = getattr(info, "path", "") or ""
    ctx = {
        "request_id": getattr(info, "request_id", None),
        "status": getattr(info, "status", None),
        "t_start": getattr(info, "arrived_wall", None),
        "t_end": time.time(),
        "route": path,
        "dialect": dialect(path),
        "stream": getattr(info, "stream", False),
        "model": getattr(gen, "model", None),
        "finish_reason": getattr(st, "finish_reason", None),
        "concurrency_start": getattr(info, "concurrency_start", None),
        "concurrency_end": concurrency_end,
        "arm": current_arm(),
        "build": build_id(),
    }
    log.append(event_from_stats(stats, ctx))
