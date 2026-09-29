"""Yunshu API extensions: request ids, queue visibility, prefill progress, per-response stats.

Everything here is additive and namespaced, so the OpenAI / Anthropic / Ollama SDKs keep
working unchanged:

- ``X-Request-Id`` is accepted (or generated) for every request and echoed on every response,
  errors included; the same id cancels the generation (``DELETE /v1/requests/{id}``).
- streaming responses get ``: yunshu-progress {...}`` SSE *comment* lines while the request
  waits or prefills (comments are ignored by every SSE parser), and a final ``x_yunshu`` object
  in the usage chunk (plus a ``: yunshu-stats`` comment when the client did not ask for usage).
- non-streaming JSON responses get a top-level ``x_yunshu`` object and ``X-Yunshu-*`` headers.
- ``X-Yunshu-Queue-Position`` / ``X-Yunshu-Queue-Est-Wait-Ms`` say how busy the server was when
  the request arrived.

The measurements come from the engine's live ``RunStats`` (the VLM batch runner) when the
request reaches it, and from the gateway's own clock otherwise, so the fields are best-effort
and ``null`` when unknown.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from yunshu_engine import settings
from yunshu_engine.request_tracker import current_request_id, current_request_info

REQUEST_ID_HEADER = "X-Request-Id"
_ID_OK = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

# Generation endpoints whose requests are tracked (queue position, progress, cancel by id).
TRACKED_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/responses",
        "/v1/messages",
        "/messages",
    }
)
# Endpoints whose JSON / SSE bodies get the ``x_yunshu`` object.
STATS_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/messages",
        "/messages",
        "/v1/responses",
    }
)
_RESPONSES_TERMINAL = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)
_RESPONSES_DELTAS = frozenset(
    {
        "response.output_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.refusal.delta",
    }
)


def dialect(path: str) -> str:
    """Which wire format a tracked path speaks: chat (OpenAI chat / completions),
    anthropic (Messages) or responses (OpenAI Responses)."""
    if path.endswith("/messages"):
        return "anthropic"
    if path.endswith("/responses"):
        return "responses"
    return "chat"


def normalize_usage(usage: dict | None) -> dict:
    """Any dialect's usage object as chat-style ``prompt_tokens`` / ``completion_tokens`` /
    ``prompt_tokens_details.cached_tokens`` (what build_stats reads); ``{}`` when empty."""
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt is not None:
        out["prompt_tokens"] = prompt
    if completion is not None:
        out["completion_tokens"] = completion
    cached = (
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or (usage.get("input_tokens_details") or {}).get("cached_tokens")
        or usage.get("cache_read_input_tokens")
    )
    if cached:
        out["prompt_tokens_details"] = {"cached_tokens": cached}
    return out


def sanitize_request_id(raw: str | bytes | None) -> str | None:
    """A client-supplied request id when it is short and header-safe, else None."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("latin-1", "replace")
    raw = raw.strip()
    return raw if _ID_OK.match(raw) else None


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


@dataclass
class RequestInfo:
    """Per-HTTP-request live record (one per tracked generation request)."""

    request_id: str
    method: str
    path: str
    arrived: float = field(default_factory=time.perf_counter)
    arrived_wall: float = field(default_factory=time.time)
    gen: Any = None  # ActiveGeneration, linked by RequestTracker.register()
    engine_request_id: str | None = None
    stream: bool = False
    status: int = 0
    t_first_chunk: float | None = None  # first content event seen on the wire
    t_done: float | None = None
    usage: dict | None = None
    queue_position: int = 0
    queue_est_wait_ms: float = 0.0
    cancel_requested: bool = False
    _rate: tuple[float, int, float] | None = None  # (t, processed, ema tokens/s)

    @property
    def stats(self):
        return self.gen.stats if self.gen is not None else None


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, RequestInfo] = {}
        self._recent: collections.deque[dict] = collections.deque(maxlen=512)

    def add(self, info: RequestInfo) -> None:
        with self._lock:
            self._active[info.request_id] = info

    def remove(self, info: RequestInfo) -> None:
        with self._lock:
            if self._active.get(info.request_id) is info:
                del self._active[info.request_id]

    def get(self, request_id: str) -> RequestInfo | None:
        with self._lock:
            info = self._active.get(request_id)
            if info is not None:
                return info
            for i in self._active.values():
                if i.engine_request_id == request_id:
                    return i
        return None

    def active(self) -> list[RequestInfo]:
        with self._lock:
            return sorted(self._active.values(), key=lambda i: i.arrived)

    def record_done(self, entry: dict) -> None:
        with self._lock:
            self._recent.append(entry)

    def recent(self, window_s: float) -> list[dict]:
        cutoff = time.time() - window_s
        with self._lock:
            return [e for e in self._recent if e["t"] >= cutoff]

    def clear(self) -> None:
        with self._lock:
            self._active.clear()
            self._recent.clear()


registry = _Registry()


# ── queue / progress ────────────────────────────────────────────────────


def _remaining_prefill_s(st) -> float:
    """Best-effort seconds of prefill left for one request's stats (0 when unknown)."""
    if st is None:
        return 0.0
    left = max(st.prefill_total - st.prefill_done, 0)
    if left <= 0 or st.t_first:
        return 0.0
    if st.t_admit and st.prefill_done > 0:
        rate = st.prefill_done / max(time.perf_counter() - st.t_admit, 1e-6)
        return left / rate if rate > 0 else 0.0
    rate = recent_rates().get("prefill_tps") or 0.0
    return left / rate if rate > 0 else 0.0


def queue_snapshot(info: RequestInfo) -> tuple[int, float]:
    """(requests ahead of ``info``, lower-bound wait in ms).

    The wait is the prefill still owed by the requests ahead: on the shared GPU thread a new
    request's prefill starts after theirs, so it is a lower bound of the time to its first
    token, not a promise.
    """
    ahead = [
        i for i in registry.active() if i is not info and i.arrived <= info.arrived
    ]
    wait = sum(_remaining_prefill_s(i.stats) for i in ahead)
    return len(ahead), round(wait * 1000, 1)


def recent_rates(window_s: float = 300.0) -> dict[str, float | None]:
    """Mean prefill / decode speed of the requests that finished in the last window."""
    rows = registry.recent(window_s)

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key)]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {"prefill_tps": mean("prefill_tps"), "decode_tps": mean("decode_tps")}


def progress_payload(info: RequestInfo) -> dict:
    """Live state of one request: queued / prefill (with tokens, %, ETA) / decode."""
    now = time.perf_counter()
    st = info.stats
    out: dict[str, Any] = {
        "request_id": info.request_id,
        "elapsed_s": round(now - info.arrived, 2),
    }
    if st is None:
        ahead, wait_ms = queue_snapshot(info)
        out["phase"] = "queued" if ahead else "starting"
        out["queue_position"] = ahead
        if ahead:
            out["queue_est_wait_ms"] = wait_ms
        return out
    out["phase"] = st.phase
    out["prompt_tokens"] = st.prompt_tokens or (st.prefill_total + st.cached_tokens)
    out["cached_tokens"] = st.cached_tokens
    if st.phase == "queued":
        ahead, wait_ms = queue_snapshot(info)
        out["queue_position"] = ahead
        out["queue_est_wait_ms"] = wait_ms
    elif st.phase == "prefill":
        done, total = st.prefill_done, max(st.prefill_total, 1)
        out["processed_tokens"] = done + st.cached_tokens
        out["percent"] = round(100.0 * done / total, 1)
        rate = _prefill_rate(info, done, now)
        out["tokens_per_second"] = round(rate, 1) if rate else None
        out["eta_s"] = round((total - done) / rate, 1) if rate else None
    else:
        out["completion_tokens"] = st.generated
    return out


def _prefill_rate(info: RequestInfo, done: int, now: float) -> float | None:
    """Prefill tokens/s: EMA over successive samples, overall rate as the fallback."""
    st = info.stats
    prev = info._rate
    rate: float | None = None
    if prev is not None and now > prev[0] and done > prev[1]:
        inst = (done - prev[1]) / (now - prev[0])
        rate = inst if not prev[2] else 0.5 * prev[2] + 0.5 * inst
    elif prev is not None:
        rate = prev[2] or None
    if rate is None and st is not None and st.t_admit and done > 0:
        rate = done / max(now - st.t_admit, 1e-6)
    if prev is None or done != prev[1]:
        info._rate = (now, done, rate or 0.0)
    return rate


def progress_comment(info: RequestInfo) -> bytes:
    return (
        ": yunshu-progress "
        + json.dumps(progress_payload(info), separators=(",", ":"))
        + "\n\n"
    ).encode()


# ── per-response stats ──────────────────────────────────────────────────


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000.0, 1)


def build_stats(info: RequestInfo, usage: dict | None = None) -> dict:
    """The ``x_yunshu`` object: TTFT, prefill / decode speed, cache hits, speculation, queue wait."""
    now = time.perf_counter()
    end = info.t_done or now
    st = info.stats
    usage = usage or info.usage or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if st is not None:
        if prompt_tokens is None:
            prompt_tokens = st.prompt_tokens or None
        if completion_tokens is None:
            completion_tokens = st.generated
        if not cached:
            cached = st.cached_tokens
    cached = int(cached or 0)

    t_first = st.t_first if st is not None and st.t_first else None
    if t_first is None and info.t_first_chunk is not None:
        t_first = info.t_first_chunk
    ttft = (t_first - info.arrived) if t_first is not None else None
    queue_wait = (
        (st.t_admit - st.t_submit)
        if st is not None and st.t_admit and st.t_submit
        else None
    )
    prefill_s = (
        (t_first - st.t_admit)
        if (st is not None and st.t_admit and t_first is not None)
        else None
    )
    fresh = (prompt_tokens - cached) if prompt_tokens is not None else None
    prefill_tps = (
        round(fresh / prefill_s, 1)
        if (prefill_s and prefill_s > 0 and fresh and fresh > 0)
        else None
    )
    t_last = st.t_last if st is not None and st.t_last else end
    decode_s = (t_last - t_first) if t_first is not None else None
    decode_tps = (
        round((completion_tokens - 1) / decode_s, 1)
        if (decode_s and decode_s > 0 and completion_tokens and completion_tokens > 1)
        else None
    )
    spec = None
    if st is not None and st.spec_mode:
        spec = {
            "mode": st.spec_mode,
            "drafted": st.spec_drafted or None,
            "accepted": st.spec_accepted or None,
            "acceptance_rate": (
                round(st.spec_accepted / st.spec_drafted, 3)
                if st.spec_drafted
                else None
            ),
        }
    return {
        "request_id": info.request_id,
        "queue_wait_ms": _ms(queue_wait),
        "ttft_ms": _ms(ttft),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "prefill_ms": _ms(prefill_s),
        "prefill_tps": prefill_tps,
        "completion_tokens": completion_tokens,
        "decode_ms": _ms(decode_s),
        "decode_tps": decode_tps,
        "total_ms": _ms(end - info.arrived),
        "speculative": spec,
        # llama.cpp `timings` field names, for tools that already read them.
        "timings": {
            "cache_n": cached,
            "prompt_n": fresh,
            "prompt_ms": _ms(prefill_s),
            "prompt_per_second": prefill_tps,
            "predicted_n": completion_tokens,
            "predicted_ms": _ms(decode_s),
            "predicted_per_second": decode_tps,
        },
    }


def stats_headers(stats: dict, info: RequestInfo) -> list[tuple[bytes, bytes]]:
    """Compact ``X-Yunshu-*`` headers for a non-streaming response."""
    pairs: list[tuple[str, Any]] = [
        ("X-Yunshu-Queue-Wait-Ms", stats.get("queue_wait_ms")),
        ("X-Yunshu-TTFT-Ms", stats.get("ttft_ms")),
        ("X-Yunshu-Prefill-Tps", stats.get("prefill_tps")),
        ("X-Yunshu-Decode-Tps", stats.get("decode_tps")),
        ("X-Yunshu-Cached-Tokens", stats.get("cached_tokens")),
        ("X-Yunshu-Total-Ms", stats.get("total_ms")),
    ]
    spec = stats.get("speculative")
    if spec:
        pairs.append(("X-Yunshu-Spec", spec.get("mode")))
        pairs.append(("X-Yunshu-Spec-Acceptance", spec.get("acceptance_rate")))
    return [(k.lower().encode(), str(v).encode()) for k, v in pairs if v is not None]


def queue_headers(info: RequestInfo) -> list[tuple[bytes, bytes]]:
    return [
        (b"x-yunshu-queue-position", str(info.queue_position).encode()),
        (b"x-yunshu-queue-est-wait-ms", str(info.queue_est_wait_ms).encode()),
    ]


def record_done(info: RequestInfo, stats: dict) -> None:
    registry.record_done(
        {
            "t": time.time(),
            "request_id": info.request_id,
            "prompt_tokens": stats.get("prompt_tokens") or 0,
            "completion_tokens": stats.get("completion_tokens") or 0,
            "prefill_tps": stats.get("prefill_tps"),
            "decode_tps": stats.get("decode_tps"),
            "ttft_ms": stats.get("ttft_ms"),
        }
    )


# ── ASGI middleware ─────────────────────────────────────────────────────


def _is_usage_event(obj: dict) -> bool:
    return bool(obj.get("usage")) and not obj.get("choices")


def _has_content(obj: dict) -> bool:
    for ch in obj.get("choices") or []:
        d = ch.get("delta") or {}
        if (
            d.get("content")
            or d.get("reasoning_content")
            or d.get("reasoning")
            or d.get("tool_calls")
        ):
            return True
        if ch.get("text"):
            return True
    return False


class YunshuExtensionsMiddleware:
    """Pure-ASGI layer (does not buffer streams) adding the extensions listed above."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw_headers = dict(scope.get("headers") or [])
        request_id = sanitize_request_id(raw_headers.get(b"x-request-id"))
        request_id = request_id or new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        path = scope.get("path", "")
        tracked = scope.get("method") == "POST" and path in TRACKED_PATHS
        info = RequestInfo(request_id, scope.get("method", ""), path)
        tokens = [(current_request_id, current_request_id.set(request_id))]
        if tracked:
            info.queue_position, info.queue_est_wait_ms = queue_snapshot(info)
            registry.add(info)
            tokens.append((current_request_info, current_request_info.set(info)))
        try:
            await self._serve(scope, receive, send, info, tracked)
        finally:
            info.t_done = time.perf_counter()
            if tracked:
                registry.remove(info)
                if info.status == 200 and path in STATS_PATHS:
                    with contextlib.suppress(Exception):
                        record_done(info, build_stats(info))
            for var, tok in reversed(tokens):
                with contextlib.suppress(ValueError):
                    var.reset(tok)

    async def _serve(self, scope, receive, send, info: RequestInfo, tracked: bool):
        send_lock = asyncio.Lock()
        state: dict[str, Any] = {
            "mode": "pass",  # pass | sse | json
            "start": None,
            "body": bytearray(),
            "done": False,
            "task": None,
            "sent_stats": False,
        }
        rid_header = (REQUEST_ID_HEADER.lower().encode(), info.request_id.encode())
        stats_path = info.path in STATS_PATHS
        kind = dialect(info.path)

        async def emit(message) -> None:
            async with send_lock:
                await send(message)

        async def progress_loop() -> None:
            interval = settings.get("YUNSHU_PROGRESS_INTERVAL_S") or 0.0
            if interval <= 0:
                return
            while not state["done"] and info.t_first_chunk is None:
                await asyncio.sleep(interval)
                st = info.stats
                if state["done"] or info.t_first_chunk is not None:
                    return
                if st is not None and st.phase in ("decode", "done"):
                    return
                try:
                    await emit(
                        {
                            "type": "http.response.body",
                            "body": progress_comment(info),
                            "more_body": True,
                        }
                    )
                except Exception:
                    return

        def attach(obj: dict, usage: dict) -> None:
            """Put ``x_yunshu`` inside a usage object (extra fields there are tolerated
            by the OpenAI / Anthropic SDK models)."""
            norm = normalize_usage(usage)
            if kind != "chat" and info.stats is not None:
                norm.pop("prompt_tokens", None)  # the engine's own count is exact
            info.usage = norm or info.usage
            info.t_done = time.perf_counter()
            usage["x_yunshu"] = build_stats(info, norm)
            state["sent_stats"] = True

        def handle_event(obj: dict) -> bool:
            """First-token detection and stats injection for one SSE event; True if changed."""
            t = obj.get("type")
            if kind == "chat":
                if info.t_first_chunk is None and _has_content(obj):
                    info.t_first_chunk = time.perf_counter()
                if _is_usage_event(obj):
                    attach_chat(obj)
                    return True
            elif kind == "anthropic":
                if info.t_first_chunk is None and t == "content_block_delta":
                    info.t_first_chunk = time.perf_counter()
                if t == "message_delta" and isinstance(obj.get("usage"), dict):
                    attach(obj, obj["usage"])
                    return True
            else:
                if info.t_first_chunk is None and t in _RESPONSES_DELTAS:
                    info.t_first_chunk = time.perf_counter()
                if t in _RESPONSES_TERMINAL:
                    u = (obj.get("response") or {}).get("usage")
                    if isinstance(u, dict):
                        attach(obj, u)
                        return True
            return False

        def attach_chat(obj: dict) -> None:
            info.usage = obj["usage"]
            info.t_done = time.perf_counter()
            obj["x_yunshu"] = build_stats(info, obj["usage"])
            state["sent_stats"] = True

        def rewrite_sse(chunk: bytes) -> bytes:
            """Stats injection + first-token detection; cheap checks first."""
            need_parse = info.t_first_chunk is None or b'"usage"' in chunk
            has_done = b"data: [DONE]" in chunk
            if not (need_parse or has_done):
                return chunk
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                return chunk
            out: list[str] = []
            changed = False
            for part in text.split("\n\n"):
                if not part:
                    continue
                lines = part.split("\n")
                di = next((i for i, ln in enumerate(lines) if ln.startswith("data:")), -1)
                if di >= 0 and lines[di][5:].strip() != "[DONE]":
                    try:
                        obj = json.loads(lines[di][5:])
                    except ValueError:
                        out.append(part)
                        continue
                    if isinstance(obj, dict) and handle_event(obj):
                        lines[di] = "data: " + json.dumps(obj, ensure_ascii=False)
                        out.append("\n".join(lines))
                        changed = True
                        continue
                elif kind == "chat" and part.strip() == "data: [DONE]":
                    if not state["sent_stats"]:
                        info.t_done = time.perf_counter()
                        out.append(
                            ": yunshu-stats "
                            + json.dumps(build_stats(info), separators=(",", ":"))
                        )
                        state["sent_stats"] = True
                        changed = True
                out.append(part)
            if not changed:
                return chunk
            return ("\n\n".join(out) + "\n\n").encode("utf-8")

        async def wrapped_send(message) -> None:
            mtype = message["type"]
            if mtype == "http.response.start":
                info.status = int(message["status"])
                headers = list(message.get("headers") or [])
                ctype = b""
                for k, v in headers:
                    if k.lower() == b"content-type":
                        ctype = v.lower()
                headers = [(k, v) for k, v in headers if k.lower() != rid_header[0]]
                headers.append(rid_header)
                if (
                    tracked
                    and info.status == 200
                    and ctype.startswith(b"text/event-stream")
                ):
                    info.stream = True
                    if stats_path:
                        state["mode"] = "sse"
                    headers.extend(queue_headers(info))
                    state["task"] = asyncio.create_task(progress_loop())
                elif (
                    tracked
                    and stats_path
                    and info.status == 200
                    and ctype.startswith(b"application/json")
                ):
                    state["mode"] = "json"
                    state["start"] = {**message, "headers": headers}
                    return
                elif tracked:
                    headers.extend(queue_headers(info))
                await emit({**message, "headers": headers})
                return
            if mtype != "http.response.body":
                await emit(message)
                return
            mode = state["mode"]
            more = message.get("more_body", False)
            if mode == "sse":
                body = message.get("body", b"")
                if body:
                    message = {**message, "body": rewrite_sse(body)}
                if not more:
                    state["done"] = True
                await emit(message)
                return
            if mode == "json":
                state["body"] += message.get("body", b"")
                if more:
                    return
                body = bytes(state["body"])
                start = state["start"]
                headers = [
                    (k, v)
                    for k, v in start["headers"]
                    if k.lower() != b"content-length"
                ]
                try:
                    obj = json.loads(body)
                except ValueError:
                    obj = None
                if kind == "chat":
                    ok = isinstance(obj, dict) and (obj.get("usage") or obj.get("choices"))
                else:
                    ok = (
                        isinstance(obj, dict)
                        and obj.get("type", obj.get("object")) in ("message", "response")
                        and isinstance(obj.get("usage"), dict)
                    )
                if ok:
                    if kind == "chat":
                        info.usage = obj.get("usage")
                        info.t_done = time.perf_counter()
                        stats = build_stats(info, obj.get("usage"))
                        obj["x_yunshu"] = stats
                    else:
                        attach(obj, obj["usage"])
                        stats = obj["usage"]["x_yunshu"]
                    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                    headers.extend(stats_headers(stats, info))
                headers.extend(queue_headers(info))
                headers.append((b"content-length", str(len(body)).encode()))
                await emit({**start, "headers": headers})
                await emit({"type": "http.response.body", "body": body})
                return
            if not more:
                state["done"] = True
            await emit(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            state["done"] = True
            task = state["task"]
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


# ── keep_alive ──────────────────────────────────────────────────────────

_DURATION = re.compile(r"^(-?\d+(?:\.\d+)?)(ms|s|m|h|d)?$")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_keep_alive(value: Any) -> float | None:
    """Ollama-style keep_alive to seconds: ``None`` = server default, ``inf`` = keep
    loaded (any negative value), ``0`` = free the model when idle. Accepts numbers
    (seconds) and duration strings such as "5m", "30s", "1h", "-1"."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(
            "keep_alive must be a number of seconds or a duration like '5m'"
        )
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        m = _DURATION.match(str(value).strip().lower())
        if not m:
            raise ValueError(
                f"invalid keep_alive {value!r}: use seconds (300), a duration ('5m', '1h'), "
                "-1 to keep the model loaded, or 0 to free it when idle"
            )
        seconds = float(m.group(1)) * _UNIT_S[m.group(2) or "s"]
    return float("inf") if seconds < 0 else seconds


def apply_keep_alive(model: str, value: Any) -> None:
    """Record a request's keep_alive on its model (multi-model mode; a single-model
    server never frees its model). Unknown models are left to the request's own 404."""
    if value is None:
        return
    from fastapi import HTTPException

    try:
        seconds = parse_keep_alive(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    from .engine import get_model_manager

    manager = get_model_manager()
    if manager is not None:
        manager.set_keep_alive(model, seconds)
