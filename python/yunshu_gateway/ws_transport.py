"""WebSocket transport for the text APIs.

One socket multiplexes many requests. Each request is dispatched *in process*
into the same ASGI app that serves HTTP (so auth, middleware, request ids and
every handler behave exactly as over SSE); the SSE stream it produces is parsed
and forwarded as JSON messages. Two wire dialects share the machinery:

* ``yunshu`` (``/v1/stream``): request/cancel/stop/update envelopes carrying a
  client request id, for chat.completions, completions, responses and messages.
* ``responses`` (``/v1/responses`` as a WebSocket): OpenAI's Responses
  WebSocket mode -- ``response.create`` in, raw ``response.*`` events out,
  ``stream_id`` lanes (FIFO within a lane, parallel across lanes).

See docs/guides/TRANSPORTS.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from yunshu_engine import settings

logger = logging.getLogger(__name__)

API_PATHS = {
    "chat.completions": "/v1/chat/completions",
    "completions": "/v1/completions",
    "responses": "/v1/responses",
    "messages": "/v1/messages",
}
_FORWARD_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "openai-organization",
    "user-agent",
)
MAX_STREAM_IDS = 32
_anon = itertools.count(1)


class AsgiCall:
    """Run one HTTP request against an ASGI app and stream the response body.

    The body queue is bounded, so a slow consumer blocks ``send`` inside the
    app, which pauses generation (backpressure) instead of buffering.
    """

    def __init__(
        self,
        app,
        *,
        method: str,
        path: str,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
        client=None,
        server=None,
        state: dict | None = None,
        queue_size: int = 64,
    ) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._gone = asyncio.Event()
        self._sent_body = False
        self._closed = False
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self._started = asyncio.Event()
        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": headers + [(b"content-length", str(len(body)).encode())],
            "client": client,
            "server": server,
        }
        if state is not None:
            scope["state"] = dict(state)
        self._scope = scope
        self._body = body
        self._app = app
        self._task = asyncio.create_task(self._run())

    async def _receive(self):
        if not self._sent_body:
            self._sent_body = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        await self._gone.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message) -> None:
        if self._closed:
            # Peer gone (like uvicorn): drop output. The app learns of the
            # disconnect through receive() and stops its generator.
            return
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in message.get("headers", [])
            }
            self._started.set()
        elif message["type"] == "http.response.body":
            data = message.get("body", b"")
            if data:
                await self._queue.put(data)

    async def _run(self) -> None:
        try:
            await self._app(self._scope, self._receive, self._send)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a 500 to the caller
            if not self._closed:
                logger.warning("ws_transport: inner request failed: %r", exc)
            if self.status is None:
                self.status = 500
        finally:
            self._started.set()
            # End-of-body marker; make room if the consumer left the queue full.
            while not self._closed:
                try:
                    self._queue.put_nowait(None)
                    break
                except asyncio.QueueFull:
                    await asyncio.sleep(0.01)

    async def wait_started(self) -> int:
        await self._started.wait()
        return self.status or 500

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            data = await self._queue.get()
            if data is None:
                return
            yield data

    async def close(self) -> None:
        """Tell the app the client is gone (stops generation) and reap the task."""
        self._closed = True
        self._gone.set()
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        if not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except BaseException:  # noqa: BLE001 - timeout or cancel: force it
                self._task.cancel()
                with contextlib.suppress(BaseException):
                    await self._task


class SSEParser:
    """Incremental text/event-stream parser -> (event, data-string) frames."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: bytes) -> list[tuple[str | None, str]]:
        self._buf += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
        frames: list[tuple[str | None, str]] = []
        while "\n\n" in self._buf:
            raw, self._buf = self._buf.split("\n\n", 1)
            event = None
            data: list[str] = []
            for line in raw.split("\n"):
                if not line or line.startswith(":"):
                    continue
                name, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if name == "event":
                    event = value
                elif name == "data":
                    data.append(value)
            if data:
                frames.append((event, "\n".join(data)))
        return frames


def is_delta(api: str, data: dict) -> bool:
    """True when ``data`` carries generated content (about one per token)."""
    if api in ("chat.completions", "completions"):
        for choice in data.get("choices") or ():
            delta = choice.get("delta") or {}
            if (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
                or choice.get("text")
            ):
                return True
        return False
    if api == "responses":
        return str(data.get("type", "")).endswith(".delta")
    if api == "messages":
        return data.get("type") == "content_block_delta"
    return False


class Job:
    def __init__(self, req_id: str, api: str, body: dict, stream_id: str | None):
        self.id = req_id
        self.api = api
        self.body = body
        self.stream_id = stream_id
        self.task: asyncio.Task | None = None
        self.call: AsgiCall | None = None
        self.reason: str | None = None  # cancelled | stopped | max_tokens
        self.max_tokens: int | None = None
        self.deltas = 0
        self.events = 0


class Connection:
    """Server side of one multiplexing WebSocket."""

    def __init__(self, ws: WebSocket, dialect: str) -> None:
        self.ws = ws
        self.dialect = dialect  # "yunshu" | "responses"
        self.out: asyncio.Queue = asyncio.Queue(
            maxsize=int(settings.get("YUNSHU_WS_SEND_QUEUE"))
        )
        self.jobs: dict[str, Job] = {}
        self.lanes: dict[str, asyncio.Lock] = {}
        self.max_inflight = int(settings.get("YUNSHU_WS_MAX_INFLIGHT"))
        self._writer: asyncio.Task | None = None
        self._pinger: asyncio.Task | None = None

    # ── output ──

    async def emit(self, msg: dict) -> None:
        await self.out.put(msg)

    def emit_nowait(self, msg: dict) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self.out.put_nowait(msg)

    async def _write_loop(self) -> None:
        try:
            while True:
                msg = await self.out.get()
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - peer gone; the reader loop ends the session
            pass

    async def _ping_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            # Heartbeats never queue behind a stalled consumer.
            self.emit_nowait({"type": "ping", "t": time.time()})

    # ── lifecycle ──

    async def run(self) -> None:
        self._writer = asyncio.create_task(self._write_loop())
        interval = float(settings.get("YUNSHU_WS_PING_INTERVAL"))
        if interval > 0 and self.dialect == "yunshu":
            self._pinger = asyncio.create_task(self._ping_loop(interval))
        if self.dialect == "yunshu":
            await self.emit(
                {
                    "type": "session.created",
                    "protocol": "yunshu.stream",
                    "apis": list(API_PATHS),
                    "limits": {
                        "max_inflight": self.max_inflight,
                        "max_stream_ids": MAX_STREAM_IDS,
                        "send_queue": self.out.maxsize,
                        "ping_interval_s": interval,
                    },
                }
            )
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                except ValueError:
                    await self._error(None, 400, "Invalid JSON message")
                    continue
                await self._dispatch(msg)
        except WebSocketDisconnect:
            pass
        finally:
            tasks = []
            for job in list(self.jobs.values()):
                job.reason = job.reason or "cancelled"
                if job.task:
                    job.task.cancel()
                    tasks.append(job.task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            side = [t for t in (self._pinger, self._writer) if t]
            for t in side:
                t.cancel()
            await asyncio.gather(*side, return_exceptions=True)

    # ── inbound ──

    async def _error(
        self, req_id: str | None, status: int, message: str, **extra: Any
    ) -> None:
        err = {
            "message": message,
            "type": "invalid_request_error" if status < 500 else "server_error",
        }
        err.update(extra)
        msg: dict = {"type": "error", "status": status, "error": err}
        if req_id is not None and self.dialect == "yunshu":
            msg["id"] = req_id
        await self.emit(msg)

    async def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        if self.dialect == "responses":
            if kind == "response.create":
                body = {
                    k: v
                    for k, v in msg.items()
                    if k not in ("type", "stream_id", "client_request_id")
                }
                sid = msg.get("stream_id")
                rid = msg.get("client_request_id") or f"r{next(_anon)}"
                await self._start(
                    str(rid), "responses", body, None if sid is None else str(sid)
                )
            elif kind == "response.cancel":
                await self._cancel(msg, "cancelled")
            elif kind == "ping":
                await self.emit({"type": "pong"})
            else:
                await self._error(None, 400, f"Unknown event type: {kind!r}")
            return
        if kind == "request":
            api = msg.get("api", "chat.completions")
            if api == "chat":
                api = "chat.completions"
            body = msg.get("body")
            req_id = msg.get("id")
            if not isinstance(req_id, (str, int)) or req_id == "":
                await self._error(None, 400, "'id' (client request id) is required")
                return
            if api not in API_PATHS or not isinstance(body, dict):
                await self._error(
                    str(req_id),
                    400,
                    f"'api' must be one of {list(API_PATHS)} and 'body' an object",
                )
                return
            sid = msg.get("stream_id")
            await self._start(
                str(req_id), api, dict(body), None if sid is None else str(sid)
            )
        elif kind == "cancel":
            await self._cancel(msg, "cancelled")
        elif kind == "stop":
            await self._cancel(msg, "stopped")
        elif kind == "update":
            job = self.jobs.get(str(msg.get("id")))
            mt = msg.get("max_tokens")
            if job is None:
                await self._error(str(msg.get("id")), 404, "No such active request")
            elif not isinstance(mt, int) or isinstance(mt, bool) or mt < 0:
                await self._error(job.id, 400, "'max_tokens' must be an integer >= 0")
            else:
                job.max_tokens = mt
                await self.emit({"type": "updated", "id": job.id, "max_tokens": mt})
        elif kind == "ping":
            await self.emit({"type": "pong", "t": msg.get("t")})
        elif kind == "pong":
            pass
        else:
            await self._error(None, 400, f"Unknown message type: {kind!r}")

    async def _cancel(self, msg: dict, reason: str) -> None:
        key = msg.get("id", msg.get("stream_id"))
        targets: list[Job] = []
        if key is not None and str(key) in self.jobs:
            targets = [self.jobs[str(key)]]
        elif key is not None:
            targets = [j for j in self.jobs.values() if j.stream_id == str(key)]
        elif self.dialect == "responses":
            targets = list(self.jobs.values())
        if not targets:
            await self._error(
                None if key is None else str(key), 404, "No such active request"
            )
            return
        for job in targets:
            job.reason = job.reason or reason
            if job.task:
                job.task.cancel()

    async def _start(
        self, req_id: str, api: str, body: dict, stream_id: str | None
    ) -> None:
        if req_id in self.jobs:
            await self._error(req_id, 409, f"Request id {req_id!r} is already active")
            return
        if len(self.jobs) >= self.max_inflight:
            await self._error(
                req_id,
                429,
                f"Too many concurrent requests on this connection (max {self.max_inflight})",
                code="too_many_requests",
            )
            return
        if (
            stream_id is not None
            and stream_id not in self.lanes
            and len(self.lanes) >= MAX_STREAM_IDS
        ):
            await self._error(
                req_id, 429, f"Too many stream_ids (max {MAX_STREAM_IDS})"
            )
            return
        body["stream"] = True
        if api in ("chat.completions", "completions"):
            so = body.get("stream_options")
            if not isinstance(so, dict):
                body["stream_options"] = {"include_usage": True}
            else:
                so.setdefault("include_usage", True)
        job = Job(req_id, api, body, stream_id)
        if stream_id is not None:
            self.lanes.setdefault(stream_id, asyncio.Lock())
        self.jobs[req_id] = job
        job.task = asyncio.create_task(self._run_job(job))

    # ── one request ──

    def _headers(self, job: Job) -> list[tuple[bytes, bytes]]:
        h = [
            (b"content-type", b"application/json"),
            (b"accept", b"text/event-stream"),
            (b"x-request-id", job.id.encode()),
        ]
        src = {k.lower(): v for k, v in self.ws.headers.items()}
        for name in _FORWARD_HEADERS:
            if name in src:
                h.append((name.encode(), src[name].encode("latin-1")))
        if "authorization" not in src:
            token = self.ws.query_params.get("token")
            if token:
                h.append((b"authorization", f"Bearer {token}".encode()))
        return h

    def _wrap(self, job: Job, event: str | None, data: Any) -> dict:
        if self.dialect == "responses":
            return data if isinstance(data, dict) else {"type": "raw", "data": data}
        return {"type": "event", "id": job.id, "event": event, "data": data}

    async def _run_job(self, job: Job) -> None:
        lock = self.lanes.get(job.stream_id) if job.stream_id is not None else None
        acquired = False
        t0 = time.perf_counter()
        ttft: float | None = None
        status = "completed"
        try:
            if lock is not None:
                await lock.acquire()
                acquired = True
            job.call = AsgiCall(
                self.ws.scope["app"],
                method="POST",
                path=API_PATHS[job.api],
                headers=self._headers(job),
                body=json.dumps(job.body).encode(),
                client=self.ws.scope.get("client"),
                server=self.ws.scope.get("server"),
                state=self.ws.scope.get("state"),
            )
            code = await job.call.wait_started()
            ctype = job.call.headers.get("content-type", "")
            if code >= 400 or "text/event-stream" not in ctype:
                raw = b"".join([c async for c in job.call.chunks()])
                try:
                    payload = json.loads(raw or b"{}")
                except ValueError:
                    payload = {"message": raw.decode("utf-8", "replace")}
                if code >= 400:
                    err = payload.get("error", payload)
                    if not isinstance(err, dict):
                        err = {"message": str(err)}
                    msg: dict = {"type": "error", "status": code, "error": err}
                    if self.dialect == "yunshu":
                        msg["id"] = job.id
                    await self.emit(msg)
                    status = "error"
                else:
                    await self.emit(self._wrap(job, None, payload))
            else:
                parser = SSEParser()
                async for chunk in job.call.chunks():
                    for event, data in parser.feed(chunk):
                        if data.strip() == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            obj = data
                        job.events += 1
                        if isinstance(obj, dict) and is_delta(job.api, obj):
                            job.deltas += 1
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                        await self.emit(self._wrap(job, event, obj))
                        if job.max_tokens is not None and job.deltas >= job.max_tokens:
                            job.reason = "max_tokens"
                            raise asyncio.CancelledError
        except asyncio.CancelledError:
            status = job.reason or "cancelled"
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_transport: request %s failed: %r", job.id, exc)
            status = "error"
            with contextlib.suppress(Exception):
                await self._error(job.id, 500, f"{type(exc).__name__}: {exc}")
        finally:
            if acquired and lock is not None:
                lock.release()
            if job.call is not None:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(job.call.close())
            self.jobs.pop(job.id, None)
            if job.stream_id is not None and not any(
                j.stream_id == job.stream_id for j in self.jobs.values()
            ):
                self.lanes.pop(job.stream_id, None)
            if self.dialect == "yunshu":
                done = {
                    "type": "done",
                    "id": job.id,
                    "reason": status,
                    "stats": {
                        "ttft_ms": None if ttft is None else round(ttft * 1000, 1),
                        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "events": job.events,
                        "deltas": job.deltas,
                    },
                }
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(self.emit(done), timeout=5.0)
