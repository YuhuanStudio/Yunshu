"""Reusable fault-injection harness for the failure-mode tests.

Everything here is CPU only: a scripted fake engine (a real ``BatchedEngine`` subclass, so the
routers take their normal text path), an ASGI driver that can disconnect at an exact chunk,
SSE parsers with a per-dialect "exactly one terminal" check, a leak audit (tracker entries, live
engine calls, threads) and fake disks. No model, no Metal, no sockets.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import threading
import time
from collections import namedtuple
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from yunshu_engine.batched_engine import BatchedEngine

# --------------------------------------------------------------------------- fake engine


class _Tok:
    def encode(self, text, *a, **k):
        return list(range(len(text)))


class Script:
    """What the fake engine does for one request, step by step.

    Steps: ``("text", "hi")``, ``("sleep", seconds)``, ``("raise", exc)``, ``("hang",)``
    (block until the request's cancel event is set, then end with finish ``cancel``) and
    ``("finish", reason)`` (the default ending is ``stop``).
    """

    def __init__(self, *steps: tuple):
        self.steps = list(steps)

    @classmethod
    def ok(cls, *words: str) -> Script:
        return cls(*[("text", w) for w in (words or ("Hello", " world"))])

    @classmethod
    def error_after(cls, n: int, exc: BaseException) -> Script:
        return cls(*[("text", f"t{i} ") for i in range(n)], ("raise", exc))

    @classmethod
    def hang_after(cls, n: int = 0) -> Script:
        return cls(*[("text", f"t{i} ") for i in range(n)], ("hang",))


class FakeBatchedEngine(BatchedEngine):
    """A loaded engine whose ``chat`` / ``stream_chat`` follow a :class:`Script`.

    It honours ``cancel_event`` and ``timeout_seconds`` like the real loops and counts the calls
    that are still running (``inflight``), so a test can prove nothing was left behind.
    """

    def __init__(self, script: Script | None = None, model: str = "fault-model"):
        super().__init__()
        self._model = object()
        self._loaded = True
        self._running = True
        self._tokenizer = _Tok()
        self.model_name = model
        self.script = script or Script.ok()
        self.calls: list[dict] = []
        self.inflight = 0
        self.cancel_events: list[Any] = []
        self.finish_reasons: list[str] = []
        # (messages, budget in tokens): run the real context-window manager on every call, the
        # way the engine does, and publish what it removed to the request
        self.truncate: tuple[list[dict], int] | None = None
        # one request runs at a time (like the single MLX executor); the others wait, queued,
        # and a cancel ends a waiting request without it ever running
        self.serial: asyncio.Semaphore | None = None
        self.waiting = 0

    async def _run(self, kwargs: dict):
        cancel = kwargs.get("cancel_event")
        sem = self.serial
        if sem is not None:
            self.waiting += 1
            try:
                while True:
                    try:
                        await asyncio.wait_for(sem.acquire(), 0.01)
                        break
                    except TimeoutError:
                        if cancel is not None and cancel.is_set():
                            self.finish_reasons.append("cancel")
                            yield SimpleNamespace(
                                new_text="",
                                finish_reason="cancel",
                                finished=True,
                                prompt_tokens=7,
                                completion_tokens=0,
                                current_state=None,
                                error=None,
                            )
                            return
            finally:
                self.waiting -= 1
            try:
                async for o in self._run_inner(kwargs):
                    yield o
            finally:
                sem.release()
            return
        async for o in self._run_inner(kwargs):
            yield o

    async def _run_inner(self, kwargs: dict):
        cancel = kwargs.get("cancel_event")
        timeout = kwargs.get("timeout_seconds")
        self.calls.append(kwargs)
        self.cancel_events.append(cancel)
        self.inflight += 1
        if self.truncate is not None:
            from yunshu_engine.context_window import ContextWindowManager

            msgs, budget = self.truncate
            mgr = ContextWindowManager(token_counter=lambda t: len(t.split()))
            mgr.compute_truncation(msgs, budget, "truncate_oldest").publish()
        t0 = time.monotonic()
        n = 0
        reason = "stop"
        ended = False
        try:
            for step in self.script.steps:
                kind = step[0]
                if cancel is not None and cancel.is_set():
                    reason = "cancel"
                    break
                if timeout and time.monotonic() - t0 > timeout:
                    reason = "timeout"
                    break
                if kind == "text":
                    n += 1
                    yield SimpleNamespace(
                        new_text=step[1],
                        finish_reason=None,
                        finished=False,
                        prompt_tokens=7,
                        completion_tokens=n,
                        current_state=None,
                        error=None,
                    )
                elif kind == "sleep":
                    await asyncio.sleep(step[1])
                elif kind == "raise":
                    raise step[1]
                elif kind == "finish":
                    reason = step[1]
                elif kind == "hang":
                    while True:
                        if cancel is not None and cancel.is_set():
                            reason = "cancel"
                            break
                        if timeout and time.monotonic() - t0 > timeout:
                            reason = "timeout"
                            break
                        await asyncio.sleep(0.005)
                    break
            self.finish_reasons.append(reason)
            ended = True
            yield SimpleNamespace(
                new_text="",
                finish_reason=reason,
                finished=True,
                prompt_tokens=7,
                completion_tokens=n,
                current_state=None,
                error=None,
            )
        finally:
            self.inflight -= 1
            if not ended:  # the consumer closed us, or the script raised
                self.finish_reasons.append("aborted")

    async def stream_chat(self, **kwargs):
        async for o in self._run(kwargs):
            yield o

    async def chat(self, **kwargs):
        text, last = [], None
        async for o in self._run(kwargs):
            text.append(o.new_text)
            last = o
        return SimpleNamespace(
            text="".join(text),
            prompt_tokens=7,
            completion_tokens=last.completion_tokens if last else 0,
            finish_reason=last.finish_reason if last else "stop",
            logprobs=None,
            reasoning_tokens=0,
            cached_tokens=0,
        )


# --------------------------------------------------------------------------- ASGI driver


@dataclass
class Reply:
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[bytes] = field(default_factory=list)
    disconnected: bool = False
    elapsed: float = 0.0

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.body)


async def asgi_call(
    app,
    method: str,
    path: str,
    body: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
    disconnect_after_chunks: int | None = None,
    disconnect_after_s: float | None = None,
    fail_send_after_chunks: int | None = None,
    timeout: float = 15.0,
) -> Reply:
    """Drive one request through the whole ASGI stack. ``disconnect_after_*`` makes the client
    vanish (``http.disconnect``) after that many body chunks / seconds, the way a closed socket
    does. The call returns when the app finishes, which proves the app noticed.
    ``fail_send_after_chunks`` makes ``send`` raise ``OSError`` after that many chunks, the way
    uvicorn does when the socket is already gone (the app is then abandoned mid-iteration)."""
    raw = json.dumps(body).encode() if body is not None else b""
    hdrs = [(b"host", b"testserver"), (b"content-type", b"application/json")]
    for k, v in (headers or {}).items():
        hdrs.append((k.lower().encode(), v.encode()))
    hdrs.append((b"content-length", str(len(raw)).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": hdrs,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
        "root_path": "",
        "state": {},
    }
    reply = Reply()
    sent_body = False
    gone = asyncio.Event()
    t0 = time.monotonic()

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            reply.status = msg["status"]
            reply.headers = {
                k.decode().lower(): v.decode() for k, v in msg.get("headers", [])
            }
        elif msg["type"] == "http.response.body" and msg.get("body"):
            if (
                fail_send_after_chunks is not None
                and len(reply.chunks) >= fail_send_after_chunks
            ):
                reply.disconnected = True
                raise OSError("client went away")
            reply.chunks.append(msg["body"])
            if (
                disconnect_after_chunks is not None
                and len(reply.chunks) >= disconnect_after_chunks
            ):
                reply.disconnected = True
                gone.set()

    async def timer():
        if disconnect_after_s is not None:
            await asyncio.sleep(disconnect_after_s)
            reply.disconnected = True
            gone.set()

    tt = asyncio.create_task(timer())
    try:
        try:
            await asyncio.wait_for(app(scope, receive, send), timeout)
        except OSError:
            if not reply.disconnected:  # uvicorn logs this and moves on
                raise
    finally:
        tt.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tt
    reply.elapsed = time.monotonic() - t0
    return reply


@contextlib.asynccontextmanager
async def running_app(app, *, warm: bool = True):
    """Run the app's lifespan (startup / shutdown) around a block. ``warm`` sends one request
    per dialect first, so lazy imports do not eat into the timings the test asserts."""
    from yunshu_gateway import engine as engine_mod

    async with app.router.lifespan_context(app):
        eng = engine_mod._engine
        if warm and isinstance(eng, FakeBatchedEngine):
            saved = (eng.script, list(eng.calls), list(eng.cancel_events))
            eng.script = Script.ok("w")
            for dialect in DIALECTS:
                for stream in (True, False):
                    path, body = request_for(dialect, stream)
                    await asgi_call(app, "POST", path, body)
            eng.script = saved[0]
            eng.calls.clear()
            eng.cancel_events.clear()
            eng.finish_reasons.clear()
        yield app


# --------------------------------------------------------------------------- SSE + dialects


def parse_sse(text: str) -> list[tuple[str | None, str]]:
    """``[(event name or None, data string)]``; comment lines are skipped."""
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            out.append((name, "\n".join(data)))
    return out


def terminals(dialect: str, text: str) -> list[str]:
    """The terminal events of a finished stream, per dialect. A correct stream has exactly one."""
    found: list[str] = []
    for name, data in parse_sse(text):
        if data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if dialect == "openai":
            if "error" in obj and "choices" not in obj:
                found.append("error")
            elif any(c.get("finish_reason") for c in obj.get("choices", [])):
                found.append("finish")
        elif dialect == "anthropic":
            t = obj.get("type") or name
            if t in ("message_stop", "error"):
                found.append(t)
        elif dialect == "responses":
            t = obj.get("type") or name
            if t in (
                "response.completed",
                "response.incomplete",
                "response.failed",
                "error",
            ):
                found.append(t)
    return found


def assert_one_terminal(dialect: str, text: str) -> str:
    t = terminals(dialect, text)
    assert len(t) == 1, f"{dialect}: expected exactly one terminal, got {t}\n{text}"
    return t[0]


def error_obj(dialect: str, payload: dict) -> dict:
    """The error object inside a JSON error body, asserting the dialect's envelope."""
    if dialect == "anthropic":
        assert payload.get("type") == "error", payload
        err = payload["error"]
        assert {"type", "message"} <= set(err), payload
        return err
    assert "error" in payload, payload
    err = payload["error"]
    assert "message" in err and "type" in err, payload
    return err


# --------------------------------------------------------------------------- leak audit


class LeakAudit:
    """Snapshot before, check after: tracker entries, engine calls still running, threads."""

    def __init__(self, *engines: FakeBatchedEngine):
        from yunshu_engine.request_tracker import get_request_tracker

        self.tracker = get_request_tracker()
        self.engines = engines
        self.threads0 = {t.ident for t in threading.enumerate()}
        self.active0 = self.tracker.active_count

    async def settle(self, seconds: float = 6.0) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._clean():
                return
            await asyncio.sleep(0.01)

    def _clean(self) -> bool:
        return self.tracker.active_count == self.active0 and all(
            e.inflight == 0 for e in self.engines
        )

    def assert_clean(self) -> None:
        assert self.tracker.active_count == self.active0, "tracker leaked: " + str(
            [
                (
                    g.request_id,
                    g.cancel_event.is_set(),
                    g.cancelled,
                    round(g.elapsed_s, 1),
                )
                for g in self.tracker.all_active()
            ]
        )
        for e in self.engines:
            assert e.inflight == 0, f"{e.inflight} engine call(s) still running"
        extra = [
            t
            for t in threading.enumerate()
            if t.ident not in self.threads0
            and t.is_alive()
            and not t.daemon
            # The default executor an event loop starts for asyncio.to_thread
            # ("asyncio_N") belongs to that loop and is joined when it closes; a test's
            # TestClient loop outlives the assertion.
            and not t.name.startswith("asyncio_")
        ]
        assert not extra, f"leaked non-daemon threads: {extra}"


# --------------------------------------------------------------------------- fake disks


def full_disk_open(real_open=open, *, match: str = ""):
    """An ``open`` replacement whose write opens fail with ENOSPC (for paths containing
    ``match``)."""

    def _open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+") and match in str(file):
            raise OSError(errno.ENOSPC, "No space left on device", str(file))
        return real_open(file, mode, *a, **k)

    return _open


def fake_usage(total: int, free: int):
    usage = namedtuple("usage", "total used free")
    return lambda path: usage(total, total - free, free)


# --------------------------------------------------------------------------- dialects

DIALECTS: dict[str, tuple[str, Any]] = {
    "openai": (
        "/v1/chat/completions",
        lambda stream, **kw: {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            **kw,
        },
    ),
    "anthropic": (
        "/v1/messages",
        lambda stream, **kw: {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            **kw,
        },
    ),
    "responses": (
        "/v1/responses",
        lambda stream, **kw: {"model": "m", "input": "hi", "stream": stream, **kw},
    ),
}


def request_for(dialect: str, stream: bool, **kw) -> tuple[str, dict]:
    path, make = DIALECTS[dialect]
    return path, make(stream, **kw)
