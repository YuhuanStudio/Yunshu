"""A persistent Python process per container (a REPL), run under srt with the same limits as a bash call.

State (variables, imports, open files) lives in the process between calls. A call is bounded in wall
time (``YUNSHU_CODE_EXECUTION_TIMEOUT``, counted only while code runs, not while it waits for the
client's answer to a tool call) and in memory (resident size of the whole tree, sampled every 100
ms); when a limit is hit the whole process group is killed and the next call starts a clean process
(the state is gone, which the result says). A cell that writes more than 1 MiB kills the process too.

A cell may await tools the request declared (programmatic tool calling): the call is returned to the
caller as ``ReplOutcome(status="tool_calls")``, the process waits, and :meth:`ReplSession.resume`
hands it the results. A session nobody resumes within ``PAUSE_TTL`` seconds is killed.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .supervisor import _limits, _tree_rss, kill_tree

logger = logging.getLogger("yunshu.sandbox")

PAUSE_TTL = 270.0  # seconds a paused cell waits for the client's tool results
IDLE_TTL = 600.0  # seconds an unused REPL process may stay alive
MAX_SESSIONS = (
    4  # live REPL processes (each holds memory); the least recently used is closed
)
START_TIMEOUT = 30.0
GRACE = 0.05  # collect tool calls issued in the same event-loop turn (asyncio.gather)
WORKER_FILE = "repl_worker.py"


class ReplError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # "no_pending" | "busy" | "unavailable"
        self.message = message


@dataclass
class ReplOutcome:
    status: str  # done | tool_calls | timeout | oom | overflow | crashed
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    calls: list[dict] = field(default_factory=list)  # {"call_id", "name", "input"}
    state_lost: bool = False  # the process was killed or replaced: variables are gone
    elapsed: float = 0.0


def worker_source() -> str:
    return (Path(__file__).parent / WORKER_FILE).read_text()


class ReplSession:
    """One container's Python process. All methods run on the gateway's event loop."""

    def __init__(self, executor, container):
        self.ex = executor
        self.c = container
        self.proc: asyncio.subprocess.Process | None = None
        self.queue: asyncio.Queue[dict] | None = None
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._pause_timer: asyncio.Task | None = None
        self.pending: dict[str, dict] = {}  # call_id -> call, while paused
        self._budget = 0.0  # wall-clock seconds this cell may still run
        self._mem = 0
        self._lost = False  # a previous process died
        self.last_used = time.monotonic()
        self.stderr_tail = b""
        self.meta: dict = {}  # the caller's bookkeeping for a paused cell (e.g. the server tool id)

    # ── process ──────────────────────────────────────────────────────────────
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def paused(self) -> bool:
        return bool(self.pending)

    async def _spawn(self) -> None:
        c = self.c
        await asyncio.to_thread(
            (c.path / WORKER_FILE).write_text, worker_source()
        )  # rewritten at every start: user code may have edited it
        out_path, err_path = c.tmp / ".repl.out", c.tmp / ".repl.err"
        argv = self.ex.srt_argv(
            c,
            [
                "/usr/bin/env",
                f"TMPDIR={c.tmp}/",
                f"TMPPREFIX={c.tmp / 'zsh'}",
                "python3",
                "-u",
                str(c.path / WORKER_FILE),
                str(out_path),
                str(err_path),
            ],
        )
        out_dir = c.out / "repl"
        out_dir.mkdir(mode=0o700, exist_ok=True)
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            env=self.ex.env(c, out_dir),
            cwd=str(c.work),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=32 * 1024 * 1024,
            start_new_session=True,
            preexec_fn=_limits(3600, 256 * 1024 * 1024, 1024),  # noqa: PLW1509
        )
        self.queue = asyncio.Queue()
        self.stderr_tail = b""
        self._tasks = [
            asyncio.create_task(self._pump_stdout(self.proc, self.queue)),
            asyncio.create_task(self._pump_stderr(self.proc)),
        ]
        try:
            msg = await asyncio.wait_for(self.queue.get(), START_TIMEOUT)
        except TimeoutError:
            self.kill()
            raise ReplError("unavailable", "the Python process did not start") from None
        if msg.get("op") != "ready":
            tail = self.stderr_tail.decode("utf-8", "replace")[-300:]
            self.kill()
            raise ReplError(
                "unavailable", f"the Python process failed to start: {tail}"
            )

    async def _pump_stdout(self, proc, q: asyncio.Queue) -> None:
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                with contextlib.suppress(ValueError):
                    m = json.loads(line)
                    if isinstance(m, dict):
                        q.put_nowait(m)
        except (asyncio.CancelledError, ValueError, asyncio.LimitOverrunError):
            pass
        q.put_nowait({"op": "eof"})

    async def _pump_stderr(self, proc) -> None:
        try:
            while True:
                b = await proc.stderr.read(4096)
                if not b:
                    return
                self.stderr_tail = (self.stderr_tail + b)[-4096:]
        except asyncio.CancelledError:
            pass

    def kill(self) -> None:
        """Kill the process group now (sync: also called from a container delete)."""
        if self._pause_timer:
            self._pause_timer.cancel()
            self._pause_timer = None
        proc, self.proc = self.proc, None
        if proc is not None:
            if proc.returncode is None:
                kill_tree(proc.pid)
            self._lost = True
            with contextlib.suppress(RuntimeError):  # no loop at interpreter exit
                self._reap = asyncio.get_running_loop().create_task(proc.wait())
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        self.pending = {}

    async def close(self) -> None:
        self.kill()
        reap = getattr(self, "_reap", None)
        if reap is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(reap, 5)

    # ── cells ────────────────────────────────────────────────────────────────
    def _send(self, obj: dict) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())

    async def run(
        self,
        code: str,
        tools: list[dict] | None,
        *,
        timeout: float,
        mem_bytes: int,
        meta: dict | None = None,
    ) -> ReplOutcome:
        """Run a cell. Cancels a cell that is still waiting for tool results (its state is lost)."""
        self.last_used = time.monotonic()
        if self.paused:
            self.kill()
        async with self._lock:
            lost = False
            if not self.alive:
                lost = self._lost or self.proc is not None
                self._lost = False
                await self._spawn()
            self._budget = float(timeout)
            self._mem = mem_bytes
            self.meta = dict(meta or {})
            self._send(
                {
                    "op": "exec",
                    "id": os.urandom(6).hex(),
                    "code": code,
                    "tools": tools or [],
                }
            )
            out = await self._drive()
            out.state_lost = out.state_lost or lost
            return out

    async def resume(self, results: dict[str, tuple[object, bool]]) -> ReplOutcome:
        """Hand the client's tool results to a paused cell; ``{call_id: (content, is_error)}``."""
        if not self.paused or set(results) != set(self.pending):
            raise ReplError(
                "no_pending",
                "These tool results do not match the code execution that is waiting.",
            )
        if self._pause_timer:
            self._pause_timer.cancel()
            self._pause_timer = None
        self.last_used = time.monotonic()
        async with self._lock:
            ids = list(self.pending)
            self.pending = {}
            for cid in ids:
                content, is_err = results[cid]
                self._send(
                    {
                        "op": "tool_result",
                        "call_id": cid,
                        "content": content,
                        "is_error": is_err,
                    }
                )
            return await self._drive()

    async def _watch_memory(self, pid: int, q: asyncio.Queue) -> None:
        while True:
            await asyncio.sleep(0.1)
            if _tree_rss(pid) > self._mem:
                kill_tree(pid)
                q.put_nowait({"op": "oom"})
                return

    async def _drive(self) -> ReplOutcome:
        assert self.proc is not None and self.queue is not None
        q, pid = self.queue, self.proc.pid
        t0 = time.monotonic()
        calls: list[dict] = []
        watcher = asyncio.create_task(self._watch_memory(pid, q))

        def spent() -> float:
            return time.monotonic() - t0

        def fail(status: str) -> ReplOutcome:
            self.kill()
            return ReplOutcome(status, elapsed=spent(), state_lost=True)

        try:
            while True:
                left = self._budget - spent()
                try:
                    msg = await asyncio.wait_for(q.get(), max(left, 0.0))
                except TimeoutError:
                    return fail("timeout")
                op = msg.get("op")
                if op == "tool_call":
                    calls.append(msg)
                    deadline = time.monotonic() + GRACE
                    while True:  # tool calls issued together (asyncio.gather)
                        wait = deadline - time.monotonic()
                        if wait <= 0:
                            break
                        try:
                            m2 = await asyncio.wait_for(q.get(), wait)
                        except TimeoutError:
                            break
                        if m2.get("op") == "tool_call":
                            calls.append(m2)
                        else:
                            q.put_nowait(m2)
                            break
                    self._budget = max(0.0, self._budget - spent())
                    self.pending = {
                        str(m["call_id"]): {
                            "call_id": str(m["call_id"]),
                            "name": str(m.get("name") or ""),
                            "input": m.get("input")
                            if isinstance(m.get("input"), dict)
                            else {},
                        }
                        for m in calls
                    }
                    self._pause_timer = asyncio.create_task(self._expire_pause())
                    return ReplOutcome(
                        "tool_calls", calls=list(self.pending.values()), elapsed=spent()
                    )
                if op == "done":
                    return ReplOutcome(
                        "done",
                        stdout=str(msg.get("stdout") or ""),
                        stderr=str(msg.get("stderr") or ""),
                        return_code=int(msg.get("return_code") or 0),
                        elapsed=spent(),
                    )
                if op == "oom":
                    return fail("oom")
                if op == "overflow":
                    return fail("overflow")
                if op == "eof":
                    tail = self.stderr_tail.decode("utf-8", "replace")
                    out = fail("crashed")
                    out.stderr = tail[-2000:]
                    return out
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watcher

    async def _expire_pause(self) -> None:
        try:
            await asyncio.sleep(PAUSE_TTL)
        except asyncio.CancelledError:
            return
        if self.paused:
            logger.info("code_exec container=%s paused cell expired", self.c.id)
            self.kill()


class ReplPool:
    def __init__(self) -> None:
        self.sessions: dict[str, ReplSession] = {}

    def get(self, executor, container) -> ReplSession:
        now = time.monotonic()
        for cid, s in list(self.sessions.items()):
            if cid != container.id and not s.paused and now - s.last_used > IDLE_TTL:
                self.drop(cid)
        s = self.sessions.get(container.id)
        if s is None:
            while len(self.sessions) >= MAX_SESSIONS:
                idle = sorted(
                    self.sessions.items(),
                    key=lambda kv: (kv[1].paused, kv[1].last_used),
                )
                self.drop(idle[0][0])
            s = self.sessions[container.id] = ReplSession(executor, container)
        else:
            s.c = container
        s.last_used = now
        return s

    def find_pending(self, tool_use_ids: set[str]) -> ReplSession | None:
        """The paused cell whose programmatic tool calls carry these tool_use ids."""
        for s in self.sessions.values():
            calls = (s.meta.get("ptc") or {}).get("calls") or {}
            if s.paused and calls and set(calls) <= set(tool_use_ids):
                return s
        return None

    def peek(self, cid: str) -> ReplSession | None:
        return self.sessions.get(cid)

    def drop(self, cid: str) -> None:
        s = self.sessions.pop(cid, None)
        if s is not None:
            s.kill()

    def drop_all(self) -> None:
        for cid in list(self.sessions):
            self.drop(cid)


_pool = ReplPool()


def get_pool() -> ReplPool:
    return _pool


def _install_hooks() -> None:
    from .containers import on_delete

    on_delete(_pool.drop)
    atexit.register(_pool.drop_all)


_install_hooks()
