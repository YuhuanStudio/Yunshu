"""Run one command with wall-clock, CPU, memory and output limits (srt gives none of these).

The command runs in its own session; on timeout, memory overrun or output overflow the whole process
tree is killed, and the session is always swept afterwards so a command cannot leave daemons behind.
Memory is the resident size of the whole tree, sampled every ``poll`` seconds: macOS does not enforce
``RLIMIT_AS``, and ``RLIMIT_NPROC`` counts every process of the user, so neither is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import resource
import signal
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class RunResult:
    stdout: bytes = b""
    stderr: bytes = b""
    return_code: int = 0
    timed_out: bool = False
    oom: bool = False  # killed for exceeding the memory limit
    output_overflow: bool = False  # killed for writing more than max_total bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    elapsed: float = 0.0
    peak_rss: int = 0
    notes: list[str] = field(default_factory=list)


def _limits(cpu_s: int, fsize: int, nofile: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        n = min(nofile, hard) if hard != resource.RLIM_INFINITY else nofile
        resource.setrlimit(resource.RLIMIT_NOFILE, (n, hard))

    return apply


def _tree(pid: int) -> list[psutil.Process]:
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return []


def _tree_rss(pid: int) -> int:
    total = 0
    for p in _tree(pid):
        with contextlib.suppress(psutil.Error):
            total += p.memory_info().rss
    return total


def kill_tree(pid: int) -> None:
    procs = _tree(pid)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)  # pid is the session / group leader
    for p in procs:
        with contextlib.suppress(psutil.Error):
            p.kill()


async def run_supervised(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: str,
    timeout: float,
    mem_bytes: int,
    per_stream: int = 64 * 1024,
    max_total: int = 1024 * 1024,
    max_file: int = 256 * 1024 * 1024,
    poll: float = 0.1,
) -> RunResult:
    res = RunResult()
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        env=env,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        preexec_fn=_limits(max(1, int(timeout)), max_file, 1024),  # noqa: PLW1509
    )
    pid = proc.pid
    got = {"total": 0}
    bufs = {"out": bytearray(), "err": bytearray()}
    killed = asyncio.Event()

    async def pump(stream, key: str) -> None:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            got["total"] += len(chunk)
            b = bufs[key]
            room = per_stream - len(b)
            if room > 0:
                b.extend(chunk[:room])
            if len(chunk) > max(room, 0):
                if key == "out":
                    res.stdout_truncated = True
                else:
                    res.stderr_truncated = True
            if got["total"] > max_total and not killed.is_set():
                res.output_overflow = True
                killed.set()
                kill_tree(pid)

    async def watch() -> None:
        while not killed.is_set():
            await asyncio.sleep(poll)
            rss = _tree_rss(pid)
            res.peak_rss = max(res.peak_rss, rss)
            if rss > mem_bytes:
                res.oom = True
                killed.set()
                kill_tree(pid)
                return

    tasks = [
        asyncio.create_task(pump(proc.stdout, "out")),
        asyncio.create_task(pump(proc.stderr, "err")),
    ]
    watcher = asyncio.create_task(watch())
    try:
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except TimeoutError:
            res.timed_out = True
            killed.set()
            kill_tree(pid)
            await proc.wait()
        # whatever the command left running (daemons, background jobs) dies with the call, and
        # only then the pipes are drained (a survivor would otherwise hold them open)
        kill_tree(pid)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(*tasks), 2.0)
    finally:
        killed.set()
        kill_tree(pid)  # sweep the session: nothing outlives the call
        watcher.cancel()
        for t in tasks:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.gather(watcher, *tasks, return_exceptions=True)
    res.stdout, res.stderr = bytes(bufs["out"]), bytes(bufs["err"])
    rc = proc.returncode if proc.returncode is not None else -9
    res.return_code = rc if rc >= 0 else 128 + (-rc)
    res.elapsed = time.monotonic() - t0
    return res
