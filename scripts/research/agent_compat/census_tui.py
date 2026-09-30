"""Drive an agent's interactive TUI in a pty to capture what /status, /model, /context, ... show.

Same isolation as census.py (isolated HOME, sandbox-exec loopback-only). The model server is the
scripted census server (or an existing URL via --url, e.g. a real Yunshu), so what the TUI prints
comes from the /v1/models fields, usage numbers and headers the server returns.

    python census_tui.py claude "/status" "/context" "/model" --name cc_tui_status
    python census_tui.py codex "/status" "/model" --name cx_tui_status --url http://127.0.0.1:18990
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import termios
import time

import census
from census_server import Census

ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Z0-9]|\x1b[=>]"
)


def clean(raw: str) -> str:
    t = ANSI.sub("", raw)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", t)


def run_tui(
    agent: str,
    inputs: list[str],
    name: str,
    url: str | None,
    model: str,
    wait: float,
    cfg: str | None,
):
    run = census.BUILD / "runs" / name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    work = census.BUILD / "work" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=work)
    srv = None
    if not url:
        srv = Census(run / "requests.jsonl", [{"text": "ok"}] * 20, port=0)
        url = srv.url
    try:
        launch = census.agents.prepare(agent, run, work, url, model, "hello")
        # drop the headless bits so the TUI starts interactively
        cmd = launch.cmd
        if agent == "claude":
            i = cmd.index("-p")
            del cmd[i : i + 2]
            for flag in ("--output-format", "stream-json", "--verbose"):
                if flag in cmd:
                    cmd.remove(flag)
        elif agent == "codex":
            j = cmd.index("exec")
            end = len(cmd) - 1  # trailing prompt
            cmd[j : end + 1] = [
                "--no-daemon",
                "-C",
                str(work),
                "--dangerously-bypass-approvals-and-sandbox",
            ]
        if cfg and cfg in census.HOOKS:
            census.HOOKS[cfg](launch.home, launch, dict(run=run, work=work))
        launch.env["TERM"] = "xterm-256color"
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(launch.cwd)
            os.execvpe(cmd[0], cmd, launch.env)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 160, 0, 0))
        out = bytearray()

        def pump(seconds: float):
            end = time.time() + seconds
            while time.time() < end:
                r, _, _ = select.select([fd], [], [], 0.2)
                if r:
                    try:
                        d = os.read(fd, 65536)
                    except OSError:
                        return False
                    if not d:
                        return False
                    out.extend(d)
            return True

        pump(wait * 2)  # startup
        marks = []
        (run / "startup.txt").write_text(clean(out.decode("utf-8", "replace")))
        for text in inputs:
            marks.append((text, len(out)))
            try:
                os.write(fd, text.encode())
            except OSError:
                print(
                    "child exited early:\n"
                    + clean(out.decode("utf-8", "replace"))[-3000:]
                )
                break
            pump(0.5)
            os.write(fd, b"\r")
            if not pump(wait):
                break
        marks.append(("<end>", len(out)))
        with __import__("contextlib").suppress(OSError):
            os.write(fd, b"\x03")
            pump(0.5)
            os.write(fd, b"\x03")
            pump(0.5)
        with __import__("contextlib").suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        with __import__("contextlib").suppress(ChildProcessError):
            os.waitpid(pid, 0)
        text = out.decode("utf-8", "replace")
        chunks = []
        for (label, a), (_, b) in zip(marks, marks[1:], strict=False):
            chunks.append(f"===== {label} =====\n{clean(text[a:b])}")
        (run / "tui.txt").write_text(
            f"===== startup =====\n{clean(text[: marks[0][1]])}\n" + "\n".join(chunks)
        )
        print((run / "tui.txt").read_text()[:6000])
    finally:
        if srv:
            srv.stop()
        dst = census.OUT_ROOT / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(run, dst, ignore=shutil.ignore_patterns("home", "sandbox.sb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=["claude", "codex"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default="census-model")
    ap.add_argument("--wait", type=float, default=4.0)
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    run_tui(a.agent, a.inputs, a.name, a.url, a.model, a.wait, a.cfg)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
