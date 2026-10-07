"""Start / stop the model server for one benchmark job (Yunshu default settings, or TensorFold).

The server gets an isolated HOME (no user config, no user caches), runs in its own session so it
can be killed with SIGKILL, and is always killed on exit. Ports come from 18990-18999 only.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

# 18990-18999 unless AGENTIC_PORT_LO / AGENTIC_PORT_HI narrow it (a worker limited to a few ports)
PORT_RANGE = range(
    int(os.environ.get("AGENTIC_PORT_LO", "18990")),
    int(os.environ.get("AGENTIC_PORT_HI", "18999")) + 1,
)
YUNSHU_MAIN = Path(
    os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
)
YUNSHU_BIN = YUNSHU_MAIN / ".venv" / "bin" / "yunshu"
TF_BIN = os.environ.get(
    "AGENTIC_TF_BIN", "/Volumes/P5Plus/yunshu-test-envs/tensorfold/bin/tensorfold"
)
TF_DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
SERVER_HOME = Path("/Volumes/P5Plus/yunshu-build/agentic/server-home")


def port_bindable(port: int) -> bool:
    """Whether a server can listen on 127.0.0.1:port (SO_REUSEADDR as uvicorn sets it)."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_ports(
    n: int,
    skip=frozenset(),
    wait_s: float = 600.0,
    interval_s: float = 5.0,
    sleep=time.sleep,
    clock=time.monotonic,
) -> list[int]:
    """n free ports of the pool. The pool is shared with other gpuq jobs, and paused (preempted)
    jobs keep their ports bound, so momentary exhaustion is normal: re-scan every interval_s for
    up to wait_s before giving up. `skip` holds ports that already lost a bind race."""
    t0 = clock()
    while True:
        out = [p for p in PORT_RANGE if p not in skip and port_bindable(p)][:n]
        if len(out) == n:
            return out
        left = wait_s - (clock() - t0)
        if left <= 0:
            lo, hi = PORT_RANGE[0], PORT_RANGE[-1]
            raise RuntimeError(f"no free port in {lo}-{hi} after {wait_s:.0f}s")
        print(
            f"[agentic] only {len(out)}/{n} ports free in the pool; waiting "
            f"(up to {left:.0f}s more)",
            flush=True,
        )
        sleep(min(interval_s, left))


def start_server(
    n_ports: int, make_server, attempts: int = 3, **port_kw
) -> tuple[Server, list[int]]:
    """Pick n_ports free ports, build make_server(ports[0]) and start it. When it dies with
    "address already in use" (another process took the port between probe and bind), choose new
    ports and retry, up to `attempts` times. Returns (server, ports)."""
    lost: set[int] = set()
    for attempt in range(1, attempts + 1):
        ports = free_ports(n_ports, skip=frozenset(lost), **port_kw)
        server = make_server(ports[0])
        try:
            return server.start(), ports
        except RuntimeError:
            if attempt == attempts or "address already in use" not in server.log_tail():
                raise
            lost.add(ports[0])
            print(
                f"[agentic] port {ports[0]} lost a bind race; retrying "
                f"({attempt}/{attempts})",
                flush=True,
            )
    raise AssertionError("unreachable")


class Server:
    def __init__(self, kind: str, model_path: str, port: int, log: Path, extra=()):
        self.kind, self.model_path, self.port, self.log = kind, model_path, port, log
        self.extra = list(extra)
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{port}"

    def command(self) -> list[str]:
        if self.kind == "yunshu":
            return [
                str(YUNSHU_BIN),
                "serve",
                "-m",
                self.model_path,
                "--port",
                str(self.port),
                *self.extra,
            ]
        if self.kind == "tensorfold":
            return [
                TF_BIN,
                "serve",
                self.model_path,
                "--port",
                str(self.port),
                "--drafter",
                TF_DRAFTER,
                "--snapshot-dir",
                str(SERVER_HOME / "tf-snapshots"),
                "--no-update-check",
                *self.extra,
            ]
        raise ValueError(self.kind)

    def start(self, ready_timeout: float = 600.0):
        SERVER_HOME.mkdir(parents=True, exist_ok=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE", "CODEX"))
        }
        env.update(HOME=str(SERVER_HOME), HF_HUB_OFFLINE="1", NO_PROXY="127.0.0.1")
        src = os.environ.get(
            "AGENTIC_YUNSHU_SRC"
        )  # a python/ dir to run instead of the checkout
        if src and self.kind == "yunshu":
            env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        self.link_models(env)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.log.open("ab") as log_file:
                self.proc = subprocess.Popen(
                    self.command(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            t0 = time.monotonic()
            while time.monotonic() - t0 < ready_timeout:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited early rc={self.proc.returncode}")
                try:
                    with urllib.request.urlopen(
                        self.url + "/v1/models", timeout=3
                    ) as r:
                        if r.status == 200:
                            self.models = json.load(r)
                            self.ready_s = time.monotonic() - t0
                            return self
                except Exception:
                    time.sleep(2)
            raise RuntimeError("server not ready in time")
        except BaseException:
            # start() runs before callers enter their cleanup block. A timeout or
            # interrupted startup must not leave our model process on the GPU.
            self.kill()
            raise

    def link_models(self, env: dict) -> None:
        """AGENTIC_YUNSHU_DRAFTER=<dir>: put the drafter where a user who ran ``yunshu pull`` has it
        (``$HOME/.yunshu/models/<org>/<name>``) so Yunshu's own discovery finds it. The isolated HOME
        otherwise hides it and the server silently runs MTP."""
        drafter = os.environ.get("AGENTIC_YUNSHU_DRAFTER")
        if not drafter or self.kind != "yunshu":
            return
        link = Path(env["HOME"]) / ".yunshu" / "models" / "incoai" / Path(drafter).name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(drafter)

    def engaged_spec_mode(self) -> str | None:
        """The draft mode the Yunshu runner reports in its log (None: no evidence)."""
        import re

        try:
            text = self.log.read_text(errors="replace")
        except OSError:
            return None
        modes = re.findall(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b", text)
        return modes[-1] if modes else None
    def log_tail(self, n: int = 30) -> str:
        try:
            return "".join(self.log.read_text(errors="replace").splitlines(True)[-n:])
        except OSError:
            return ""

    @property
    def model_id(self) -> str:
        data = self.models.get("data") or []
        return data[0]["id"] if data else Path(self.model_path).name

    def kill(self):
        if self.proc and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=30)
