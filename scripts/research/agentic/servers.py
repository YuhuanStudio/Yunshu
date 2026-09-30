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

PORT_RANGE = range(18990, 19000)
YUNSHU_MAIN = Path(
    os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
)
YUNSHU_BIN = YUNSHU_MAIN / ".venv" / "bin" / "yunshu"
TF_BIN = "/Volumes/P5Plus/yunshu-test-envs/tensorfold/bin/tensorfold"
TF_DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
SERVER_HOME = Path("/Volumes/P5Plus/yunshu-build/agentic/server-home")


def free_ports(n: int) -> list[int]:
    out = []
    for p in PORT_RANGE:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        out.append(p)
        if len(out) == n:
            return out
    raise RuntimeError("no free port in 18990-18999")


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
        self.log.parent.mkdir(parents=True, exist_ok=True)
        f = self.log.open("ab")
        self.proc = subprocess.Popen(
            self.command(),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early rc={self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=3) as r:
                    if r.status == 200:
                        self.models = json.load(r)
                        self.ready_s = time.time() - t0
                        return self
            except Exception:
                time.sleep(2)
        raise RuntimeError("server not ready in time")

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
