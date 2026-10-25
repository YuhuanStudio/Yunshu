"""Real check that the console shows engine facts as sampled: serve --observe, two concurrent streams,
the worktree console on 18996, the debug section screenshot every 2 s.

Usage (through gpuq):  uihonesty_run.py --model PATH --port 18991 --home DIR --out DIR [--seconds 90]

Exits nonzero when the screenshot run reports a red mismatch, a displayed count ahead of the engine, or
too few samples with both requests live. Every process it starts is killed on exit.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

YUNSHU = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
CONSOLE_PORTS = (18996, 18998, 18997, 18993, 18992)


def pick_console_port(engine_port: int) -> int | None:
    """The first free port of the console list (18996 first); None when all are taken."""
    return next((p for p in CONSOLE_PORTS if p != engine_port and port_free(p)), None)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def stream_forever(base: str, prompt: str, stop: threading.Event) -> None:
    body = (
        '{"model":"m","stream":true,"max_tokens":900,"messages":[{"role":"user","content":%s}]}'
        % __import__("json").dumps(prompt)
    )
    while not stop.is_set():
        try:
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                body.encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                while r.read(4096):
                    if stop.is_set():
                        return
        except Exception:
            time.sleep(1)


def killgroup(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=int, default=90)
    a = ap.parse_args()
    if not port_free(a.port):
        print(f"port {a.port} is busy", file=sys.stderr)
        return 2
    console_port = pick_console_port(a.port)
    if console_port is None:
        print("no free console port", file=sys.stderr)
        return 2
    print(f"console port {console_port}", flush=True)
    root = Path(__file__).resolve().parents[2]
    home = Path(a.home)
    home.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        HOME=str(home),
        HF_HUB_OFFLINE="1",
        NO_PROXY="127.0.0.1",
        PYTHONPATH=str(root / "python"),
    )
    for k in [k for k in env if k.startswith("YUNSHU_")]:
        del env[k]
    base = f"http://127.0.0.1:{a.port}"
    server = vite = None
    stop = threading.Event()
    try:
        server = subprocess.Popen(
            [YUNSHU, "serve", "--observe", "--no-console", "-m", a.model]
            + ["--port", str(a.port)],
            env=env,
            stdout=open(home / "server.log", "wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = time.time()
        while True:
            if server.poll() is not None:
                print("server exited", server.returncode, file=sys.stderr)
                return 3
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=3).read()
                break
            except Exception:
                time.sleep(1)
            if time.time() - t0 > 300:
                print("server not ready", file=sys.stderr)
                return 3
        vite = subprocess.Popen(
            ["pnpm", "exec", "vite", "--host", "127.0.0.1", "--port", str(console_port)]
            + ["--strictPort"],
            cwd=root / "frontend",
            env=dict(env, VITE_CONSOLE_PROXY=base),
            stdout=open(home / "vite.log", "wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(60):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{console_port}/console/", timeout=2
                ).read()
                break
            except Exception:
                time.sleep(1)
        else:
            print("console not ready", file=sys.stderr)
            return 4
        for prompt in (
            "Write a long story about a lighthouse keeper, in many paragraphs.",
            "Explain how a transformer language model works, in great detail.",
        ):
            threading.Thread(
                target=stream_forever, args=(base, prompt, stop), daemon=True
            ).start()
        shots = subprocess.run(
            [
                "node",
                "scripts/uihonesty-shots.mjs",
                f"http://127.0.0.1:{console_port}",
                base,
                str(out),
                str(a.seconds),
            ],
            cwd=root / "frontend",
            env=dict(
                env,
                PLAYWRIGHT_BROWSERS_PATH=os.path.expanduser(
                    "~/Library/Caches/ms-playwright"
                ),
            ),
        )
        return shots.returncode
    finally:
        stop.set()
        killgroup(vite)
        killgroup(server)


if __name__ == "__main__":
    sys.exit(main())
