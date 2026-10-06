"""One gpuq M3 job: serve a model on loopback until told to stop (or a deadline), then write a receipt.

    gpuq submit --device m3 --mem-gb 9 --out OUT.jsonl --expect-complete --timeout 40 -- \
        python scripts/research/agent_compat/m3_serve.py --model /Volumes/P5Plus/models/Qwen3.5-9B-MLX-4bit \
        --port 18994 --control-port 18995 --minutes 35 --out OUT.jsonl

Both ports bind 127.0.0.1 only; the M5 reaches them through `ssh -L`. The client stops the job by sending
the line `stop` to the control port (the stop file of the plan, over loopback so nothing is written on the laptop
beyond the job's own checkout). The wrapper prints `READY <port>` once /v1/models answers, `HEARTBEAT` lines
while it waits, kills the server on every exit path, and writes one JSON line containing `complete` to --out
only when the server was healthy until a client asked it to stop (a crash or the deadline is `complete: false`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def wait_stop(control: socket.socket, deadline: float, alive, beat=lambda: None, every: float = 30.0) -> str:
    """Block until a client sends `stop` (-> "stop"), the deadline passes (-> "deadline") or alive() is False
    (-> "server-died")."""
    control.settimeout(1.0)
    last = time.monotonic()
    while True:
        if not alive():
            return "server-died"
        if time.monotonic() >= deadline:
            return "deadline"
        try:
            conn, _ = control.accept()
        except TimeoutError:
            if time.monotonic() - last >= every:
                beat()
                last = time.monotonic()
            continue
        with conn:
            conn.settimeout(2.0)
            try:
                line = conn.recv(64).decode("utf-8", "replace").strip()
            except OSError:
                line = ""
            if line == "stop":
                with contextlib.suppress(OSError):
                    conn.sendall(b"stopping\n")
                return "stop"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--control-port", type=int, required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--ready-timeout", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args(argv)
    for p in (a.port, a.control_port):
        if not 18990 <= p <= 18999:
            print(f"port {p} outside 18990-18999", file=sys.stderr)
            return 2
    t_start = time.monotonic()
    control = socket.socket()
    control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control.bind(("127.0.0.1", a.control_port))
    control.listen(4)
    cmd = [sys.executable, "-m", "yunshu_cli", "serve", "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port), *a.extra]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "NO_PROXY": "127.0.0.1"}
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    reason, ready_s, model_id = "start-failed", None, None
    try:
        while time.monotonic() - t_start < a.ready_timeout:
            if proc.poll() is not None:
                reason = "server-exited-early"
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=3) as r:
                    model_id = json.load(r)["data"][0]["id"]
                    ready_s = round(time.monotonic() - t_start, 1)
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        if ready_s is not None:
            print(f"READY {a.port} {model_id} {ready_s}s", flush=True)
            reason = wait_stop(
                control,
                time.monotonic() + a.minutes * 60,
                lambda: proc.poll() is None,
                beat=lambda: print(f"HEARTBEAT {time.monotonic() - t_start:.0f}s", flush=True),
            )
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        control.close()
    row = dict(
        complete=reason == "stop",
        success=reason == "stop",
        reason=reason,
        model=Path(a.model).name,
        ready_s=ready_s,
        device=os.environ.get("GPUQ_DEVICE", "m3"),
        evidence="portability evidence (not M5)",
    )
    Path(a.out).write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return 0 if reason == "stop" else 1


if __name__ == "__main__":
    sys.exit(main())
