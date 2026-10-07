"""Graceful shutdown with requests in flight, on a real uvicorn process (vLLM launcher.py shutdown
modes: drain N seconds, or abort at 0). The engine is the scripted one, so this is CPU-only."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PORTS = range(
    18990, 18999
)  # the range this repo's tests may use; other workers' servers share it
PORT = 18991  # rebound per test by _start: a fixed port can be held by someone else's server


def test_graceful_shutdown_timeout_values():
    from yunshu_cli.serve import graceful_shutdown_timeout as g

    assert g(0) == 0 and g(0.4) == 1 and g(30.0) == 30 and g(-1) == 0


def _free_port() -> int:
    """A port of the shared range nobody listens on. A fixed port flaked in full runs: another
    worker's server already on it answered the readiness probe and the test then streamed from
    (and SIGTERMed nothing of) the wrong process."""
    for port in PORTS:
        with (
            socket.socket() as probe
        ):  # a live listener (SO_REUSEADDR may still let bind pass)
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                continue
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise AssertionError(f"no free port in {PORTS}")


def _start(drain: str, delay="0.2", n="15"):
    global PORT
    PORT = _free_port()
    env = {
        **os.environ,
        "YUNSHU_DRAIN_TIMEOUT": drain,
        "PYTHONPATH": str(ROOT / "python"),
    }
    env.pop("YUNSHU_AUTH_TOKEN", None)
    p = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/research/scripted_server.py"),
            str(PORT),
            delay,
            n,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if (
            p.poll() is not None
        ):  # our server died (port taken, import error): never probe on
            raise AssertionError("server exited: " + p.stdout.read().decode()[-500:])
        try:
            if (
                httpx.get(f"http://127.0.0.1:{PORT}/health/live", timeout=1).status_code
                < 500
            ):
                return p
        except httpx.HTTPError:
            time.sleep(0.2)
    p.kill()
    raise AssertionError("server did not start: " + p.stdout.read().decode()[-500:])


def _stream_then_sigterm(p):
    got = {"text": "", "err": None}

    def run():
        try:
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                timeout=30,
            ) as r:
                for line in r.iter_lines():
                    got["text"] += line + "\n"
        except Exception as e:  # noqa: BLE001
            got["err"] = e

    t = threading.Thread(target=run)
    t.start()
    for _ in range(100):  # wait for the first tokens; many more are still to come
        if "w0" in got["text"] or got["err"]:
            break
        time.sleep(0.1)
    assert "w0" in got["text"], (got, p.poll())
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    t.join(30)
    return got, time.time() - t0


def test_sigterm_drains_the_in_flight_stream_then_exits():
    p = _start("20")
    try:
        got, _ = _stream_then_sigterm(p)
        assert got["err"] is None, got["err"]
        assert "[DONE]" in got["text"] and '"finish_reason": "stop"' in got["text"]
        assert "w14" in got["text"]
        assert p.wait(30) in (
            0,
            -signal.SIGTERM,
        )  # uvicorn re-raises the captured signal
    finally:
        if p.poll() is None:
            p.kill()


def test_drain_zero_aborts_the_stream_fast():
    p = _start("0")
    try:
        got, took = _stream_then_sigterm(p)
        assert took < 8, took
        assert "w14" not in got["text"]
        p.wait(15)
    finally:
        if p.poll() is None:
            p.kill()


def test_free_port_skips_a_port_someone_else_listens_on(monkeypatch):
    with socket.socket() as holder, socket.socket() as spare:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        spare.bind(("127.0.0.1", 0))
        taken, free = holder.getsockname()[1], spare.getsockname()[1]
        spare.close()
        monkeypatch.setattr(sys.modules[__name__], "PORTS", [taken, free])
        assert _free_port() == free
