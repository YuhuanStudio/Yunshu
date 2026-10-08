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


def _reserve_port(wait_s=60):
    """Own the listener before launching, so other suites/paused jobs cannot steal it."""
    deadline = time.monotonic() + wait_s
    while True:
        for port in range(18990, 19000):
            listener = socket.socket()
            try:
                listener.bind(("127.0.0.1", port))
                listener.listen(128)
            except OSError:
                listener.close()
                continue
            return listener
        if time.monotonic() >= deadline:
            raise AssertionError("no free scripted-server port before deadline")
        time.sleep(0.1)


def test_graceful_shutdown_timeout_values():
    from yunshu_cli.serve import graceful_shutdown_timeout as g

    assert g(0) == 0 and g(0.4) == 1 and g(30.0) == 30 and g(-1) == 0


def _start(drain: str, delay="0.2", n="15"):
    env = {
        **os.environ,
        "YUNSHU_DRAIN_TIMEOUT": drain,
        "PYTHONPATH": str(ROOT / "python"),
    }
    env.pop("YUNSHU_AUTH_TOKEN", None)
    listener = _reserve_port()
    port = listener.getsockname()[1]
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/research/scripted_server.py"),
                str(port),
                delay,
                n,
                str(listener.fileno()),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(listener.fileno(),),
        )
    finally:
        listener.close()
    p._yunshu_test_port = port
    for _ in range(100):
        if p.poll() is not None:
            raise AssertionError(
                "scripted server exited: " + p.stdout.read().decode()[-500:]
            )
        try:
            if (
                httpx.get(f"http://127.0.0.1:{port}/health/live", timeout=1).status_code
                < 500
            ):
                return p
        except httpx.HTTPError:
            time.sleep(0.2)
    p.kill()
    raise AssertionError("server did not start: " + p.stdout.read().decode()[-500:])


def _stream_then_sigterm(p):
    port = p._yunshu_test_port
    got = {"text": "", "err": None}

    def run():
        try:
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
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


def test_reserved_scripted_listener_cannot_be_stolen():
    listener = _reserve_port()
    try:
        port = listener.getsockname()[1]
        assert 18990 <= port <= 18999
        with socket.socket() as competing:
            import pytest

            with pytest.raises(OSError):
                competing.bind(("127.0.0.1", port))
    finally:
        listener.close()
