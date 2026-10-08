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
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    # An OS-assigned port, never one of the shared 18990-18999 pool that live servers
    # (gpuq jobs) may already hold; a fixed port made the test talk to a foreign server.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = _free_port()


def test_graceful_shutdown_timeout_values():
    from yunshu_cli.serve import graceful_shutdown_timeout as g

    assert g(0) == 0 and g(0.4) == 1 and g(30.0) == 30 and g(-1) == 0


def _start(drain: str, delay="0.2", n="15", home: Path | None = None):
    global PORT
    PORT = _free_port()
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("YUNSHU_")},
        "YUNSHU_DRAIN_TIMEOUT": drain,
        "PYTHONPATH": str(ROOT / "python"),
    }
    if home is not None:
        env["HOME"] = str(home)
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
        if p.poll() is not None:
            raise AssertionError(
                "our scripted server exited: " + p.stdout.read().decode()[-500:]
            )
        try:
            response = httpx.get(f"http://127.0.0.1:{PORT}/health/live", timeout=1)
            if response.status_code < 500 and response.headers.get(
                "x-yunshu-scripted-pid"
            ) == str(p.pid):
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


def test_sigterm_drains_the_in_flight_stream_then_exits(tmp_path):
    p = _start("20", home=tmp_path)
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


def test_drain_zero_aborts_the_stream_fast(tmp_path):
    p = _start("0", home=tmp_path)
    try:
        got, took = _stream_then_sigterm(p)
        assert took < 8, took
        assert "w14" not in got["text"]
        p.wait(15)
    finally:
        if p.poll() is None:
            p.kill()


def test_failed_start_never_uses_an_unowned_healthy_server(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace

    monkeypatch.setattr(
        "tests.unit.test_graceful_shutdown_real_server._free_port", lambda: 23456
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(
            pid=12345, poll=lambda: 3, stdout=io.BytesIO(b"bind failed")
        ),
    )
    seen = []
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: seen.append(a) or SimpleNamespace(status_code=200)
    )
    with pytest.raises(AssertionError, match="our scripted server exited: bind failed"):
        _start("20", home=tmp_path)
    assert seen == []


def test_port_is_outside_the_shared_server_pool():
    assert not 18990 <= PORT <= 18999
