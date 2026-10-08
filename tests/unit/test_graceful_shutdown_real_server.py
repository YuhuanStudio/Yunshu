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


def _pick_port():
    for port in range(18990, 19000):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free shutdown-test port in 18990..18999")


def test_graceful_shutdown_timeout_values():
    from yunshu_cli.serve import graceful_shutdown_timeout as g

    assert g(0) == 0 and g(0.4) == 1 and g(30.0) == 30 and g(-1) == 0


def _start(drain: str, delay="0.2", n="15"):
    port = _pick_port()
    env = {
        **os.environ,
        "YUNSHU_DRAIN_TIMEOUT": drain,
        "PYTHONPATH": str(ROOT / "python"),
        "YUNSHU_AUTH_DISABLED": "1",
    }
    env.pop("YUNSHU_AUTH_TOKEN", None)
    p = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/research/scripted_server.py"),
            str(port),
            delay,
            n,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if p.poll() is not None:
            break
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/_scripted_owner", timeout=1)
            if response.status_code == 200 and response.json() == {"pid": p.pid}:
                return p, port
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.2)
    if p.poll() is None:
        p.kill()
    p.wait()
    raise AssertionError("server did not start: " + p.stdout.read().decode()[-500:])


def _stream_then_sigterm(p, port):
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
    p, port = _start("20")
    try:
        got, _ = _stream_then_sigterm(p, port)
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
    p, port = _start("0")
    try:
        got, took = _stream_then_sigterm(p, port)
        assert took < 8, took
        assert "w14" not in got["text"]
        p.wait(15)
    finally:
        if p.poll() is None:
            p.kill()


def test_start_rejects_a_foreign_listener(monkeypatch):
    import io

    import pytest

    class Child:
        pid = 123
        stdout = io.BytesIO(b"bind failed")
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 3

        def wait(self):
            return 3

    class Foreign:
        status_code = 200

        def json(self):
            return {"pid": 999}

    monkeypatch.setattr(sys.modules[__name__], "_pick_port", lambda: 18990)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Foreign())
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(AssertionError, match="bind failed"):
        _start("0")
