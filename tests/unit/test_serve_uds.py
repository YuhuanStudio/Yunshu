"""`yunshu serve --uds PATH` hands uvicorn a Unix socket instead of host/port."""

from __future__ import annotations

import socket
import tempfile

from typer.testing import CliRunner

from yunshu_cli import app
from yunshu_engine import settings


def test_serve_uds_passes_uds_to_uvicorn(monkeypatch):
    import uvicorn

    monkeypatch.setenv("YUNSHU_UDS", "")  # restored on teardown (serve exports env)
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    d = tempfile.mkdtemp(prefix="ys", dir="/tmp")
    path = d + "/s.sock"
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(path)  # a leftover socket file from a crashed run
    stale.close()
    try:
        r = CliRunner().invoke(app, ["serve", "--uds", path])
        assert r.exit_code == 0, r.output
        assert seen["uds"] == path
        assert "host" not in seen and "port" not in seen
        import os

        assert not os.path.exists(path)  # stale socket removed so bind succeeds
    finally:
        settings.clear_overrides()


def test_serve_default_is_tcp(monkeypatch):
    import uvicorn

    monkeypatch.setenv("YUNSHU_UDS", "")  # restored on teardown (serve exports env)
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    # A free port: worker servers own the fixed 18990-18999 range during runs.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    try:
        r = CliRunner().invoke(app, ["serve", "--port", str(port)])
        assert r.exit_code == 0, r.output
        assert seen["port"] == port and "uds" not in seen
    finally:
        settings.clear_overrides()
