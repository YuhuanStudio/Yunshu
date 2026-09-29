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
    try:
        r = CliRunner().invoke(app, ["serve", "--port", "18991"])
        assert r.exit_code == 0, r.output
        assert seen["port"] == 18991 and "uds" not in seen
    finally:
        settings.clear_overrides()
