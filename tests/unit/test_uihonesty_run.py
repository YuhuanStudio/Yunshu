"""The uihonesty real-run script's helpers work without a GPU (CPU only)."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "uihonesty_run", ROOT / "scripts/research/uihonesty_run.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_port_free_sees_a_listener():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert not mod.port_free(port)
    assert mod.port_free(port)


def test_killgroup_accepts_none():
    mod.killgroup(None)


def test_pick_console_port_skips_busy_and_engine_ports(monkeypatch):
    with socket.socket() as a, socket.socket() as b:
        a.bind(("127.0.0.1", 0))
        a.listen()
        b.bind(("127.0.0.1", 0))
        b.listen()
        busy = a.getsockname()[1]
        engine = b.getsockname()[1]
        free = busy + 1 if busy + 1 != engine else busy + 2
        monkeypatch.setattr(mod, "CONSOLE_PORTS", (busy, engine, free))
        assert mod.pick_console_port(engine) == free
