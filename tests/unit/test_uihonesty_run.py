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
