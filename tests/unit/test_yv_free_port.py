"""Harness servers take a free port, not a fixed one (2026-11-02: a fixed port collided with an
orphan server and the job died in 0 s with Address already in use)."""

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import memory_ab  # noqa: E402
from verify import gate  # noqa: E402


def _held():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_memory_ab_skips_a_busy_port_of_the_pool():
    held, port = _held()
    try:
        got = memory_ab.free_port(first=port, last=port)
        assert got != port
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", got))  # really free
    finally:
        held.close()


def test_gate_job_port_is_free_at_submit_time():
    held, port = _held()
    try:
        got = gate._free_port(first=port, last=port)
        assert got != port and got > 0
    finally:
        held.close()


def test_no_fixed_gate_port_left():
    assert not hasattr(gate, "GATE_PORT")
