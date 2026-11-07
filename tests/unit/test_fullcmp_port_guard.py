"""tfbench refuses to start a cell on a bound port (fail closed, name the pid, never kill it).

No GPU, no server: a real loopback listener stands in for a live gpuq server, and lsof is faked.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import tfbench  # noqa: E402

POOL = range(
    18990, 19000
)  # the shared gpuq pool, 18990-18999 (free_port's TFB_PORT_LAST default)


@pytest.fixture
def listener():
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(4)
    yield sock.getsockname()[1]
    sock.close()


def test_parse_lsof_pids_reads_one_pid_per_line_and_ignores_junk():
    assert tfbench.parse_lsof_pids("4242\n\n17\nnot-a-pid\n4242\n") == [17, 4242]
    assert tfbench.parse_lsof_pids("") == []


def test_listening_pids_builds_the_read_only_lsof_query():
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="909\n", stderr="")

    assert tfbench.listening_pids(18991, run=fake_run) == [909]
    assert seen["argv"] == ["lsof", "-nP", "-iTCP:18991", "-sTCP:LISTEN", "-t"]


def test_bound_port_is_refused_with_the_holder_named_and_nothing_killed(listener):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="4242\n", stderr="")

    with pytest.raises(RuntimeError) as exc:
        tfbench.assert_port_free(listener, run=fake_run)
    msg = str(exc.value)
    assert f"port {listener} is already bound by pid 4242" in msg
    assert "Nothing was killed" in msg
    assert calls and calls[0][-1] == "-t"
    # the listener is still up: the guard never touched it
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", listener)) == 0


def test_free_port_passes_without_asking_lsof():
    port = tfbench.ephemeral_port()

    def boom(argv, **kw):
        raise AssertionError("lsof must not run for a free port")

    tfbench.assert_port_free(port, run=boom)


def test_bound_port_with_unnamed_holder_still_refuses(listener, monkeypatch):
    monkeypatch.setattr(tfbench, "listening_pids", lambda port, run=None: [])
    with pytest.raises(RuntimeError, match="unknown"):
        tfbench.assert_port_free(listener)


def test_real_server_launch_refuses_a_bound_port_before_creating_anything(
    listener, monkeypatch, tmp_path
):
    monkeypatch.setattr(tfbench, "free_port", lambda *a, **k: listener)
    monkeypatch.setattr(tfbench, "listening_pids", lambda port, run=None: [os.getpid()])
    monkeypatch.setattr(tfbench, "OUT", tmp_path / "out")

    def no_popen(*a, **k):
        raise AssertionError("no server process may start on a bound port")

    monkeypatch.setattr(tfbench.subprocess, "Popen", no_popen)
    with pytest.raises(RuntimeError, match=f"pid {os.getpid()}"):
        tfbench.Srv("yunshu", {}, "guard-test")
    assert not (tmp_path / "out" / "home").exists()


def test_stub_runs_take_an_ephemeral_port_outside_the_pool(monkeypatch):
    monkeypatch.setenv("TFB_STUB_SERVER", str(HERE))
    for _ in range(5):
        port = tfbench.free_port()
        assert port not in POOL
        assert port > 1024
