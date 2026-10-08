"""Failed benchmark startup must release its own server and parent log handle."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def servers(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/research/agentic/servers.py"
    spec = importlib.util.spec_from_file_location("agentic_servers_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SERVER_HOME", tmp_path / "home")
    return module


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "launch"])
def test_failed_start_cleans_up_owned_process(servers, monkeypatch, tmp_path, failure):
    logs = []

    class Process:
        returncode = None
        pid = 123456

        def poll(self):
            return None

    def launch(*args, **kwargs):
        logs.append(kwargs["stdout"])
        if failure == "launch":
            raise OSError("cannot launch")
        return Process()

    server = servers.Server("yunshu", "existing-model", 18999, tmp_path / "server.log")
    cleaned = []
    monkeypatch.setattr(servers.subprocess, "Popen", launch)
    monkeypatch.setattr(server, "kill", lambda: cleaned.append(server.proc))
    if failure == "interrupt":

        def interrupted():
            raise KeyboardInterrupt()

        monkeypatch.setattr(servers.time, "monotonic", interrupted)
        error = KeyboardInterrupt
    else:
        error = OSError if failure == "launch" else RuntimeError
    with pytest.raises(error):
        server.start(ready_timeout=0)
    assert len(cleaned) == 1
    assert logs[0].closed
    assert (cleaned[0] is None) is (failure == "launch")


def _hold(servers, ports):
    import socket

    socks = []
    for p in ports:
        s = socket.socket()
        s.bind(("127.0.0.1", p))
        s.listen()
        socks.append(s)
    return socks


def test_free_ports_waits_for_a_port_then_succeeds(servers, monkeypatch):
    monkeypatch.setattr(servers, "PORT_RANGE", range(28990, 28992))
    socks = _hold(servers, [28990, 28991])
    ticks = []

    def sleep(_):
        ticks.append(1)
        if len(ticks) == 2:
            for s in socks:
                s.close()

    try:
        assert servers.free_ports(2, wait_s=60, interval_s=0.01, sleep=sleep) == [
            28990,
            28991,
        ]
    finally:
        for s in socks:
            s.close()
    assert len(ticks) == 2


def test_free_ports_raises_after_the_bounded_wait(servers, monkeypatch):
    monkeypatch.setattr(servers, "PORT_RANGE", range(28990, 28991))
    socks = _hold(servers, [28990])
    now = [0.0]
    try:
        with pytest.raises(RuntimeError, match="no free port"):
            servers.free_ports(
                1,
                wait_s=10,
                interval_s=3,
                sleep=lambda d: now.__setitem__(0, now[0] + d),
                clock=lambda: now[0],
            )
    finally:
        for s in socks:
            s.close()
    assert now[0] == 10


def test_start_server_retries_on_a_bind_race(servers, monkeypatch, tmp_path):
    monkeypatch.setattr(servers, "PORT_RANGE", range(28990, 28996))
    made = []

    class Fake:
        def __init__(self, port):
            self.port = port
            self.log = tmp_path / f"{port}.log"
            made.append(port)

        def start(self):
            if len(made) < 3:
                self.log.write_text("OSError: address already in use\n")
                raise RuntimeError("server exited early rc=1")
            return self

        def log_tail(self):
            return self.log.read_text()

    server, ports = servers.start_server(2, Fake, wait_s=0)
    assert made == [28990, 28991, 28992]
    assert server.port == 28992 and len(ports) == 2


def test_start_server_does_not_retry_other_failures(servers, monkeypatch, tmp_path):
    monkeypatch.setattr(servers, "PORT_RANGE", range(28990, 28996))
    made = []

    class Fake:
        def __init__(self, port):
            made.append(port)
            self.log = tmp_path / "x.log"
            self.log.write_text("model load failed\n")

        def start(self):
            raise RuntimeError("server exited early")

        def log_tail(self):
            return self.log.read_text()

    with pytest.raises(RuntimeError):
        servers.start_server(1, Fake, wait_s=0)
    assert len(made) == 1
