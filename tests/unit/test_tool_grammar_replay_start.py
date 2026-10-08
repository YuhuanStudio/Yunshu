"""tool_grammar_replay waits for a free pool port instead of failing at once."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def replay():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/tool_grammar_replay.py"
    )
    spec = importlib.util.spec_from_file_location("tool_grammar_replay_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_instant_failure_on_busy_pool(replay, monkeypatch, tmp_path):
    import servers

    busy = {"n": 0}

    def bindable(port):
        busy["n"] += 1
        return busy["n"] > 3 * len(servers.PORT_RANGE)  # busy for three scans

    monkeypatch.setattr(servers, "port_bindable", bindable)
    sleeps = []
    ports = []
    srv = replay.start_server(
        "m",
        None,
        {},
        tmp_path / "s.log",
        make=lambda c, s, e, log, port: ports.append(port) or "srv",
        sleep=sleeps.append,
    )
    assert srv == "srv" and len(sleeps) == 3 and ports[0] in servers.PORT_RANGE


def test_bind_race_retries_on_another_port(replay, monkeypatch, tmp_path):
    import servers

    monkeypatch.setattr(servers, "port_bindable", lambda p: True)
    log = tmp_path / "s.log"
    tried = []

    def make(c, s, e, log_, port):
        tried.append(port)
        if len(tried) == 1:
            log_.write_text("OSError: [Errno 48] address already in use\n")
            raise RuntimeError("server exited early rc=1")
        return port

    assert replay.start_server("m", None, {}, log, make=make) == tried[1]
    assert tried[0] != tried[1]


def test_other_failures_are_not_retried(replay, monkeypatch, tmp_path):
    import servers

    monkeypatch.setattr(servers, "port_bindable", lambda p: True)

    def make(*a):
        raise RuntimeError("server exited early rc=1")

    with pytest.raises(RuntimeError):
        replay.start_server("m", None, {}, tmp_path / "s.log", make=make)
