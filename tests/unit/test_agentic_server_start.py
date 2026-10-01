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
