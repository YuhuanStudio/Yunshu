"""Benchmark servers must not launch the shared .venv/bin/yunshu console script.

A plain `uv run` re-syncs the main checkout's .venv and rewrites that script while gpuq
jobs start it (FileNotFoundError at job start). The server command must be the venv
interpreter running the CLI module instead.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_tfbench():
    script = Path(__file__).resolve().parents[2] / "scripts/research/tfbench.py"
    spec = importlib.util.spec_from_file_location("venv_entry_tfbench", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaunchStoppedError(Exception):
    pass


@pytest.fixture
def captured(tmp_path, monkeypatch):
    module = _load_tfbench()
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "YUNSHU_SRC", "")
    monkeypatch.setattr(module, "host_used_bytes", lambda: 0)
    monkeypatch.setattr(module, "free_port", lambda: 18999)
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        raise LaunchStoppedError

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "spec_request", lambda e, env, m: (None, env))
    return module, calls


def test_yunshu_server_command_runs_module_not_console_script(captured):
    module, calls = captured
    with pytest.raises(LaunchStoppedError):
        module.Srv("yunshu", {}, "venv-entry-unit", model="/tmp/fake-model")
    assert len(calls) == 1
    cmd = calls[0]
    assert "bin/yunshu" not in cmd[0]
    assert Path(cmd[0]).name.startswith("python")
    assert cmd[1:3] == ["-m", "yunshu_cli"]
    assert cmd[3] == "serve"
