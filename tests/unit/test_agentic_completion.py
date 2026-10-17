"""A time-budget stop is resumable output, never a successful complete measurement."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_budget_expiry_writes_incomplete_and_fails(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/agentic/run_agentic.py"
    )
    spec = importlib.util.spec_from_file_location("agentic_completion_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "BUILD", tmp_path)
    monkeypatch.setattr(runner, "hw", lambda: {})
    monkeypatch.setattr(runner, "git_sha", lambda: "test")
    monkeypatch.setattr(runner.agents, "agent_version", lambda *a: "test")
    tasks = [SimpleNamespace(id="task")]
    monkeypatch.setattr(runner.tasklib, "all_tasks", lambda *a: tasks)
    monkeypatch.setattr(runner.tasklib, "select", lambda *a: tasks)
    monkeypatch.setattr(runner.tasklib, "dataset_commit", lambda: "test")
    out = tmp_path / "run.jsonl"
    args = SimpleNamespace(
        output=str(out),
        engine_label="test",
        agent="codex",
        tasks="all",
        repeat=1,
        shard_i=0,
        shard_n=1,
        serve=None,
        engine_url="http://127.0.0.1:18990",
        server_pid=0,
        model="fake",
        proxy_port=18991,
        checkpoint="fake",
        server_arg=[],
        context=32768,
        timeout_min=1,
        budget_min=0,
    )
    assert runner.cmd_run(args) == 1
    final = json.loads(out.read_text().splitlines()[-1])
    assert (
        final["complete"] is False and final["completed"] == 0 and final["planned"] == 1
    )
