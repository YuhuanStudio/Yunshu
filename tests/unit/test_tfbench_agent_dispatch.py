"""The advertised --part agent replay must actually call the agent battery."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_agent_part_is_dispatched_and_server_closed(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/research/tfbench.py"
    spec = importlib.util.spec_from_file_location("cspec_tfbench", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    server = SimpleNamespace(
        ready_s=0,
        cmd=[],
        url="unused",
        model="fake",
        kill=lambda: calls.append("kill"),
        extra_env={},
        requested_spec_mode=None,
        engaged_spec_mode=None,
        verify_spec_mode=lambda: None,
    )
    monkeypatch.setattr(module, "Srv", lambda *args, **kwargs: server)
    monkeypatch.setattr(module, "send", lambda *args: {})
    monkeypatch.setattr(module, "part_agent", lambda *args: calls.append("agent"))
    output = tmp_path / "agent.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tfbench", "--engine", "yunshu", "--part", "agent", "--out", str(output)],
    )
    module.main()
    assert calls == ["agent", "kill"]
    assert "part_done" in output.read_text()
