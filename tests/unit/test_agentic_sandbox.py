"""srt settings the agentic harness generates (pure, no GPU, no srt needed)."""

import importlib.util
import json
import sys
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts/research/agentic/agents.py"
_spec = importlib.util.spec_from_file_location("agentic_agents_srt", _p)
agents = importlib.util.module_from_spec(_spec)
sys.modules["agentic_agents_srt"] = agents
_spec.loader.exec_module(agents)


def test_settings_policy(tmp_path):
    run, work = tmp_path / "run", tmp_path / "work"
    s = agents.srt_settings([run, work], 18995)
    assert s["filesystem"]["allowWrite"] == [str(run), str(work)]
    assert s["filesystem"]["allowRead"] == []
    deny = s["filesystem"]["denyRead"]
    assert str(agents.REAL_HOME / ".ssh") in deny
    assert str(agents.REAL_HOME / ".claude") in deny
    assert len(deny) == len(agents.DENY_HOME)
    # network: only our port by name, never a wildcard or a public host
    assert s["network"]["allowedDomains"] == ["localhost:18995", "127.0.0.1:18995"]
    assert s["network"]["deniedDomains"] == []  # srt requires the key
    assert s["network"]["allowLocalBinding"] is True


def test_wrap_srt_command(tmp_path):
    cmd = agents.wrap_srt(
        ["claude", "-p", "it's a prompt; rm -rf /"],
        tmp_path / "s.json",
        [tmp_path],
        1234,
    )
    assert cmd[0] == agents.NODE and cmd[1].endswith("sandbox-runtime/dist/cli.js")
    assert cmd[2:4] == ["--settings", str(tmp_path / "s.json")] and cmd[4] == "-c"
    import shlex

    assert shlex.split(cmd[5]) == ["claude", "-p", "it's a prompt; rm -rf /"]
    assert json.loads((tmp_path / "s.json").read_text())["network"]["allowedDomains"][
        0
    ].endswith(":1234")


def test_zsh_tmpprefix_inside_run_home(tmp_path):
    env = agents.base_env(tmp_path / "home")
    assert env["TMPPREFIX"].startswith(str(tmp_path / "home"))


def test_prepare_modes(tmp_path, monkeypatch):
    for mode, head in (("srt", agents.NODE), ("seatbelt", "/usr/bin/sandbox-exec")):
        monkeypatch.setattr(agents, "SANDBOX_MODE", mode)
        l = agents.prepare(
            "claude",
            tmp_path / mode,
            tmp_path / mode / "w",
            "http://127.0.0.1:18993",
            "m",
            "hi",
        )
        assert l.cmd[0] == head
