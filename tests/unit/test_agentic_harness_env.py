"""Agent harness env: zsh heredocs must work under the sandbox profile; Codex gets the product catalog."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research/agentic"))
import agents  # noqa: E402


def test_tmpprefix_under_run_tmp(tmp_path):
    env = agents.base_env(tmp_path / "home")
    assert env["TMPPREFIX"].startswith(str(tmp_path))
    # the short TMPDIR link resolves into the run's own tmp (inside the sandbox write list)
    assert str(Path(env["TMPDIR"]).resolve()).startswith(str(tmp_path.resolve()))


def test_tmpdir_is_short_enough_for_a_unix_socket(tmp_path):
    """Claude Code 2.1.316 binds srt-mux-<pid>-<n>.sock in TMPDIR; macOS caps socket paths at 104
    bytes (listen EINVAL ended every agentbench task)."""
    deep = tmp_path / ("x" * 60) / "run" / "home"
    env = agents.base_env(deep)
    assert len(env["TMPDIR"] + "srt-mux-99999-0.sock") < 104


def _sandbox_usable() -> bool:
    """sandbox-exec exists and may apply a profile here (it refuses inside an already sandboxed process)."""
    if not Path("/usr/bin/sandbox-exec").exists() or not Path("/bin/zsh").exists():
        return False
    probe = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


@pytest.mark.skipif(
    not _sandbox_usable(),
    reason="sandbox-exec unavailable here (not macOS, or nested inside another sandbox)",
)
def test_zsh_heredoc_works_in_sandbox(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    env = agents.base_env(home)
    prof = tmp_path / "p.sb"
    cmd = agents.wrap_sandbox(
        ["/bin/zsh", "-c", "cat <<'EOF'\nhello\nEOF"], prof, [home, work]
    )
    r = subprocess.run(cmd, env=env, cwd=work, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "hello", r.stderr
    # without TMPPREFIX the old behaviour reproduces the failure
    bad = {k: v for k, v in env.items() if k != "TMPPREFIX"}
    r = subprocess.run(cmd, env=bad, cwd=work, capture_output=True, text=True)
    assert r.returncode != 0 and "here document" in r.stderr


def test_codex_launch_writes_product_catalog(tmp_path):
    env = agents.base_env(tmp_path / "home")
    agents._codex(
        env,
        tmp_path / "home",
        tmp_path,
        "http://127.0.0.1:1",
        "m27",
        "hi",
        131072,
        4096,
    )
    chome = tmp_path / "home" / ".codex"
    t = tomllib.loads((chome / "config.toml").read_text())
    cat = json.loads(Path(t["model_catalog_json"]).read_text())
    (e,) = cat["models"]
    assert e["slug"] == "m27" and e["apply_patch_tool_type"] == "freeform"
