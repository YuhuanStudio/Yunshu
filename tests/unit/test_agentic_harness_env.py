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
    assert env["TMPPREFIX"].startswith(env["TMPDIR"])
    assert env["TMPPREFIX"].startswith(str(tmp_path))


@pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").exists() or not Path("/bin/zsh").exists(),
    reason="macOS only",
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
        env, tmp_path / "home", tmp_path, "http://127.0.0.1:1", "m27", "hi", 131072, 4096
    )
    chome = tmp_path / "home" / ".codex"
    t = tomllib.loads((chome / "config.toml").read_text())
    cat = json.loads(Path(t["model_catalog_json"]).read_text())
    (e,) = cat["models"]
    assert e["slug"] == "m27" and e["apply_patch_tool_type"] == "freeform"
