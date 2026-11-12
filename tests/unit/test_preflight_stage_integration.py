"""stage_preflight end to end on a real two-commit git repo (no GPU, no pytest run needed).

The candidate adds a python/ module with no unit test: the stage fails unless the module is
allowlisted with a reason, and a malformed allowlist fails closed. The import check runs the real
interpreter against the stub packages written here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from verify import stages  # noqa: E402
from verify.core import Arm, RunDir  # noqa: E402

ALLOW = "scripts/verify/preflight_allowlist.txt"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    for pkg in ("yunshu_engine", "yunshu_gateway", "yunshu_kv"):
        d = root / "python" / pkg
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")
    (root / "tests" / "unit").mkdir(parents=True)
    (root / "scripts" / "verify").mkdir(parents=True)
    (root / "scripts" / "verify" / "__init__.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "python" / "yunshu_engine" / "orphan.py").write_text("Y = 2\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "cand")
    cand = _git(root, "rev-parse", "HEAD")
    return root, base, cand


def _ctx(tmp_path: Path, root: Path, base: str, cand: str):
    run = RunDir(tmp_path / "run")
    return SimpleNamespace(
        run=run,
        exe=SimpleNamespace(jobs=[]),
        base=Arm("base", "x", base, root, ""),
        cand=Arm("cand", "x", cand, root, ""),
        env={},
        cand_env={},
        base_env={},
        model="",
        model_name="",
        suite={},
        mem_gb=8.0,
        py=sys.executable,
    )


def _allow(root: Path, text: str) -> None:
    (root / ALLOW).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "allowlist")


def test_unmapped_module_fails_the_stage(tmp_path):
    root, base, cand = _repo(tmp_path)
    res = stages.stage_preflight(_ctx(tmp_path, root, base, cand))
    assert res.passed is False
    assert res.numbers["unmapped_python"] == ["python/yunshu_engine/orphan.py"]
    assert any("orphan.py" in r for r in res.reasons)


def test_allowlisted_module_passes(tmp_path):
    root, base, cand = _repo(tmp_path)
    _allow(root, "python/yunshu_engine/orphan.py  # covered by the smoke stage\n")
    res = stages.stage_preflight(_ctx(tmp_path, root, base, cand))
    assert res.passed is True, res.reasons
    assert res.numbers["unmapped_python"] == []


def test_entry_without_reason_fails_closed(tmp_path):
    root, base, cand = _repo(tmp_path)
    _allow(root, "python/yunshu_engine/orphan.py\n")
    res = stages.stage_preflight(_ctx(tmp_path, root, base, cand))
    assert res.passed is False
    assert any("needs a path and a" in r for r in res.reasons)


@pytest.mark.parametrize("entry", ["# only a comment\n"])
def test_empty_allowlist_is_not_an_exemption(tmp_path, entry):
    root, base, cand = _repo(tmp_path)
    _allow(root, entry)
    res = stages.stage_preflight(_ctx(tmp_path, root, base, cand))
    assert res.passed is False
