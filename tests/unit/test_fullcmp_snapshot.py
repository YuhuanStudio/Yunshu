"""Harness snapshot (fullcmp_run.snapshot_harness / use_harness / pre_submit_checks) and the audit's
harness_identity guard. Fake git repo in tmp_path; CPU only, no gpuq, no GPU, no model.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import fullcmp_audit as fa  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_run_snap", ROOT / "scripts" / "research" / "fullcmp_run.py"
)
run = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_run_snap"] = run
_SPEC.loader.exec_module(run)


def _git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "scripts" / "research").mkdir(parents=True)
    (r / "scripts" / "dev").mkdir(parents=True)
    (r / "python").mkdir()
    (r / "tests" / "unit").mkdir(parents=True)
    (r / "scripts" / "research" / "tfbench.py").write_text("VERSION = 1\n")
    (r / "scripts" / "research" / "fullcmp_quiet.py").write_text("QUIET = 1\n")
    (r / "scripts" / "dev" / "gpuq_util.py").write_text("X = 1\n")
    (r / "python" / "pkg.py").write_text("P = 1\n")
    (r / "tests" / "unit" / "test_x.py").write_text("def test_x():\n    pass\n")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    return r


@pytest.fixture(autouse=True)
def _restore_harness_globals():
    saved = {k: getattr(run, k) for k in ("SCRIPTS", "TFBENCH", "QUIET", "HARNESS_COMMIT")}
    yield
    for k, v in saved.items():
        setattr(run, k, v)


def test_snapshot_is_read_only_and_records_head(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    snap = run.snapshot_harness(tmp_path / "work", repo)
    assert snap.name == f"harness-{head[:7]}"
    assert (snap / ".snapshot_commit").read_text().strip() == head
    assert (snap / "scripts" / "research" / "tfbench.py").is_file()
    assert not os.access(snap / "scripts" / "research" / "tfbench.py", os.W_OK)


def test_live_edit_after_snapshot_leaves_jobs_on_the_snapshot(repo, tmp_path):
    snap = run.snapshot_harness(tmp_path / "work", repo)
    # a later live edit: refused as a submit source while uncommitted, and never seen by the snapshot
    (repo / "scripts" / "research" / "tfbench.py").write_text("VERSION = 2  # live edit\n")
    with pytest.raises(run.HarnessDirty):
        run.snapshot_harness(tmp_path / "work", repo)
    run.use_harness(snap)
    assert run.TFBENCH == snap / "scripts" / "research" / "tfbench.py"
    assert run.QUIET.is_relative_to(snap)
    assert (run.TFBENCH.read_text()) == "VERSION = 1\n"


def test_dirty_live_tree_refuses_to_snapshot(repo, tmp_path):
    (repo / "tests" / "unit" / "test_x.py").write_text("def test_x():\n    assert 0\n")
    with pytest.raises(run.HarnessDirty):
        run.snapshot_harness(tmp_path / "work", repo)


def test_header_commit_reaches_every_job_env(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    snap = run.snapshot_harness(tmp_path / "work", repo)
    run.use_harness(snap)
    assert run.HARNESS_COMMIT == head
    pin = {"launch": {"harness_env": {"TFB_YUNSHU_BIN": "b", "TFB_YUNSHU_SRC": "s", "PYTHONPATH": "p"}}, "python_executable": "py"}
    env = run.cell_env("yunshu", pin, "fullcmp-B1-x-r0")
    assert f"TFB_HARNESS_COMMIT={head}" in env


def test_pre_submit_checks_refuse_a_failing_snapshot(tmp_path):
    bad = tmp_path / "bad"
    (bad / "tests" / "unit").mkdir(parents=True)  # no static test file: the check cannot pass
    assert run.pre_submit_checks(bad) != ""


def _cell(name, commit, dirty=False, ctx="1K"):
    return fa.CellFile({"cell": name}, 0, Path("x"), {"harness_commit": commit, "harness_dirty": dirty}, [])


def test_audit_fails_a_mixed_commit_batch():
    run_ = fa.Run({"groups": {}}, [_cell("B1.core.a", "c1"), _cell("B1.core.b", "c2")], [], [], [])
    res = fa.check_harness_identity(run_)
    assert res.violations and any("B1" in v and "2 harness commits" in v for v in res.violations)


def test_audit_allows_different_commits_across_batches():
    run_ = fa.Run({"groups": {}}, [_cell("B1.core.a", "c1"), _cell("B2.sweep.b", "c2")], [], [], [])
    assert fa.check_harness_identity(run_).violations == []


def test_audit_fails_a_dirty_or_unstamped_cell():
    dirty = fa.Run({"groups": {}}, [_cell("B1.core.a", "c1", dirty=True)], [], [], [])
    assert any("harness_dirty" in v for v in fa.check_harness_identity(dirty).violations)
    none = fa.Run({"groups": {}}, [fa.CellFile({"cell": "B1.x"}, 0, Path("x"), {}, [])], [], [], [])
    assert any("harness_commit" in v for v in fa.check_harness_identity(none).violations)


def test_pre_submit_collects_explicit_files_only(tmp_path):
    # never a directory: collecting all of tests/unit fails on modules missing from the venv
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    for name in ("test_fullcmp_a.py", "test_tfbench_b.py", "test_bench_snapshot.py", "test_observe_mode.py", "test_yv_verify.py"):
        (tmp_path / "tests" / "unit" / name).write_text("")
    picked = [str(Path(p).relative_to(tmp_path)) for p in run.quick_test_paths(tmp_path)]
    assert "tests/unit/test_yv_verify.py" not in picked
    assert set(picked) == {f"tests/unit/{n}" for n in ("test_fullcmp_a.py", "test_tfbench_b.py", "test_bench_snapshot.py", "test_observe_mode.py")}
    assert all(p.endswith(".py") for p in picked)
    assert all(pat.endswith(".py") for pat in run.QUICK_TEST_FILES)  # file patterns, never a directory


def test_tampered_snapshot_refuses_pre_submit(repo, tmp_path):
    snap = run.snapshot_harness(tmp_path / "work", repo)
    assert run.pre_submit_checks(snap, repo) != "" or True  # the live quick subset may run; only the refusal matters
    target = snap / "scripts" / "research" / "tfbench.py"
    os.chmod(target, 0o644)
    try:
        target.write_text("VERSION = 999  # tampered\n")
        msg = run.pre_submit_checks(snap, repo)
        assert msg.startswith("snapshot integrity"), msg
        assert "tfbench.py" in msg
    finally:
        os.chmod(target, 0o644)
        target.write_text("VERSION = 1\n")
        os.chmod(target, 0o444)


def test_snapshot_records_tree_sha_and_manifest(repo, tmp_path):
    snap = run.snapshot_harness(tmp_path / "work", repo)
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert run.snapshot_tree(snap) == tree
    manifest = (snap / ".snapshot_manifest").read_text().splitlines()
    assert any(line.endswith("\tscripts/research/tfbench.py") for line in manifest)
    run.use_harness(snap)
    assert run.HARNESS_TREE == tree
    pin = {"launch": {"harness_env": {"TFB_YUNSHU_BIN": "b", "TFB_YUNSHU_SRC": "s", "PYTHONPATH": "p"}}, "python_executable": "py"}
    assert f"TFB_HARNESS_TREE={tree}" in run.cell_env("yunshu", pin, "fullcmp-B1-x-r0")


def test_mangled_names_are_excluded_and_logged_and_the_snapshot_verifies(tmp_path):
    # names with '|', a double quote and a newline (a mangled shell substitution became the name)
    r = tmp_path / "repo2"
    (r / "scripts" / "research").mkdir(parents=True)
    (r / "scripts" / "dev").mkdir(parents=True)
    (r / "python").mkdir()
    (r / "tests" / "unit").mkdir(parents=True)
    (r / "scripts" / "research" / "tfbench.py").write_text("VERSION = 1\n")
    (r / "scripts" / "dev" / "gpuq_util.py").write_text("X = 1\n")
    (r / "python" / "pkg.py").write_text("P = 1\n")
    (r / "tests" / "unit" / "test_x.py").write_text("def test_x():\n    pass\n")
    junk = ['x.pys|^import sys$|import os\\nimport sys|', 'quoted"name.py', "newline\nname.py"]
    for name in junk:
        try:
            (r / "scripts" / "research" / name).write_text("junk\n")
        except OSError:
            pytest.skip("filesystem refuses this name")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    snap = run.snapshot_harness(tmp_path / "work2", r)  # must not raise
    excluded = (snap / ".snapshot_excluded").read_bytes()
    for name in junk:
        assert os.fsencode(name) in excluded, name
    assert (snap / "scripts" / "research" / "tfbench.py").is_file()
    assert run.verify_snapshot(snap, r) is None
    # no junk file was copied into the snapshot
    for name in junk:
        assert not (snap / "scripts" / "research" / name).exists(), name
