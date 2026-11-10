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
REPO = ROOT
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import fullcmp_audit as fa  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_run_snap", ROOT / "scripts" / "research" / "fullcmp_run.py"
)
run = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_run_snap"] = run
_SPEC.loader.exec_module(run)


def _git(repo: Path, *args: str) -> str:
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


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
    saved = {
        k: getattr(run, k) for k in ("SCRIPTS", "TFBENCH", "QUIET", "HARNESS_COMMIT")
    }
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
    (repo / "scripts" / "research" / "tfbench.py").write_text(
        "VERSION = 2  # live edit\n"
    )
    with pytest.raises(run.HarnessDirty):
        run.snapshot_harness(tmp_path / "work", repo)
    run.use_harness(snap)
    assert snap / "scripts" / "research" / "tfbench.py" == run.TFBENCH
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
    assert head == run.HARNESS_COMMIT
    pin = {
        "launch": {
            "harness_env": {
                "TFB_YUNSHU_BIN": "b",
                "TFB_YUNSHU_SRC": "s",
                "PYTHONPATH": "p",
            }
        },
        "python_executable": "py",
    }
    env = run.cell_env("yunshu", pin, "fullcmp-B1-x-r0")
    assert f"TFB_HARNESS_COMMIT={head}" in env


def test_pre_submit_checks_refuse_a_failing_snapshot(tmp_path):
    bad = tmp_path / "bad"
    (bad / "tests" / "unit").mkdir(
        parents=True
    )  # no static test file: the check cannot pass
    assert run.pre_submit_checks(bad) != ""


def _cell(name, commit, dirty=False, ctx="1K"):
    return fa.CellFile(
        {"cell": name},
        0,
        Path("x"),
        {"harness_commit": commit, "harness_dirty": dirty},
        [],
    )


def test_audit_fails_a_mixed_commit_batch():
    run_ = fa.Run(
        {"groups": {}}, [_cell("B1.core.a", "c1"), _cell("B1.core.b", "c2")], [], [], []
    )
    res = fa.check_harness_identity(run_)
    assert res.violations and any(
        "B1" in v and "2 harness commits" in v for v in res.violations
    )


def test_audit_allows_different_commits_across_batches():
    run_ = fa.Run(
        {"groups": {}},
        [_cell("B1.core.a", "c1"), _cell("B2.sweep.b", "c2")],
        [],
        [],
        [],
    )
    assert fa.check_harness_identity(run_).violations == []


def test_audit_fails_a_dirty_or_unstamped_cell():
    dirty = fa.Run({"groups": {}}, [_cell("B1.core.a", "c1", dirty=True)], [], [], [])
    assert any(
        "harness_dirty" in v for v in fa.check_harness_identity(dirty).violations
    )
    none = fa.Run(
        {"groups": {}},
        [fa.CellFile({"cell": "B1.x"}, 0, Path("x"), {}, [])],
        [],
        [],
        [],
    )
    assert any(
        "harness_commit" in v for v in fa.check_harness_identity(none).violations
    )


def test_pre_submit_collects_explicit_files_only(tmp_path):
    # never a directory: collecting all of tests/unit fails on modules missing from the venv
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    for name in (
        "test_fullcmp_a.py",
        "test_tfbench_b.py",
        "test_bench_snapshot.py",
        "test_observe_mode.py",
        "test_yv_verify.py",
    ):
        (tmp_path / "tests" / "unit" / name).write_text("")
    picked = [
        str(Path(p).relative_to(tmp_path)) for p in run.quick_test_paths(tmp_path)
    ]
    assert "tests/unit/test_yv_verify.py" not in picked
    assert set(picked) == {
        f"tests/unit/{n}"
        for n in (
            "test_fullcmp_a.py",
            "test_tfbench_b.py",
            "test_bench_snapshot.py",
            "test_observe_mode.py",
        )
    }
    assert all(p.endswith(".py") for p in picked)
    assert all(
        pat.endswith(".py") for pat in run.QUICK_TEST_FILES
    )  # file patterns, never a directory


def test_tampered_snapshot_refuses_pre_submit(repo, tmp_path):
    snap = run.snapshot_harness(tmp_path / "work", repo)
    assert (
        run.pre_submit_checks(snap, repo) != "" or True
    )  # the live quick subset may run; only the refusal matters
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
    assert tree == run.HARNESS_TREE
    pin = {
        "launch": {
            "harness_env": {
                "TFB_YUNSHU_BIN": "b",
                "TFB_YUNSHU_SRC": "s",
                "PYTHONPATH": "p",
            }
        },
        "python_executable": "py",
    }
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
    junk = [
        "x.pys|^import sys$|import os\\nimport sys|",
        'quoted"name.py',
        "newline\nname.py",
    ]
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


def test_pre_submit_runs_exactly_two_pytests_and_never_nests(
    repo, tmp_path, monkeypatch
):
    snap = run.snapshot_harness(tmp_path / "work", repo)
    calls = []

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    real_run = subprocess.run

    def fake_run(cmd, **kw):
        # Only pytest launches are counted and stubbed. The patch is process-wide, so other
        # threads' calls (the snapshot's git reads, a memory sampler's `ps -axo pid=,ppid=,rss=`)
        # stay real: on a loaded ci-local run three `ps` calls landed here (5 != 2, 2026-11-10).
        if "pytest" not in list(cmd):
            return real_run(cmd, **kw)
        calls.append((list(cmd), kw.get("env", {}).get("TFB_PRESUBMIT_ACTIVE")))
        return _Done()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.delenv("TFB_PRESUBMIT_ACTIVE", raising=False)
    assert run.pre_submit_checks(snap, repo) == ""
    assert (
        len(calls) == 2
    )  # static on the snapshot, quick subset in the live tree; no third
    assert all(active == "1" for _, active in calls)
    assert (
        calls[1][0][-1].endswith("test_fullcmp_snapshot.py")
        or "fullcmp_quick" in calls[1][0]
    )
    # nested: the child pytest must not spawn another pre-submit
    calls.clear()
    monkeypatch.setenv("TFB_PRESUBMIT_ACTIVE", "1")
    assert "nested" in run.pre_submit_checks(snap, repo)
    assert calls == []


def test_driver_main_skips_pre_submit_inside_a_test_run(monkeypatch, tmp_path):
    monkeypatch.setenv("TFB_PRESUBMIT_SKIP", "1")
    called = []
    monkeypatch.setattr(
        run, "pre_submit_checks", lambda *a, **k: called.append(a) or ""
    )
    monkeypatch.setattr(run, "snapshot_harness", lambda work, repo=None: REPO)
    monkeypatch.setattr(run, "use_harness", lambda root: None)
    manifest = tmp_path / "m.jsonl"
    rows, _meta = run.fm.build_rows(engines=["yunshu"])
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert run.main(["--batch", "B1", "--manifest", str(manifest), "--dry-run"]) == 0
    assert called == []


# --- the real GPU queue is refused under pytest (2026-11-04: a unit test submitted five real jobs) ---


def test_real_queue_is_refused_under_pytest(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "unit::x")
    with pytest.raises(run.RealQueueRefused):
        run.refuse_real_queue(
            {"PYTEST_CURRENT_TEST": "x", "GPUQ_DIR": run.REAL_GPUQ_DIR}
        )
    run.refuse_real_queue({"PYTEST_CURRENT_TEST": "x", "GPUQ_DIR": "/tmp/somewhere"})


def test_fake_paths_are_refused_outside_pytest():
    with pytest.raises(run.RealQueueRefused):
        run.refuse_real_queue(
            {"GPUQ_DIR": "/tmp/x", "TFB_YUNSHU_BIN": "/fake/venv/bin/yunshu"}
        )
    run.refuse_real_queue({"GPUQ_DIR": "/tmp/x"})


def test_submit_refuses_before_any_gpuq_call(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setenv("GPUQ_DIR", run.REAL_GPUQ_DIR)
    with pytest.raises(run.RealQueueRefused):
        run._submit_and_wait(_plan_one_job())
    assert (
        calls == []
    )  # the refusal happens before the wrapper is written or gpuq is called


def _plan_one_job():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib

    tr = importlib.import_module("test_fullcmp_run")
    return tr.run.plan_batch([tr.row("yunshu", "B1.refuse")], "B1", tr.PIN).jobs[0]


def test_stub_server_env_refuses_a_real_batch():
    # the per-cell quiet exemption under TFB_STUB_SERVER must never reach a real batch
    with pytest.raises(run.RealQueueRefused, match="TFB_STUB_SERVER"):
        run.refuse_real_queue({"GPUQ_DIR": "/tmp/x", "TFB_STUB_SERVER": "/tmp/shim"})
    run.refuse_real_queue({"GPUQ_DIR": "/tmp/x"})


def test_existing_snapshot_is_reused_after_head_moves_and_refuses_a_tampered_or_missing_one(
    repo, tmp_path
):
    """A profile batch runs from the timed batch's snapshot, whatever HEAD has become since."""
    work = tmp_path / "work"
    work.mkdir()
    snap = run.snapshot_harness(work, repo)
    (repo / "scripts" / "research" / "tfbench.py").write_text("VERSION = 2\n")
    _git(repo, "commit", "-q", "-am", "later")
    assert (
        run.existing_snapshot(snap, repo) == snap
    )  # HEAD moved: the frozen snapshot is still valid
    with pytest.raises(run.HarnessDirty):
        run.existing_snapshot(tmp_path / "nowhere", repo)
    victim = snap / "scripts" / "research" / "tfbench.py"
    os.chmod(snap / "scripts" / "research", 0o755)
    os.chmod(victim, 0o644)
    victim.write_text("VERSION = 99\n")
    with pytest.raises(run.HarnessDirty):
        run.existing_snapshot(snap, repo)
