"""scripts/dev/yv_tree_gc.py: which verify worktrees are removed, and which are kept.

Each test builds a throwaway git repo (tmp_path) with main, a verify-style trees dir and a runs dir."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "yv_tree_gc", ROOT / "scripts/dev/yv_tree_gc.py"
)
gc = importlib.util.module_from_spec(_spec)
sys.modules["yv_tree_gc"] = gc
assert _spec.loader is not None
_spec.loader.exec_module(gc)

OLD = time.time() - 100 * 3600  # older than the 48 h default


def _git(cwd: Path, *args: str) -> str:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    r = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return r.stdout.strip()


@pytest.fixture()
def env(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    verify = tmp_path / "verify"
    trees = verify / "trees"
    runs = verify / "runs"
    trees.mkdir(parents=True)
    runs.mkdir(parents=True)
    return {
        "repo": repo,
        "verify": verify,
        "trees": trees,
        "runs": runs,
        "base": base,
        "tmp": tmp_path,
    }


def _tree(env: dict, name: str, commit: str, old: bool = True) -> Path:
    path = env["trees"] / name
    _git(env["repo"], "worktree", "add", "-q", "--detach", str(path), commit)
    if old:
        os.utime(path, (OLD, OLD))
    return path


def _verdict(
    env: dict, tree: Path, now: float | None = None, census: bool = False, procs=None
):
    refs = gc.protecting_runs(env["runs"], now or time.time())
    return gc.classify(
        tree,
        env["repo"],
        now or time.time(),
        0.0 if census else 48.0,
        refs,
        procs or [],
        census,
    )


def test_remove_old_clean_merged(env):
    tree = _tree(env, "aaaa", env["base"])
    v = _verdict(env, tree)
    assert v.remove, v.reason
    assert gc.remove_tree(env["repo"], tree) == "removed"
    assert not tree.exists()


def test_keep_dirty(env):
    tree = _tree(env, "bbbb", env["base"])
    (tree / "scratch.txt").write_text("untracked\n")
    os.utime(tree, (OLD, OLD))
    v = _verdict(env, tree)
    assert not v.remove
    assert v.reason.startswith("dirty")


def test_keep_recent(env):
    tree = _tree(env, "cccc", env["base"], old=False)
    v = _verdict(env, tree)
    assert not v.remove
    assert v.reason.startswith("recent")


def test_keep_referenced_by_running_run(env):
    tree = _tree(env, "dddd", env["base"])
    run = env["runs"] / "live-dddd"
    run.mkdir()
    (run / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": 0,
                "updated": time.time(),
                "base": {"commit": env["base"], "path": str(tree)},
                "cand": {"commit": env["base"]},
                "args": {"base": env["base"][:8], "cand": env["base"][:8]},
            }
        )
    )
    v = _verdict(env, tree)
    assert not v.remove
    assert v.reason == "referenced by active run live-dddd"


def test_stale_running_run_does_not_protect(env):
    # status running but not updated for days: the pid may be reused, so the run is not active
    tree = _tree(env, "kkkk", env["base"])
    run = env["runs"] / "stale-kkkk"
    run.mkdir()
    (run / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": os.getpid(),
                "updated": time.time() - 200 * 3600,
                "base": {"commit": env["base"]},
                "cand": {"commit": env["base"]},
            }
        )
    )
    assert _verdict(env, tree).remove


def test_finished_run_with_verdict_does_not_protect(env):
    tree = _tree(env, "eeee", env["base"])
    run = env["runs"] / "done-eeee"
    run.mkdir()
    (run / "state.json").write_text(
        json.dumps(
            {
                "status": "finished",
                "pid": 0,
                "updated": time.time(),
                "base": {"commit": env["base"]},
                "cand": {"commit": env["base"]},
            }
        )
    )
    (run / "verdict.json").write_text("{}")
    assert _verdict(env, tree).remove


def test_keep_unreachable_commit(env):
    # a commit that only a detached worktree points at: not in main, no branch or tag contains it
    tree = _tree(env, "ffff", env["base"])
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-q", "-m", "orphan")
    head = _git(tree, "rev-parse", "HEAD")
    os.utime(tree, (OLD, OLD))
    v = _verdict(env, tree)
    assert not v.remove
    assert v.head == head
    assert v.reason.startswith("HEAD not in main")


def test_remove_branch_reachable_commit(env):
    tree = _tree(env, "gggg", env["base"])
    (tree / "c.txt").write_text("c\n")
    _git(tree, "add", "c.txt")
    _git(tree, "commit", "-q", "-m", "feature")
    head = _git(tree, "rev-parse", "HEAD")
    _git(env["repo"], "branch", "feature", head)
    os.utime(tree, (OLD, OLD))
    assert _verdict(env, tree).remove


def test_ignored_files_only_need_force(env):
    (env["repo"] / ".gitignore").write_text("*.venv_marker\n")
    _git(env["repo"], "add", ".gitignore")
    _git(env["repo"], "commit", "-q", "-m", "ignore")
    head = _git(env["repo"], "rev-parse", "HEAD")
    tree = _tree(env, "hhhh", head)
    (tree / "x.venv_marker").write_text("ignored\n")
    os.utime(tree, (OLD, OLD))
    v = _verdict(env, tree)
    assert v.remove, v.reason
    assert gc.remove_tree(env["repo"], tree).startswith("removed")
    assert not tree.exists()


def test_keep_in_use_by_process(env):
    tree = _tree(env, "iiii", env["base"])
    v = _verdict(env, tree, procs=[f"python -m something {tree}/python"])
    assert not v.remove
    assert v.reason == "in use by a live process"


def test_unreadable_recent_run_state_fails_closed(env):
    run = env["runs"] / "broken"
    run.mkdir()
    (run / "state.json").write_text("{not json")
    with pytest.raises(gc.GcError):
        gc.protecting_runs(env["runs"], time.time())


def test_census_requires_ancestor_of_main(env):
    tree = _tree(env, "jjjj", env["base"], old=False)
    assert _verdict(env, tree, census=True).remove
    (tree / "d.txt").write_text("d\n")
    _git(tree, "add", "d.txt")
    _git(tree, "commit", "-q", "-m", "not in main")
    _git(env["repo"], "branch", "-q", "-f", "keepme", _git(tree, "rev-parse", "HEAD"))
    v = _verdict(env, tree, census=True)
    assert not v.remove
    assert v.reason == "HEAD is not an ancestor of main"


def test_a_tree_named_by_a_pin_file_is_never_removed(tmp_path, monkeypatch):
    """2026-11-08: the census pass removed the fullcmp pinned worktrees (clean, merged into main)."""
    tree = tmp_path / "wt-fullcmp-yunshu"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: x")
    pin = tmp_path / "competitors" / "x" / "yunshu_pin.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"worktree": str(tree)}))
    monkeypatch.setattr(
        gc, "PIN_GLOBS", (str(tmp_path / "competitors" / "*" / "*_pin.json"),)
    )
    v = gc.classify(
        tree, tmp_path, now=1e12, min_age_h=0, refs=[], procs=[], census=True
    )
    assert not v.remove and "pinned by" in v.reason
