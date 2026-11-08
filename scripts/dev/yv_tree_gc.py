#!/usr/bin/env python3
"""Garbage-collect detached verify worktrees (and, with --census-root, other detached worktrees).

yv (scripts/verify/core.py) pins one detached git worktree per verified commit under
<verify root>/trees/<commit12> and never removes them. This tool removes a worktree only when ALL hold:

  - the tree directory's mtime is older than --min-age-hours (default 48);
  - `git status --porcelain` in the tree is empty (ignored files such as venvs do not count);
  - no run under <verify root>/runs references the tree (its state.json names the commit, the path or a
    ref given on the command line) while that run is active: updated within 24 h and either still
    running or without a verdict.json (a pid is not used: pids get reused, and a stale run is not active);
  - HEAD is an ancestor of main, or reachable from a local branch or tag;
  - no live process command line mentions the tree path.

If any state cannot be read (a run state.json that is unreadable, or `ps` failing), the tool refuses to
apply and exits 2: an unknown run might be using a tree.

Default is a dry run. --apply removes with `git worktree remove`; --force is used only when the tree holds
ignored files alone (logged).

--census-root DIR (repeatable) applies the same checks to the other worktrees under DIR (outside the verify
trees), without the age rule and with a strict "HEAD is an ancestor of main" rule. --exclude NAME
(repeatable) keeps worktrees whose directory name is NAME.

    scripts/dev/yv_tree_gc.py [--verify-root DIR] [--min-age-hours 48] [--apply]
                              [--census-root DIR --exclude wt-pb2-generic ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_VERIFY_ROOT = Path(
    os.environ.get("YV_ROOT", "/Volumes/P5Plus/yunshu-build/verify")
)
RUN_WINDOW_S = 24 * 3600
MAIN = "main"


class GcError(RuntimeError):
    """State that cannot be read: the tool must not guess (fail closed)."""


def git(cwd: Path, *args: str) -> tuple[int, str, str]:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


@dataclass
class RunRef:
    run: str
    commits: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)


def protecting_runs(
    runs: Path, now: float, window_s: float = RUN_WINDOW_S
) -> list[RunRef]:
    """Runs whose state keeps a tree alive. Raises GcError on an unreadable recent state."""
    out: list[RunRef] = []
    if not runs.is_dir():
        return out
    for sj in sorted(runs.glob("*/state.json")):
        try:
            st = json.loads(sj.read_text())
        except (OSError, ValueError) as e:
            if now - sj.stat().st_mtime < window_s:
                raise GcError(f"cannot read run state {sj}: {e}") from e
            continue
        if not isinstance(st, dict):
            continue
        updated = float(st.get("updated") or sj.stat().st_mtime)
        has_verdict = (sj.parent / "verdict.json").exists()
        running = st.get("status") == "running"
        active = (running or not has_verdict) and now - updated < window_s
        if not active:
            continue
        ref = RunRef(run=sj.parent.name)
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        for side in ("base", "cand"):
            s = st.get(side) if isinstance(st.get(side), dict) else {}
            if s.get("commit"):
                ref.commits.add(str(s["commit"]).lower())
            if s.get("path"):
                ref.paths.add(os.path.realpath(str(s["path"])))
            for value in (s.get("spec"), args.get(side)):
                if value:
                    ref.refs.add(str(value).lower())
        out.append(ref)
    return out


def referenced_by(head: str, tree: Path, refs: list[RunRef]) -> str | None:
    real = os.path.realpath(tree)
    for r in refs:
        if head in r.commits or real in r.paths:
            return r.run
        if any(len(x) >= 7 and head.startswith(x) for x in r.refs):
            return r.run
    return None


def live_process_lines() -> list[str]:
    r = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True)
    if r.returncode != 0:
        raise GcError(f"ps failed: {r.stderr.strip()[:200]}")
    return r.stdout.splitlines()


def dir_size_kb(path: Path) -> int:
    r = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


@dataclass
class Verdict:
    tree: Path
    remove: bool
    reason: str
    head: str = ""
    size_kb: int = 0


PIN_GLOBS = (
    "/Volumes/P5Plus/yunshu-build/competitors/*/*/*_pin.json",
    "/Volumes/P5Plus/yunshu-build/competitors/*/*_pin.json",
)


_PINNED: dict[Path, str] | None = None  # main() reads the pin files once


def pinned_trees(globs: tuple[str, ...] | None = None) -> dict[Path, str]:
    """Worktrees a comparison pin file names (fullcmp yunshu_pin.json and friends): never removed.
    2026-11-08 the census pass removed wt-fullcmp-{yunshu,pb2,next-engine} (clean, merged) and the
    fullcmp harness then refused every Yunshu cell for a missing pinned tree."""
    import glob
    import json

    out: dict[Path, str] = {}
    for pattern in PIN_GLOBS if globs is None else globs:
        for f in glob.glob(pattern, recursive=True):
            try:
                wt = json.loads(Path(f).read_text()).get("worktree")
            except (OSError, ValueError, AttributeError):
                continue
            if isinstance(wt, str) and wt:
                out[Path(wt).resolve()] = f
    return out


def classify(
    tree: Path,
    repo: Path,
    now: float,
    min_age_h: float,
    refs: list[RunRef],
    procs: list[str],
    census: bool = False,
) -> Verdict:
    def keep(reason: str, head: str = "") -> Verdict:
        return Verdict(tree, False, reason, head)

    if not (tree / ".git").exists():
        return keep("not a worktree")
    pinned = _PINNED if _PINNED is not None else pinned_trees()
    if tree.resolve() in pinned:
        return keep(f"pinned by {pinned[tree.resolve()]}")
    rc, head, _ = git(tree, "rev-parse", "HEAD")
    if rc != 0:
        return keep("HEAD unreadable")
    head = head.strip()
    age_h = (now - tree.stat().st_mtime) / 3600
    if age_h < min_age_h:
        return keep(f"recent ({age_h:.1f} h < {min_age_h:g} h)", head)
    rc, status, _ = git(tree, "status", "--porcelain")
    if rc != 0:
        return keep("git status failed", head)
    if status.strip():
        n = len([x for x in status.splitlines() if x.strip()])
        return keep(f"dirty ({n} changed paths)", head)
    needles = {str(tree), os.path.realpath(tree)}
    if any(n in line for line in procs for n in needles):
        return keep("in use by a live process", head)
    run = referenced_by(head, tree, refs)
    if run:
        return keep(f"referenced by active run {run}", head)
    if census:
        rc, _, _ = git(repo, "merge-base", "--is-ancestor", head, MAIN)
        if rc != 0:
            return keep("HEAD is not an ancestor of main", head)
        why = "clean, HEAD ancestor of main"
    else:
        rc, _, _ = git(repo, "merge-base", "--is-ancestor", head, MAIN)
        if rc != 0:
            rc2, refs_out, _ = git(
                repo,
                "for-each-ref",
                "--contains",
                head,
                "--format=%(refname)",
                "refs/heads",
                "refs/tags",
            )
            if rc2 != 0 or not refs_out.strip():
                return keep(
                    "HEAD not in main and not reachable from a branch or tag", head
                )
            why = "clean, merged or reachable from a branch or tag"
        else:
            why = "clean, HEAD ancestor of main"
    return Verdict(tree, True, f"{why}, idle {age_h:.0f} h", head)


def remove_tree(repo: Path, tree: Path) -> str:
    rc, _, err = git(repo, "worktree", "remove", str(tree))
    if rc == 0:
        return "removed"
    # Porcelain status was empty (no tracked or untracked-non-ignored change), so only ignored
    # files can be blocking the plain remove. Confirm that before forcing.
    rc2, st, _ = git(tree, "status", "--porcelain", "--ignored")
    lines = [x for x in st.splitlines() if x.strip()]
    if rc2 == 0 and all(x.startswith("!!") for x in lines):
        rc3, _, err3 = git(repo, "worktree", "remove", "--force", str(tree))
        if rc3 == 0:
            return "removed with --force (ignored files only)"
        return f"FAILED: {err3.strip()[:200]}"
    return f"FAILED: {err.strip()[:200]}"


def census_worktrees(
    repo: Path, root: Path, verify_trees: Path, exclude: set[str]
) -> list[Path]:
    rc, out, _ = git(repo, "worktree", "list", "--porcelain")
    if rc != 0:
        raise GcError("git worktree list failed")
    paths = [
        Path(line[len("worktree ") :])
        for line in out.splitlines()
        if line.startswith("worktree ")
    ]
    res: list[Path] = []
    for p in paths:
        if p == repo.resolve() or not str(p).startswith(str(root) + os.sep):
            continue
        if str(p).startswith(str(verify_trees) + os.sep):
            continue  # the verify trees are handled by the main pass
        if p.name in exclude:
            continue
        res.append(p)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify-root", type=Path, default=DEFAULT_VERIFY_ROOT)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--min-age-hours", type=float, default=48.0)
    ap.add_argument("--apply", action="store_true", help="remove (default: dry run)")
    ap.add_argument("--census-root", type=Path, action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    verify_trees = args.verify_root / "trees"
    now = time.time()
    try:
        global _PINNED
        _PINNED = pinned_trees()
        refs = protecting_runs(args.verify_root / "runs", now)
        procs = live_process_lines()
        jobs: list[tuple[Path, bool]] = []
        if verify_trees.is_dir():
            jobs += [(p, False) for p in sorted(verify_trees.iterdir()) if p.is_dir()]
        for root in args.census_root:
            for p in census_worktrees(
                repo, root.resolve(), verify_trees.resolve(), set(args.exclude)
            ):
                jobs.append((p, True))
    except GcError as e:
        print(f"yv_tree_gc: refusing to continue: {e}", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY RUN"
    print(
        f"[{mode}] {len(jobs)} worktrees; min age {args.min_age_hours:g} h; "
        f"{len(refs)} active run(s) protect trees"
    )
    removable: list[Verdict] = []
    kept: list[Verdict] = []
    for tree, census in jobs:
        v = classify(
            tree, repo, now, 0.0 if census else args.min_age_hours, refs, procs, census
        )
        (removable if v.remove else kept).append(v)

    total_kb = 0
    freed_kb = 0
    failed = 0
    for v in removable:
        v.size_kb = dir_size_kb(v.tree)
        total_kb += v.size_kb
        if args.apply:
            result = remove_tree(repo, v.tree)
            if result.startswith("FAILED"):
                failed += 1
            else:
                freed_kb += v.size_kb
        else:
            result = "would remove"
        print(
            f"REMOVE {v.tree.name} ({v.size_kb / 1048576:.2f} GB) {v.reason}: {result}"
        )
    for v in kept:
        print(f"KEEP   {v.tree.name} {v.reason}")

    print(
        f"summary: remove {len(removable)} ({total_kb / 1048576:.2f} GB), keep {len(kept)}"
        + (
            f", freed {freed_kb / 1048576:.2f} GB, failed {failed}"
            if args.apply
            else ""
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
