#!/usr/bin/env python3
"""Regenerate the automatic sections of docs/research/INDEX.md.

The index is hand-written except for blocks between marker pairs:

    <!-- auto:lines -->  ...  <!-- /auto:lines -->    one row per line (worktree branch)
    <!-- auto:merged --> ...  <!-- /auto:merged -->   merges into main since a base tag

Per line: branch, ahead/behind main, last commit age and subject, the head of the worker report
(`<codex>/<branch>_last.md`, `sonnet-<branch>_last.md`, `codex-<branch>_last.md`), READY TO MERGE
status, whether a `codex exec` worker is alive for the worktree, and queued/running gpuq jobs whose
label starts with `<branch>-`.

No network, no GPU, a few dozen git calls (well under 5 s). Idempotent: the same inputs give the same
bytes (the stamp is rounded to the minute). The lead runs it on every watchdog event and hourly; every
line also runs it before committing so its own row is current.

    scripts/dev/research_index.py [--index PATH] [--codex DIR] [--jobs DIR] [--base-tag v0.1.4]
                                  [--main BRANCH] [--check-age]   # --check-age: exit 1 if stamp > 60 min
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODEX = Path("/Volumes/P5Plus/yunshu-build/codex")
JOBS = Path("/Volumes/P5Plus/yunshu-gpuq/jobs")
STAMP_RE = re.compile(r"<!-- auto:stamp (\d+) -->")
READY_RE = re.compile(r"READY TO MERGE\s+([0-9a-f]{7,40})")
SKIP_BRANCHES = {"main"}


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def main_root(repo: Path) -> Path:
    common = _git(
        repo, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    return Path(common).parent if common else repo


def worktrees(repo: Path) -> list[tuple[str, Path]]:
    """(branch, path) of every non-detached worktree except the main branch."""
    out, path, rows = _git(repo, "worktree", "list", "--porcelain"), None, []
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = Path(line[9:])
        elif line.startswith("branch refs/heads/") and path is not None:
            name = line[len("branch refs/heads/") :]
            if name not in SKIP_BRANCHES:
                rows.append((name, path))
    return sorted(rows)


def _age(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec // 3600}h"
    return f"{sec // 86400}d"


def report_for(branch: str, codex: Path) -> Path | None:
    stem = branch.split("/")[-1]
    for name in (f"sonnet-{stem}_last.md", f"{stem}_last.md", f"codex-{stem}_last.md"):
        p = codex / name
        if p.exists():
            return p
    return None


def report_head(path: Path) -> tuple[str, str]:
    """(first meaningful line, READY sha or '')."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "", ""
    ready = READY_RE.findall(text)
    head = ""
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s and not s.startswith(("---", "```")):
            head = s
            break
    head = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", head)  # links would break when cut
    head = re.sub(r"[`*|\[\]]", "", head)
    return head[:110], (ready[-1][:8] if ready else "")


def live_workers(codex: Path, ps_out: str | None = None) -> list[str]:
    """Command lines of running `codex exec` workers (plus pids registered in workers.json)."""
    if ps_out is None:
        r = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True)
        ps_out = r.stdout
    return [ln for ln in ps_out.splitlines() if "codex" in ln and " exec " in ln]


def gpuq_counts(jobs_dir: Path) -> list[tuple[str, str]]:
    """(label, state) of every pending or running gpuq job."""
    rows: list[tuple[str, str]] = []
    if not jobs_dir.exists():
        return rows
    for p in jobs_dir.glob("*.json"):
        try:
            j = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        st = j.get("state")
        if st not in ("pending", "running"):
            continue
        rows.append((j.get("label") or "", st))
    return rows


def line_rows(
    repo: Path, codex: Path, jobs_dir: Path, main: str, now: float
) -> list[str]:
    workers = live_workers(codex)
    jobs = gpuq_counts(jobs_dir)
    rows = []
    for branch, path in worktrees(repo):
        lr = _git(
            repo, "rev-list", "--left-right", "--count", f"{main}...{branch}"
        ).split()
        behind, ahead = (lr + ["?", "?"])[:2]
        info = (
            _git(repo, "log", "-1", "--format=%ct%x09%s", branch)
            .rstrip("\n")
            .split("\t", 1)
        )
        when = _age(now - float(info[0])) if info and info[0].isdigit() else "?"
        subj = (info[1] if len(info) > 1 else "")[:80].replace("|", "/")
        rep = report_for(branch, codex)
        head, ready = report_head(rep) if rep else ("", "")
        rep_cell = f"[{rep.name}]({rep}): {head}".replace("|", "/") if rep else "-"
        live = (
            "live" if any(str(path) in w or f"-C {path}" in w for w in workers) else "-"
        )
        stem = branch.split("/")[-1]
        mine = [st for lb, st in jobs if lb.startswith(stem + "-")]
        pend, run = mine.count("pending"), mine.count("running")
        q = f"{run}r/{pend}q" if (pend or run) else "-"
        head_sha = _git(repo, "rev-parse", branch).strip()
        if ahead == "0":
            status = "merged"
        elif ready:
            status = (
                f"READY {ready}"
                if head_sha.startswith(ready)
                else f"READY(old) {ready}"
            )
        else:
            status = "open"
        rows.append(
            f"| {branch} | +{ahead}/-{behind} | {status} | {live} | {q} | {when} {subj} | {rep_cell} |"
        )
    return rows


def parity_verdict(repo: Path) -> str:
    """Read the board without converting missing/stale evidence into a success."""
    path = repo / "docs/research/parityboard/board.json"
    try:
        board = json.loads(path.read_text())
        n, total, missing = board["parity"], board["total"], board["missing"]
        if (
            type(n) is not int
            or type(total) is not int
            or not 0 <= n <= total
            or total == 0
            or not isinstance(missing, list)
        ):
            raise ValueError("invalid counts")
        return f"parity: {n}/{total} items, missing: {len(missing)} (board snapshot; rerun scripts/dev/parityboard to refresh)"
    except (OSError, ValueError, KeyError, TypeError):
        return "parity: unknown, missing: board unavailable or invalid"


def lines_block(repo: Path, codex: Path, jobs_dir: Path, main: str, now: float) -> str:
    stamp = int(now // 60 * 60)
    iso = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")
    head = [
        f"<!-- auto:stamp {stamp} -->",
        f"自動產生 {iso}（`scripts/dev/research_index.py`；不要手改此區）。"
        f"ahead/behind 相對 {main}；q = gpuq 佇列（running/queued）。",
        "",
        parity_verdict(repo),
        "",
        "| 線 (branch) | +ahead/-behind | 狀態 | worker | gpuq | 最後 commit | 報告 |",
        "|---|---|---|---|---|---|---|",
    ]
    return "\n".join(head + line_rows(repo, codex, jobs_dir, main, now))


def merged_block(repo: Path, base_tag: str, main: str) -> str:
    out = _git(
        repo,
        "log",
        "--first-parent",
        "--merges",
        "--format=%h%x09%cs%x09%s",
        f"{base_tag}..{main}",
    )
    rows = ["| commit | 日期 | 合併內容 |", "|---|---|---|"]
    for ln in out.splitlines():
        h, d, s = (ln.split("\t", 2) + ["", ""])[:3]
        rows.append(f"| {h} | {d} | {s[:150].replace('|', '/')} |")
    return "\n".join([f"自 {base_tag} 起合併進 {main}：{len(rows) - 2} 筆", "", *rows])


def replace_block(text: str, name: str, body: str) -> str:
    a, b = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
    i, j = text.find(a), text.find(b)
    if i < 0 or j < i:
        raise SystemExit(f"research_index: marker pair auto:{name} missing in index")
    return text[: i + len(a)] + "\n" + body + "\n" + text[j:]


def stamp_age_min(text: str, now: float) -> float | None:
    m = STAMP_RE.search(text)
    return (now - int(m.group(1))) / 60 if m else None


def regenerate(
    index: Path,
    repo: Path,
    codex: Path = CODEX,
    jobs_dir: Path = JOBS,
    base_tag: str = "v0.1.4",
    main: str = "main",
    now: float | None = None,
) -> bool:
    """Rewrite the auto blocks; returns True when the file changed."""
    now = time.time() if now is None else now
    text = index.read_text()
    new = text
    if "<!-- auto:lines -->" in new:
        new = replace_block(new, "lines", lines_block(repo, codex, jobs_dir, main, now))
    if "<!-- auto:merged -->" in new:
        new = replace_block(new, "merged", merged_block(repo, base_tag, main))
    if new != text:
        index.write_text(new)
    return new != text


def main_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--codex", type=Path, default=CODEX)
    ap.add_argument("--jobs", type=Path, default=JOBS)
    ap.add_argument("--base-tag", default="v0.1.4")
    ap.add_argument("--main", default="main")
    ap.add_argument("--check-age", action="store_true")
    a = ap.parse_args(argv)
    root = main_root(a.repo)
    index = a.index or root / "docs" / "research" / "INDEX.md"
    if not index.exists():
        print(f"research_index: {index} not found", file=sys.stderr)
        return 2
    if a.check_age:
        age = stamp_age_min(index.read_text(), time.time())
        print("no stamp" if age is None else f"auto section is {age:.0f} min old")
        return 0 if age is not None and age <= 60 else 1
    changed = regenerate(index, root, a.codex, a.jobs, a.base_tag, a.main)
    print(f"{'updated' if changed else 'unchanged'} {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
