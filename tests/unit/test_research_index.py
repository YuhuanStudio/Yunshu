"""research_index.py regenerates INDEX.md auto blocks from a fake repo; watchdog index-stale hook."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
import research_index as ri  # noqa: E402

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]


def run(cwd: Path, *a: str) -> None:
    subprocess.run([*GIT, *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def fake(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("a")
    run(repo, "add", ".")
    run(repo, "commit", "-m", "base")
    run(repo, "tag", "v0.1.4")
    wt = tmp_path / "wt-foo"
    run(repo, "worktree", "add", "-b", "foo", str(wt))
    (wt / "b.txt").write_text("b")
    run(wt, "add", ".")
    run(wt, "commit", "-m", "foo work | pipe")
    run(repo, "merge", "--no-ff", "-m", "Merge foo: thing", "foo")
    wt2 = tmp_path / "wt-bar"
    run(repo, "worktree", "add", "-b", "bar", str(wt2))
    (wt2 / "c.txt").write_text("c")
    run(wt2, "add", ".")
    run(wt2, "commit", "-m", "bar wip")
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "sonnet-bar_last.md").write_text(
        "# Bar report\nREADY TO MERGE abcdef1234\n"
    )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for i, st in enumerate(["running", "pending", "done"]):
        (jobs / f"{i}.json").write_text(
            json.dumps({"id": str(i), "state": st, "label": "bar-x"})
        )
    index = tmp_path / "INDEX.md"
    index.write_text(
        "# T\nhand\n<!-- auto:lines -->\nold\n<!-- /auto:lines -->\nmid\n"
        "<!-- auto:merged -->\nold\n<!-- /auto:merged -->\nend\n"
    )
    return repo, codex, jobs, index


def test_regenerate_rows_and_idempotent(fake) -> None:
    repo, codex, jobs, index = fake
    assert ri.regenerate(index, repo, codex, jobs, now=1_800_000_000)
    text = index.read_text()
    assert (
        "hand" in text and "mid" in text and "end" in text
    )  # hand-written text untouched
    assert "| bar | +1/-0 | READY(old) abcdef12 |" in text
    assert "1r/1q" in text
    assert "Bar report" in text
    assert (
        "| foo | +0/-" in text and "merged" in text
    )  # merged branch has nothing ahead
    assert "foo work / pipe" in text  # pipes escaped
    assert "Merge foo: thing" in text and "1 筆" in text
    assert not ri.regenerate(
        index, repo, codex, jobs, now=1_800_000_000 + 30
    )  # same minute
    assert ri.stamp_age_min(text, 1_800_000_000 + 61 * 60) > 60


def test_report_head_strips_links(tmp_path: Path) -> None:
    f = tmp_path / "r.md"
    f.write_text("\n# 審查完成：[完整 findings](/x/y/z.md) `code` a|b\n")
    head, ready = ri.report_head(f)
    assert head == "審查完成：完整 findings code ab" and ready == ""


def test_missing_marker_fails(fake) -> None:
    repo, codex, jobs, index = fake
    with pytest.raises(SystemExit):
        ri.replace_block("no markers", "lines", "x")


def test_cli_check_age(fake) -> None:
    repo, codex, jobs, index = fake
    assert ri.main_cli(["--index", str(index), "--repo", str(repo), "--check-age"]) == 1
    ri.regenerate(index, repo, codex, jobs, now=__import__("time").time())
    assert ri.main_cli(["--index", str(index), "--repo", str(repo), "--check-age"]) == 0


WATCHDOG = Path("/Volumes/P5Plus/yunshu-build/codex/watchdog.py")


@pytest.mark.skipif(not WATCHDOG.exists(), reason="lead watchdog not on this machine")
def test_watchdog_index_stale(tmp_path: Path, monkeypatch) -> None:
    index = tmp_path / "INDEX.md"
    monkeypatch.setenv("WATCHDOG_BASE", str(tmp_path))
    monkeypatch.setenv("WATCHDOG_INDEX", str(index))
    spec = importlib.util.spec_from_file_location("wd_under_test", WATCHDOG)
    wd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wd)
    now = 1_800_000_000
    assert wd._index_stale(now) == []  # no index: silent
    index.write_text(f"<!-- auto:stamp {now - 30 * 60} -->")
    assert wd._index_stale(now) == []  # fresh
    index.write_text(f"<!-- auto:stamp {now - 90 * 60} -->")
    ev = wd._index_stale(now)
    assert ev and ev[0].startswith("index-stale")
    assert wd._index_stale(now + 60) == []  # at most hourly
    assert wd._index_stale(now + 3700)  # again after an hour
    assert os.fspath(tmp_path) in str(wd.BASE)


def test_parity_board_invalid_and_present(tmp_path):
    assert "unknown" in ri.parity_verdict(tmp_path)
    p = tmp_path / "docs/research/parityboard/board.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"parity": 2, "total": 7, "missing": ["x"]}))
    assert "parity: 2/7 items, missing: 1" in ri.parity_verdict(tmp_path)
    p.write_text(json.dumps({"parity": 8, "total": 7, "missing": []}))
    assert "unknown" in ri.parity_verdict(tmp_path)


def test_json_output_matches_rows(fake, tmp_path: Path, capsys) -> None:
    repo, codex, jobs, index = fake
    out = tmp_path / "lines.json"
    before = index.read_text()
    argv = ["--repo", str(repo), "--codex", str(codex), "--jobs", str(jobs)]
    assert ri.main_cli([*argv, "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    bar = next(d for d in data["lines"] if d["branch"] == "bar")
    assert bar["status"] == "ready-old" and bar["ready_sha"] == "abcdef12"
    assert (bar["gpuq_running"], bar["gpuq_pending"], bar["ahead"]) == (1, 1, "1")
    assert bar["report_path"].endswith("sonnet-bar_last.md")
    foo = next(d for d in data["lines"] if d["branch"] == "foo")
    assert foo["status"] == "merged"
    assert index.read_text() == before  # --json never writes the index
    assert ri.main_cli([*argv, "--json", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["lines"]
