"""CPU tests for scripts/dev/agentbench: the job plan and the fail-closed verdict."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _ab():
    path = str(ROOT / "scripts/dev/agentbench")
    loader = importlib.machinery.SourceFileLoader("agentbench_mod", path)
    spec = importlib.util.spec_from_loader("agentbench_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def row(agent="claude", task="t1", rep=1, passed=True, **kw):
    base = dict(
        type="run",
        agent=agent,
        task=task,
        repeat=rep,
        passed=passed,
        prompt_tokens=1000,
        cached_tokens=900,
        api_errors=0,
        other_http_errors=0,
        malformed_tool_calls=0,
        leaked_tool_markup=0,
        timed_out=False,
        agent_exit=0,
        requests=5,
        max_prompt_tokens=800,
        wall_s=60.0,
        server_peak_gib=30.0,
    )
    base.update(kw)
    return base


def test_plan_is_one_low_priority_job_per_cell(tmp_path):
    ab = _ab()
    jobs = ab.plan(
        ["claude", "codex"], ["a", "b"], 2, tmp_path, Path("/t/python"), "apicov", -1
    )
    assert len(jobs) == 8
    j = jobs[0]
    assert j["label"].startswith("apicov-claude-a-r1")
    assert "--priority=-1" in j["submit"] and "--device" in j["submit"]
    assert j["env"]["AGENTIC_YUNSHU_SRC"] == "/t/python"
    assert (j["env"]["AGENTIC_PORT_LO"], j["env"]["AGENTIC_PORT_HI"]) == (
        "18994",
        "18996",
    )
    assert j["cmd"][j["cmd"].index("--tasks") + 1] == "a"
    assert j["cmd"][j["cmd"].index("--agent") + 1] == "claude"
    assert "--serve" in j["cmd"] and "--timeout-min" in j["cmd"]
    # queue timeout above the per-run timeout, so a slow run is a failed run, not a killed job
    tmo = int(j["submit"][j["submit"].index("--timeout") + 1])
    assert tmo > ab.RUN_TIMEOUT_MIN
    assert len({x["label"] for x in jobs}) == 8


def test_verdict_fails_closed(tmp_path):
    ab = _ab()
    planned = [{"agent": "claude", "task": t, "repeat": 1} for t in ("a", "b")]
    ok = [row(task="a"), row(task="b")]
    v = ab.judge(planned, ok, [])
    assert v["verdict"] == "PASS" and v["current"]["claude"]["cache_hit"] == 0.9
    assert ab.judge(planned, ok[:1], [])["verdict"] == "FAIL"  # a missing run
    for field in ("api_errors", "malformed_tool_calls", "leaked_tool_markup"):
        bad = [row(task="a"), row(task="b", **{field: 1})]
        v = ab.judge(planned, bad, [])
        assert v["verdict"] == "FAIL" and v["problems"][0].startswith("claude:")
    assert ab.judge([], [], [])["verdict"] == "PASS"  # nothing planned, nothing wrong


def test_runs_that_sent_no_request_fail_the_verdict():
    # 2026-11-01 d6dea8f: 60 rows, 0 requests, agent_exit 1 -- judged PASS without a baseline
    ab = _ab()
    planned = [{"agent": "claude", "task": t, "repeat": 1} for t in ("a", "b")]
    dead = [row(task="a", requests=0, passed=False, agent_exit=1), row(task="b")]
    v = ab.judge(planned, dead, [])
    assert v["verdict"] == "FAIL"
    assert v["problems"] == [
        "claude: 1 runs sent no request to the engine (agent launch or harness failure)"
    ]


def test_regression_against_baseline_uses_the_wilson_lower_bound():
    ab = _ab()
    planned = [{"agent": "claude", "task": f"t{i}", "repeat": 1} for i in range(10)]
    base = [
        row(task=f"t{i}", passed=True) for i in range(15)
    ]  # 15/15: lower bound ~0.80
    good = [row(task=f"t{i}", passed=i != 0) for i in range(10)]  # 9/10 = 90%
    assert ab.judge(planned, good, base)["verdict"] == "PASS"
    worse = [row(task=f"t{i}", passed=i < 5) for i in range(10)]  # 50%
    v = ab.judge(planned, worse, base)
    assert v["verdict"] == "REGRESSION" and "claude" in v["regressions"][0]
    # a baseline with fewer than 5 runs says nothing
    assert ab.judge(planned, worse, base[:3])["verdict"] == "PASS"


def test_load_rows_skips_garbage_and_meta(tmp_path):
    ab = _ab()
    f = tmp_path / "x.jsonl"
    f.write_text(
        json.dumps({"type": "meta"}) + "\nnot json\n" + json.dumps(row()) + "\n"
    )
    rows, metas = ab.load_rows([str(f)])
    assert len(rows) == 1 and len(metas) == 1


def test_dry_run_plans_every_agent_and_task(capsys):
    ab = _ab()
    assert (
        ab.main(
            [
                "--dry-run",
                "--tasks",
                "cli-add-flag,shell-report",  # built-in tasks: polyglot ones need the downloaded cache
                "--agents",
                "codex",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert [j["task"] for j in out] == ["cli-add-flag", "shell-report"]


def test_repeat_jobs_tell_the_child_its_repeat_number(tmp_path):
    ab = _ab()
    jobs = ab.plan(["claude"], ["a"], 2, tmp_path, Path("/t/python"), "x", -1)
    starts = [j["cmd"][j["cmd"].index("--repeat-start") + 1] for j in jobs]
    assert starts == ["1", "2"]
    assert all(j["cmd"][j["cmd"].index("--repeat") + 1] == "1" for j in jobs)
    planned = [dict(j, out=j["out"]) for j in jobs]
    rows = [row(task="a", rep=1), row(task="a", rep=2)]
    assert ab.judge(planned, rows, [])["verdict"] == "PASS"


def test_child_repeat_start_numbers_its_runs():
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "scripts/research/agentic"))
    import run_agentic

    assert list(
        run_agentic.repeat_numbers(SimpleNamespace(repeat=1, repeat_start=3))
    ) == [3]
    assert list(run_agentic.repeat_numbers(SimpleNamespace(repeat=2))) == [1, 2]


def test_budget_never_skips_a_jobs_first_run():
    # a one-task job with --budget-min 1 whose server start-up was paused past the budget ran nothing
    import sys

    sys.path.insert(0, str(ROOT / "scripts/research/agentic"))
    import run_agentic

    assert run_agentic.starts_another(0, 600.0, 1)
    assert run_agentic.starts_another(1, 30.0, 1)
    assert not run_agentic.starts_another(1, 61.0, 1)


def _finish_fixture(ab, tmp_path, monkeypatch, wait_rc):
    from types import SimpleNamespace

    jobs = ab.plan(["claude"], ["a"], 1, tmp_path, Path("/t/python"), "x", -1)
    jobs[0]["id"] = "job1"
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "sha": "abcdef0",
                "ref": "main",
                "jobs": [{**j, "out": str(j["out"])} for j in jobs],
            }
        )
    )
    jobs[0]["out"].write_text(json.dumps(row(task="a")) + "\n")
    asked = []

    def fake_run(cmd, **kw):
        asked.append(cmd)
        return SimpleNamespace(returncode=wait_rc)

    monkeypatch.setattr(ab.subprocess, "run", fake_run)
    return asked


def test_failed_queue_job_fails_the_verdict(tmp_path, monkeypatch):
    ab = _ab()
    _finish_fixture(ab, tmp_path, monkeypatch, wait_rc=1)
    assert ab.finish(tmp_path, ["job1"], 10, []) == 1
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "FAIL"


def test_collect_checks_the_saved_queue_ids(tmp_path, monkeypatch):
    ab = _ab()
    asked = _finish_fixture(ab, tmp_path, monkeypatch, wait_rc=1)
    assert ab.finish(tmp_path, [], 10, []) == 1
    assert "job1" in asked[0]
    asked.clear()
    monkeypatch.setattr(
        ab.subprocess,
        "run",
        lambda cmd, **kw: __import__("types").SimpleNamespace(returncode=0),
    )
    assert ab.finish(tmp_path, [], 10, []) == 0


def _stub_gpuq(tmp_path, rcs):
    """An executable stand-in for gpuq: `wait` returns the next rc of rcs (the last one repeats)."""
    d = tmp_path / "gpuq-stub"
    d.mkdir()
    (d / "seq.json").write_text(json.dumps(rcs))
    exe = d / "gpuq"
    exe.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "d = Path(__file__).resolve().parent\n"
        "seq = json.loads((d / 'seq.json').read_text())\n"
        "n = int((d / 'n.txt').read_text()) if (d / 'n.txt').exists() else 0\n"
        "(d / 'n.txt').write_text(str(n + 1))\n"
        "with (d / 'calls.txt').open('a') as f:\n"
        "    f.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.exit(seq[min(n, len(seq) - 1)])\n"
    )
    exe.chmod(0o755)
    return exe, d


def _run_dir(tmp_path, rows_by_job, ids=("j1", "j2")):
    """A run dir with plan.json and one JSONL per job (rows_by_job[i] is the list for job i)."""
    outdir = tmp_path / "run"
    outdir.mkdir()
    jobs = []
    for i, (jid, rows) in enumerate(zip(ids, rows_by_job, strict=True)):
        out = outdir / f"claude-t{i}-r1.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        jobs.append(
            {
                "name": f"claude-t{i}-r1",
                "agent": "claude",
                "task": f"t{i}",
                "repeat": 1,
                "id": jid,
                "out": str(out),
            }
        )
    (outdir / "plan.json").write_text(
        json.dumps({"sha": "abc", "ref": "main", "jobs": jobs})
    )
    return outdir


def test_watch_collect_waits_for_final_then_writes_a_pass(tmp_path, monkeypatch):
    ab = _ab()
    exe, stub_dir = _stub_gpuq(
        tmp_path, [2, 2, 0]
    )  # two polls still pending, then all final
    monkeypatch.setattr(ab, "GPUQ", str(exe))
    outdir = _run_dir(tmp_path, [[row(task="t0")], [row(task="t1")]])
    assert ab.watch_collect(outdir, [], 60, poll_s=5) == 0
    v = json.loads((outdir / "verdict.json").read_text())
    assert v["verdict"] == "PASS" and v["wait_rc"] == 0
    calls = (stub_dir / "calls.txt").read_text().splitlines()
    # three polls on the job ids, then the judge's own final check (finish waits 0 seconds)
    assert [c.split()[:2] for c in calls] == [["wait", "--max-seconds"]] * 4
    assert calls[0].split()[2] == "5" and calls[-1].split()[2] == "0"
    assert calls[0].split()[3:] == ["j1", "j2"]


def test_watch_collect_fails_when_a_job_failed_even_with_all_rows(
    tmp_path, monkeypatch
):
    ab = _ab()
    exe, _ = _stub_gpuq(tmp_path, [1])  # gpuq: a job failed (final, rc 1)
    monkeypatch.setattr(ab, "GPUQ", str(exe))
    outdir = _run_dir(tmp_path, [[row(task="t0")], [row(task="t1")]])
    assert ab.watch_collect(outdir, [], 60, poll_s=5) == 1
    v = json.loads((outdir / "verdict.json").read_text())
    assert v["verdict"] == "FAIL" and v["wait_rc"] == 1
    assert any("gpuq wait rc=1" in p for p in v["problems"])


def test_watch_collect_fails_closed_on_a_missing_result(tmp_path, monkeypatch):
    ab = _ab()
    exe, _ = _stub_gpuq(tmp_path, [0])  # every job done, but one job wrote no run row
    monkeypatch.setattr(ab, "GPUQ", str(exe))
    outdir = _run_dir(tmp_path, [[row(task="t0")], []])
    assert ab.watch_collect(outdir, [], 60, poll_s=5) == 1
    v = json.loads((outdir / "verdict.json").read_text())
    assert v["verdict"] == "FAIL"
    assert "missing run claude/t1/r1" in v["problems"]


def test_watch_collect_gives_up_on_unfinished_jobs_as_a_failure(tmp_path, monkeypatch):
    ab = _ab()
    exe, _ = _stub_gpuq(tmp_path, [2])  # always pending
    monkeypatch.setattr(ab, "GPUQ", str(exe))
    outdir = _run_dir(tmp_path, [[row(task="t0")], [row(task="t1")]])
    assert ab.watch_collect(outdir, [], 60, poll_s=5, max_wait_s=0) == 1
    v = json.loads((outdir / "verdict.json").read_text())
    assert v["verdict"] == "FAIL" and v["wait_rc"] == 2
    assert any("unfinished" in p for p in v["problems"])


def test_watch_collect_without_a_plan_writes_a_failing_verdict(tmp_path, monkeypatch):
    ab = _ab()
    exe, stub_dir = _stub_gpuq(tmp_path, [0])
    monkeypatch.setattr(ab, "GPUQ", str(exe))
    outdir = tmp_path / "run"
    outdir.mkdir()
    assert ab.watch_collect(outdir, [], 60) == 1
    v = json.loads((outdir / "verdict.json").read_text())
    assert v["verdict"] == "FAIL" and "no plan.json" in v["problems"][0]
    assert not (stub_dir / "calls.txt").exists()  # nothing was waited on


def test_detached_collector_runs_the_watch_mode_on_the_same_run_dir(tmp_path):
    ab = _ab()
    argv = ab.watcher_argv(tmp_path / "run", "none", 900)
    assert argv[0] == sys.executable
    assert argv[1].endswith("scripts/dev/agentbench")
    assert argv[2:] == [
        "--collect",
        str(tmp_path / "run"),
        "--watch-collect",
        "--baseline",
        "none",
        "--max-seconds",
        "900",
    ]
