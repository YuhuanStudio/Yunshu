"""CPU tests for scripts/dev/agentbench: the job plan and the fail-closed verdict."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
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
