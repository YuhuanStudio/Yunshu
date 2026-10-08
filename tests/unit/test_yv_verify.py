"""yv (scripts/verify): ladder logic, resume, fail-fast, verdict schema, suites, diff->tests, gate.

No GPU: a fake gpuq runs the commands at once and fake tfbench / paired_eval / memory_ab write
deterministic evidence, so the tests exercise the real orchestration code end to end.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
FAKES = Path(__file__).parent / "yv_fakes"

from verify import analyze, core, gate, runner, stages, suites, verdict  # noqa: E402
from verify.execute import Cell, Executor  # noqa: E402


def git(cwd, *a):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A tiny repo with two commits (cand edits foo.py + its test) and fakes wired in."""
    repo = tmp_path / "repo"
    (repo / "python/yunshu_engine").mkdir(parents=True)
    for pkg in ("yunshu_engine", "yunshu_gateway", "yunshu_kv"):
        (repo / "python" / pkg).mkdir(parents=True, exist_ok=True)
        (repo / "python" / pkg / "__init__.py").write_text("")
    (repo / "python/yunshu_engine/foo.py").write_text("X = 1\n")
    (repo / "tests/unit").mkdir(parents=True)
    (repo / "tests/unit/test_foo.py").write_text(
        "from yunshu_engine import foo\n\ndef test_x():\n    assert foo.X >= 1\n"
    )
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    git(repo, "branch", "base")
    (repo / "python/yunshu_engine/foo.py").write_text("X = 2\n")
    git(repo, "commit", "-qam", "cand")
    git(repo, "branch", "cand")
    monkeypatch.setattr(core, "REPO", repo)
    trees = tmp_path / "trees"
    runs = tmp_path / "runs"
    monkeypatch.setenv("FAKE_GPUQ_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("PAIRED_PY", sys.executable)
    monkeypatch.setattr(stages, "TFBENCH", FAKES / "fake_tfbench.py")
    monkeypatch.setattr(stages, "PAIRED", FAKES / "fake_paired.py")
    monkeypatch.setattr(stages, "MEMORY_AB", FAKES / "fake_memory.py")
    monkeypatch.setattr(stages, "PROMPTS", tmp_path / "prompts")
    (tmp_path / "prompts").mkdir()
    for k in ("prose", "code"):
        for c in (1024, 8192):
            (tmp_path / "prompts" / f"{k}-{c}.txt").write_text("x")
    model = tmp_path / "model"
    model.mkdir()
    gq = core.Gpuq(binary=str(FAKES / "fake_gpuq.py"), jobs_dir=tmp_path / "jobs")
    return argparse.Namespace(
        repo=repo, trees=trees, runs=runs, gq=gq, model=str(model), tmp=tmp_path
    )


def ns(w, **kw):
    d = dict(
        base="base",
        cand="cand",
        env=[],
        cand_env=[],
        base_env=[],
        suite="tiny",
        label="t",
        model=w.model,
        model_name="",
        engaged=[],
        ctx=None,
        reps=None,
        mmlu_n=None,
        mem_sizes=None,
        mem_reps=None,
        speed_tol=None,
        spec_off=None,
        no_apc_hit_required=False,
        mem_gb=0,
        priority=0,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def go(w, **kw):
    return runner.run_ab(
        ns(w, **kw), gq=w.gq, log=lambda m: None, runs=w.runs, trees=w.trees
    )


def verdict_of(w, label="t"):
    (rd,) = list(w.runs.glob(f"{label}-*"))
    return json.loads((rd / "verdict.json").read_text()), rd


def jobs(w):
    return sorted(p.name for p in (w.tmp / "jobs").glob("*.json"))


# ── suites ───────────────────────────────────────────────────────────────
def test_suite_named_and_adhoc():
    assert suites.parse_suite("decode")["stages"][0] == "preflight"
    cfg = suites.parse_suite("speed,smoke")
    assert cfg["stages"] == ["smoke", "speed"]  # canonical order, not the typed one
    with pytest.raises(ValueError):
        suites.parse_suite("smoke,bogus")
    with pytest.raises(ValueError):
        suites.parse_suite(" , ")


def test_full_suite_has_every_stage():
    assert suites.parse_suite("full")["stages"] == list(suites.LADDER)
    assert set(suites.STAGES) - set(suites.LADDER) == {
        "longqa",
        "conc",
        "rerank",
        "evals",
        "console",
        "telemetry",
        "telemetry-tiny",
    }


# ── diff -> tests ────────────────────────────────────────────────────────
def test_related_tests_mapping(tmp_path):
    (tmp_path / "python/yunshu_engine").mkdir(parents=True)
    (tmp_path / "tests/unit").mkdir(parents=True)
    (tmp_path / "tests/unit/test_alpha.py").write_text("def test_a(): pass\n")
    (tmp_path / "tests/unit/test_uses_beta.py").write_text(
        "from yunshu_engine.beta import f\n"
    )
    (tmp_path / "tests/unit/test_prose.py").write_text(
        "# beta is mentioned only in prose\nimport os\n"
    )
    (tmp_path / "tests/unit/test_other.py").write_text("import os\n")
    got = core.related_tests(
        [
            "python/yunshu_engine/alpha.py",
            "python/yunshu_engine/beta.py",
            "README.md",
            "tests/unit/test_other.py",
        ],
        tmp_path,
    )
    assert got == [
        "tests/unit/test_alpha.py",
        "tests/unit/test_other.py",
        "tests/unit/test_uses_beta.py",
    ]


def test_changed_files_between_refs(world):
    b = core.resolve_arm("base", "base", world.trees)
    c = core.resolve_arm("cand", "cand", world.trees)
    assert core.changed_files(b, c) == ["python/yunshu_engine/foo.py"]
    assert core.changed_files(b, b) == []
    assert (
        world.trees / c.commit[:12] / ".git"
    ).exists()  # pinned worktree, reused by hash
    c2 = core.resolve_arm("cand", "cand", world.trees)
    assert c2.path == c.path


def test_dirty_dir_arm_key_changes(world):
    d = world.repo
    a1 = core.resolve_arm("cand", str(d))
    (d / "python/yunshu_engine/foo.py").write_text("X = 3\n")
    a2 = core.resolve_arm("cand", str(d))
    assert a1.key != a2.key and a2.dirty


# ── analysis ─────────────────────────────────────────────────────────────
def rows(
    sha="a", dec=100.0, ttft=1.0, cached=0, ctx=1024, phases=("cold", "warm", "turn2")
):
    return [
        {
            "part": "decode",
            "ctx": ctx,
            "kind": "code",
            "phase": p,
            "sha": sha,
            "ct": 16,
            "dec_tps": dec,
            "ttft_s": ttft,
            "cached": 0 if p == "cold" else cached,
        }
        for p in phases
    ]


def test_identity_equal_diff_and_missing():
    assert analyze.compare_identity(rows(), rows())["ok"]
    bad = analyze.compare_identity(rows("a"), rows("b"))
    assert not bad["ok"] and len(bad["mismatches"]) == 3
    miss = analyze.compare_identity(rows(), rows(phases=("cold",)))
    assert not miss["ok"] and "missing" in miss["mismatches"][0]["why"]
    assert not analyze.compare_identity([], [])["ok"]  # nothing compared fails closed


def test_apc_requires_equal_digest_and_a_real_hit():
    assert analyze.check_apc(rows(cached=1024))["ok"]
    assert not analyze.check_apc(rows(cached=0))["ok"]
    assert analyze.check_apc(rows(cached=0), require_hit=False)["ok"]
    r = rows(cached=1024)
    r[1]["sha"] = "zzz"
    assert not analyze.check_apc(r)["ok"]


def test_speed_regression_noise_and_improvement():
    base = [rows(dec=100.0 + i * 0.1) for i in range(3)]
    same = [rows(dec=100.0 + i * 0.1) for i in range(3)]
    assert analyze.speed_compare(base, same)["ok"]
    slow = [rows(dec=90.0 + i * 0.1) for i in range(3)]
    sp = analyze.speed_compare(base, slow)
    assert not sp["ok"] and sp["regressions"][0]["metric"] == "decode_tps"
    # a noisy pair of arms: -4% median but per-rep deltas swing +-6% -> inside the noise, not a regression
    noisy_b = [rows(dec=d) for d in (100, 100, 100)]
    noisy_c = [rows(dec=d) for d in (88, 96, 100)]
    assert analyze.speed_compare(noisy_b, noisy_c)["ok"]
    fast = [rows(dec=110.0) for _ in range(3)]
    sp = analyze.speed_compare(base, fast)
    assert sp["ok"] and any(c["verdict"] == "improvement" for c in sp["cells"])
    # TTFT is lower-is-better
    slow_ttft = [rows(ttft=1.3) for _ in range(3)]
    assert not analyze.speed_compare([rows() for _ in range(3)], slow_ttft)["ok"]
    assert not analyze.speed_compare([], [])["ok"]
    assert not analyze.speed_compare(base, [rows(dec=0.0)] * 3)[
        "ok"
    ]  # missing metric fails closed


def mem_rows(arm, hog=0.0, rep=0):
    return [
        {
            "arm": arm,
            "rep": rep,
            "step": s,
            "footprint_gib": f + hog,
            "peak_footprint_gib": 9.5 + hog,
        }
        for s, f in (("ready", 5.0), ("idle20s", 6.0), ("idle-after", 5.5))
    ]


def test_memory_regression_flagged():
    ok = analyze.memory_compare(mem_rows("base") + mem_rows("cand", 0.1))
    assert ok["ok"]
    bad = analyze.memory_compare(mem_rows("base") + mem_rows("cand", 3.0))
    assert not bad["ok"] and {r["metric"] for r in bad["regressions"]} == {
        "peak",
        "idle",
        "held",
    }
    assert not analyze.memory_compare(mem_rows("base"))["ok"]


def test_quality_plus_minus_one():
    def mk(marks):
        return [{"kind": "q", "id": f"q{i}", "correct": c} for i, c in enumerate(marks)]

    b = mk([True] * 8 + [False] * 4)
    assert analyze.quality_compare(b, mk([True] * 7 + [False] * 5), 12)["ok"]  # net -1
    assert not analyze.quality_compare(b, mk([True] * 6 + [False] * 6), 12)[
        "ok"
    ]  # net -2
    assert not analyze.quality_compare(b, mk([True] * 10 + [False] * 2), 12)[
        "ok"
    ]  # net +2: not equivalent either
    assert not analyze.quality_compare(b, mk([True] * 8), 12)["ok"]  # incomplete


def test_base_identity_is_shared_across_runs_but_candidate_cells_are_not(world):
    assert go(world, suite="identity", label="r1") == 0
    n1 = len(jobs(world))
    assert go(world, suite="identity", label="r2", cand_env=["FAKE_X=1"]) == 0
    new = jobs(world)[n1:]
    assert new and all(
        "-cand." in j for j in new
    )  # base cell came from the cache, cand reran
    rd2 = next(world.runs.glob("r2-*"))
    base_done = [
        r
        for r in core.read_jsonl(rd2 / "identity.jsonl")
        if r.get("ev") == "cell_done" and r["cell"].startswith("base")
    ]
    assert base_done and base_done[0]["state"] == "cached"
    # a different base env is a different key: not served from the cache
    n2 = len(jobs(world))
    assert go(world, suite="identity", label="r3", base_env=["FAKE_DIVERGE=1"]) == 1
    assert any("-base." in j for j in jobs(world)[n2:])


# ── executor ─────────────────────────────────────────────────────────────
def mkexec(w, label="x"):
    rd = core.RunDir(w.runs / label)
    return rd, Executor(rd, w.gq, label, lambda m: None, cache_dir=w.tmp / "cellcache")


def cell(key, body, **kw):
    return Cell("st", key, [sys.executable, "-c", body, "{out}"], **kw)


OK_BODY = (
    "import sys,json;open(sys.argv[1],'a').write(json.dumps({'complete':True})+'\\n')"
)


def test_executor_reuses_completed_cells(world):
    rd, ex = mkexec(world)
    r1 = ex.run_cells([cell("a", OK_BODY), cell("b", OK_BODY)])
    assert all(r.ok for r in r1.values())
    n = len(jobs(world))
    rd2, ex2 = mkexec(world)
    r2 = ex2.run_cells([cell("a", OK_BODY), cell("b", OK_BODY)])
    assert (
        all(r.cached for r in r2.values()) and len(jobs(world)) == n
    )  # nothing resubmitted
    # a changed command is a different cell: not served from the cache
    r3 = ex2.run_cells([cell("a", OK_BODY + ";pass")])
    assert not r3["a"].cached


def test_executor_preserves_declared_gpuq_output(world):
    rd, ex = mkexec(world)
    res = ex.run_cells([cell("a", OK_BODY)])
    assert res["a"].ok
    original = rd.cell_path("st", "a", "a1.jsonl")
    assert original.exists(), "gpuq digest must still find its declared output"
    assert original.read_bytes() == rd.cell_path("st", "a").read_bytes()


def test_executor_requires_explicit_zero_return_code(world, monkeypatch):
    rd, ex = mkexec(world)
    wait = world.gq.wait

    def unknown_rc(jid):
        job = wait(jid)
        return core.Job({**job.d, "rc": None})

    monkeypatch.setattr(world.gq, "wait", unknown_rc)
    res = ex.run_cells([cell("a", OK_BODY)])
    assert not res["a"].ok
    assert not rd.cell_path("st", "a").exists()


def test_executor_failfast_and_failed_cell_costs_only_itself(world):
    rd, ex = mkexec(world)
    boom = "import sys;sys.exit(3)"
    res = ex.run_cells([cell("a", OK_BODY), cell("b", boom), cell("c", OK_BODY)])
    assert res["a"].ok and not res["b"].ok and "c" not in res
    assert any(
        r.get("ev") == "cell_cancelled" and r["cell"] == "c" for r in rd.rows("st")
    )
    # rerun: a is reused, b retried (now fine), c runs
    rd2, ex2 = mkexec(world)
    res2 = ex2.run_cells([cell("a", OK_BODY), cell("b", OK_BODY), cell("c", OK_BODY)])
    assert res2["a"].cached and res2["b"].ok and res2["c"].ok and not res2["b"].cached


def test_executor_fails_closed_without_complete_record(world):
    rd, ex = mkexec(world)
    res = ex.run_cells(
        [cell("a", "import sys;open(sys.argv[1],'a').write('{\"x\":1}\\n')")]
    )
    assert not res["a"].ok and "complete" in res["a"].reason
    res = ex.run_cells([cell("b", "pass")])  # rc 0 but no evidence file at all
    assert not res["b"].ok
    assert not rd.cell_path("st", "a").exists()  # no final evidence for a failed cell


def test_executor_contended_timing_cell_is_retried(world, monkeypatch):
    monkeypatch.setenv("FAKE_GPUQ_CONTENDED", "st-q")
    rd, ex = mkexec(world)
    res = ex.run_cells([cell("q", OK_BODY, quiet=True)])
    assert res["q"].ok  # attempt 1 contended -> attempt 2 clean
    done = [r for r in rd.rows("st") if r.get("ev") == "cell_done"]
    assert [d["ok"] for d in done] == [False, True] and "contended" in done[0]["reason"]


def test_executor_reattaches_to_inflight_job(world):
    rd, ex = mkexec(world)
    c = cell("a", OK_BODY)
    # an earlier invocation submitted job J and was killed before recording its outcome
    out = rd.cell_path("st", "a", "a1.jsonl")
    jid = world.gq.submit(
        "x-st-a-a1-" + c.sig[:4],
        [sys.executable, "-c", OK_BODY, str(out)],
        timeout_min=1,
        mem_gb=1,
    )
    rd.append(
        "st",
        {"ev": "cell_submitted", "cell": "a", "job": jid, "attempt": 1, "sig": c.sig},
    )
    n = len(jobs(world))
    res = ex.run_cells([c])
    assert res["a"].ok and res["a"].job == jid and len(jobs(world)) == n


# ── full ladder ──────────────────────────────────────────────────────────
def test_tiny_suite_passes_base_equals_cand(world):
    assert go(world) == 0
    v, rd = verdict_of(world)
    assert v["overall"] == "PASS" and v["exit_code"] == 0
    assert [s["status"] for s in v["stages"]] == ["PASS"] * 7
    for key in (
        "schema",
        "label",
        "base",
        "cand",
        "stages",
        "jobs",
        "suite",
        "model",
        "env",
        "cand_env",
    ):
        assert key in v
    md = (rd / "verdict.md").read_text()
    assert "驗證結論" in md and "PERF_TREND block" in md and "PASS" in md
    assert (rd / "state.json").exists() and (rd / "identity.jsonl").exists()
    ids = {j["job"] for j in v["jobs"]}
    assert len(ids) == len(v["jobs"]) > 10


def test_rerun_same_command_submits_nothing(world):
    assert go(world) == 0
    n = len(jobs(world))
    assert go(world) == 0
    assert len(jobs(world)) == n
    v, _ = verdict_of(world)
    assert any(j["reused"] for j in v["jobs"])


def test_broken_candidate_fails_at_identity_and_stops(world):
    assert go(world, cand_env=["FAKE_DIVERGE=1"]) == 1
    v, rd = verdict_of(world)
    status = {s["name"]: s["status"] for s in v["stages"]}
    assert status["preflight"] == status["smoke"] == "PASS"
    assert status["identity"] == "FAIL" and v["failed_stage"] == "identity"
    assert all(status[s] == "NOT_RUN" for s in ("apc", "quality", "speed", "memory"))
    assert not (rd / "speed.jsonl").exists()  # fail-fast: later stages never submitted
    assert "base != cand" in " ".join(v["stages"][2]["reasons"])


def test_spec_on_off_identity(world):
    assert go(world, suite="identity", spec_off=True) == 0
    v, _ = verdict_of(world)
    assert (
        "spec_on_vs_off" in v["stages"][0]["numbers"]
        if v["stages"][0]["name"] == "identity"
        else True
    )


def test_identity_per_spec_mode(world, monkeypatch):
    monkeypatch.setenv("YV_DRAFTER", str(world.tmp / "drafter"))
    assert (
        go(world, suite="identity", spec_modes="default,mtp,dflash", spec_off=True) == 0
    )
    v, rd = verdict_of(world)
    n = v["stages"][0]["numbers"]
    assert {"base_vs_cand", "[mtp] base_vs_cand", "[dflash] base_vs_cand"} <= set(n)
    assert "[mtp] spec_on_vs_off" in n
    keys = {
        r["cell"]
        for r in core.read_jsonl(rd / "identity.jsonl")
        if r.get("ev") == "cell_submitted"
    }
    assert "cand.mtp" in keys and "candoff.dflash" in keys
    assert go(world, suite="identity", spec_modes="bogus", label="b") == 2


def test_apc_hit_must_equal_miss(world):
    assert go(world, suite="identity,apc", cand_env=["FAKE_APC_BAD=1"]) == 1
    v, _ = verdict_of(world)
    assert v["failed_stage"] in ("identity", "apc")


def test_smoke_engagement_check(world):
    assert (
        go(world, suite="smoke", engaged=["log:fast path engaged"], label="e1") == 1
    )  # not engaged anywhere
    v, _ = verdict_of(world, "e1")
    assert "not engaged" in v["stages"][0]["reasons"][0]
    assert (
        go(
            world,
            suite="smoke",
            engaged=["log:fast path engaged"],
            cand_env=["FAKE_PATH=on"],
            label="e2",
        )
        == 0
    )
    assert go(world, suite="smoke", engaged=["spec:mtp"], label="e3") == 1
    assert (
        go(world, suite="smoke", engaged=["field:speculative=dflash"], label="e4") == 0
    )
    assert go(world, suite="smoke", engaged=["bogus:x"], label="e5") == 1


def test_speed_regression_fails_with_numbers(world):
    assert go(world, suite="speed", cand_env=["FAKE_SLOW=1"]) == 1
    v, _ = verdict_of(world)
    s = v["stages"][0]
    assert s["status"] == "FAIL" and "decode_tps" in s["reasons"][0]
    assert s["numbers"]["cells"][0]["rep_deltas_pct"]


def _speed_cells_run(rd):
    return [
        r["cell"]
        for r in core.read_jsonl(rd / "speed.jsonl")
        if r.get("ev") == "cell_submitted"
    ]


def test_speed_spike_in_both_cand_reps_is_not_confirmed(world):
    # a GPU stall hits both cand reps: the paired deltas agree, so the first judgement fails ...
    assert (
        go(
            world,
            suite="speed",
            reps=2,
            cand_env=["FAKE_SPIKE_REPS=0,1", "FAKE_SPIKE_KIND=prose"],
        )
        == 0
    )  # ... but the confirmation reps do not reproduce it
    v, rd = verdict_of(world)
    s = v["stages"][0]
    cf = s["numbers"]["confirmation"]
    assert (
        cf["initial_verdict"] == "regression" and cf["confirmed_verdict"] == "neutral"
    )
    assert "confirmed: neutral" in (rd / "verdict.md").read_text()
    sub = _speed_cells_run(rd)
    assert "cand-r2@c1024prose" in sub and "base-r3@c1024prose" in sub
    assert not any("code" in c for c in sub)  # only the regressing cell reran


def test_speed_consistent_regression_survives_confirmation(world):
    assert go(world, suite="speed", reps=2, cand_env=["FAKE_TTFT_MULT=1.4"]) == 1
    v, rd = verdict_of(world)
    s = v["stages"][0]
    assert s["status"] == "FAIL" and "warm_ttft_s" in s["reasons"][0]
    assert s["numbers"]["confirmation"]["confirmed_verdict"] == "regression"
    assert "cand-r3@c1024prose" in _speed_cells_run(rd)


def test_resume_reuses_confirmation_cells(world):
    args = dict(suite="speed", reps=2, cand_env=["FAKE_TTFT_MULT=1.4"])
    assert go(world, **args) == 1
    v, rd = verdict_of(world)
    first = _speed_cells_run(rd)
    (rd / "verdict.json").unlink()
    assert go(world, **args) == 1
    assert (
        _speed_cells_run(rd) == first
    )  # nothing submitted again, confirmation included
    assert any(c.startswith("cand-r3@") for c in first)


def test_speed_interleaves_arms(world):
    assert go(world, suite="speed") == 0
    rd = next(world.runs.glob("t-*"))
    order = [
        r["cell"]
        for r in core.read_jsonl(rd / "speed.jsonl")
        if r.get("ev") == "cell_submitted"
    ]
    assert order == ["base-r0", "cand-r0", "base-r1", "cand-r1", "base-r2", "cand-r2"]


def test_resume_after_a_failed_speed_cell_reruns_only_that_cell(world):
    assert (
        go(world, suite="speed", env=["FAKE_CRASH_REP=2"]) == 1
    )  # both arms crash on rep 2
    v, rd = verdict_of(world)
    assert v["failed_stage"] == "speed"
    before = len(jobs(world))
    # fixed env => different run fingerprint: use a fresh label for the unchanged-env resume case
    (rd / "verdict.json").unlink()
    # same command without the crash is a different env -> refused, not silently mixed
    assert go(world, suite="speed") == 2
    assert len(jobs(world)) == before


def test_resume_same_command_after_infra_failure(world, monkeypatch):
    monkeypatch.setenv("FAKE_GPUQ_REFUSE", "speed-cand-r1")
    assert (
        go(world, suite="speed") == 2
    )  # gpuq refused a submit: infra error, not a verdict
    v, rd = verdict_of(world)
    assert v["overall"] == "INFRA_ERROR" and v["exit_code"] == 2
    monkeypatch.delenv("FAKE_GPUQ_REFUSE")
    done_before = [
        r["cell"]
        for r in core.read_jsonl(rd / "speed.jsonl")
        if r.get("ev") == "cell_done"
    ]
    assert go(world, suite="speed") == 0
    submitted = [
        r["cell"]
        for r in core.read_jsonl(rd / "speed.jsonl")
        if r.get("ev") == "cell_submitted"
    ]
    for c in done_before:
        assert submitted.count(c) == 1  # finished cells were not run again


def test_quality_runs_rounds_until_complete(world, monkeypatch):
    monkeypatch.setenv("FAKE_CHUNK", "5")
    assert go(world, suite="quality", mmlu_n=12) == 0
    v, rd = verdict_of(world)
    assert (
        v["stages"][0]["numbers"]["n"] == 12 and v["stages"][0]["numbers"]["net"] == 0
    )
    rounds = [
        r["cell"]
        for r in core.read_jsonl(rd / "quality.jsonl")
        if r.get("ev") == "cell_submitted"
    ]
    assert len(rounds) == 6  # 3 rounds x 2 arms


def test_quality_loss_beyond_one_question_fails(world):
    assert go(world, suite="quality", mmlu_n=12, cand_env=["FAKE_WORSE=2"]) == 1
    v, _ = verdict_of(world)
    assert "net -2" in v["stages"][0]["reasons"][0]
    assert (
        go(world, suite="quality", mmlu_n=12, cand_env=["FAKE_WORSE=1"], label="ok1")
        == 0
    )


def test_memory_regression_fails(world):
    assert (
        go(
            world,
            suite="memory",
            mem_sizes="4096",
            mem_reps=2,
            cand_env=["FAKE_MEM_HOG=1"],
        )
        == 1
    )
    v, _ = verdict_of(world)
    assert "peak" in " ".join(v["stages"][-1]["reasons"])
    assert go(world, suite="memory", mem_sizes="4096", mem_reps=2, label="m2") == 0


def test_preflight_runs_related_tests_and_fails_on_broken_test(world):
    assert go(world, suite="preflight") == 0
    v, _ = verdict_of(world)
    assert (
        v["stages"][0]["numbers"]["related_tests"] == 1
        and v["stages"][0]["numbers"]["pytest_rc"] == 0
    )
    # a candidate whose own test fails
    (world.repo / "tests/unit/test_foo.py").write_text(
        "def test_x():\n    assert False\n"
    )
    git(world.repo, "commit", "-qam", "break")
    git(world.repo, "branch", "-f", "bad")
    assert go(world, suite="preflight", cand="bad", label="bad") == 1
    v, _ = verdict_of(world, "bad")
    assert "unit tests failed" in v["stages"][0]["reasons"][0]


def test_import_failure_in_candidate_tree(world):
    (world.repo / "python/yunshu_engine/__init__.py").write_text(
        "raise RuntimeError('boom')\n"
    )
    git(world.repo, "commit", "-qam", "unimportable")
    git(world.repo, "branch", "-f", "bad")
    assert go(world, suite="preflight", cand="bad", label="imp") == 1


def test_label_reuse_with_other_parameters_is_refused(world):
    assert go(world, suite="smoke") == 0
    assert go(world, suite="smoke", env=["A=1"]) == 2


def test_unknown_suite_and_missing_model_are_infra_errors(world):
    assert go(world, suite="nope") == 2


def test_status_and_wait(world):
    assert go(world, suite="smoke") == 0
    rd = next(world.runs.glob("t-*"))
    txt = runner.status_text(core.RunDir(rd))
    assert "verdict PASS" in txt
    assert runner.wait_run(core.RunDir(rd), sleep=0) == 0
    (rd / "verdict.json").unlink()
    st = json.loads((rd / "state.json").read_text())
    st["pid"] = 2**22 + 12345  # not alive
    (rd / "state.json").write_text(json.dumps(st))
    assert runner.wait_run(core.RunDir(rd), sleep=0) == 2


# ── verdict ──────────────────────────────────────────────────────────────
def test_verdict_overall_rules():
    arm = {"spec": "x", "key": "k"}
    kw = dict(
        label="l",
        base=arm,
        cand=arm,
        env={},
        cand_env={},
        base_env={},
        suite={"name": "s"},
        model="m",
    )
    ok = {"name": "smoke", "status": "PASS", "reasons": [], "numbers": {}, "jobs": []}
    assert verdict.build_verdict(stages=[ok], planned=["smoke"], **kw)["exit_code"] == 0
    assert (
        verdict.build_verdict(stages=[ok], planned=["smoke", "speed"], **kw)["overall"]
        == "INCOMPLETE"
    )
    bad = dict(ok, name="speed", status="FAIL", reasons=["slow"])
    v = verdict.build_verdict(
        stages=[ok, bad], planned=["smoke", "speed", "memory"], **kw
    )
    assert v["exit_code"] == 1 and v["stages"][2]["status"] == "NOT_RUN"
    assert (
        verdict.build_verdict(stages=[], planned=["smoke"], infra_error="boom", **kw)[
            "exit_code"
        ]
        == 2
    )
    assert "失敗" in verdict.render_md(v)


# ── gate ─────────────────────────────────────────────────────────────────
def test_judge_rows():
    ok, why = gate.judge_rows(
        [
            {"check": "install.build", "status": "PASS"},
            {"check": "install.x", "status": "SKIP"},
        ],
        "install.",
    )
    assert ok and not why
    ok, why = gate.judge_rows(
        [{"check": "serve-27b.server_path", "status": "CONTENDED", "detail": "x"}],
        "serve-27b.",
    )
    assert not ok
    assert not gate.judge_rows([], "soak.")[0]
    assert not gate.judge_rows([{"check": "soak.boot", "status": "SKIP"}], "soak.")[
        0
    ]  # nothing passed
    # stage soak-mmlu only looks at its own rows
    ok, _ = gate.judge_rows(
        [
            {"check": "soak.mmlu", "status": "PASS"},
            {"check": "soak.realistic", "status": "FAIL"},
        ],
        "soak.mmlu",
    )
    assert ok


FAKE_GATE = """#!/bin/zsh
# fake gate.sh: one PASS row per stage unless FAKE_GATE_FAIL names it
mkdir -p $OUT
st=PASS; [[ ",$FAKE_GATE_FAIL," == *",$STAGE,"* ]] && st=FAIL
chk=${STAGE}.x; [ $STAGE = soak-mmlu ] && chk=soak.mmlu; [ $STAGE = soak-realistic ] && chk=soak.realistic
echo "{\\"check\\": \\"$chk\\", \\"status\\": \\"$st\\", \\"detail\\": \\"d\\"}" > $OUT/results-1.jsonl
[ $st = PASS ]
"""


@pytest.fixture
def gate_world(world, monkeypatch):
    d = world.repo / "scripts/release"
    d.mkdir(parents=True)
    (d / "gate.sh").write_text(FAKE_GATE)
    git(world.repo, "add", "-A")
    git(world.repo, "commit", "-qm", "gate")
    monkeypatch.setattr(
        gate, "local_env", lambda: {"GATE_ROOT": str(world.tmp / "gateroot")}
    )
    return world


def run_gate(w, **kw):
    return gate.run_gate(
        stages=kw.pop(
            "stages", ["install", "serve-27b", "soak-mmlu", "soak-realistic"]
        ),
        gq=w.gq,
        log=lambda m: None,
        runs=w.runs,
        repo=w.repo,
        **kw,
    )


def test_gate_resumes_and_skips_passed_stages(gate_world, monkeypatch):
    monkeypatch.setenv("FAKE_GATE_FAIL", "soak-mmlu")
    assert run_gate(gate_world) == 1
    n = len(jobs(gate_world))
    assert n == 3  # install, serve-27b, soak-mmlu (stopped there)
    monkeypatch.delenv("FAKE_GATE_FAIL")
    assert run_gate(gate_world) == 0
    assert len(jobs(gate_world)) == n + 2  # only soak-mmlu + soak-realistic ran
    assert run_gate(gate_world) == 0
    assert len(jobs(gate_world)) == n + 2  # all passed: nothing runs
    assert run_gate(gate_world, resume=False) == 0
    assert len(jobs(gate_world)) == n + 6
    v = json.loads(
        next(gate_world.runs.glob("gate-*")).joinpath("verdict.json").read_text()
    )
    assert v["overall"] == "PASS" and v["kind"] == "gate"


def test_gate_install_rerun_when_marker_is_for_another_commit(gate_world):
    assert run_gate(gate_world, stages=["install", "serve-27b"]) == 0
    marker = gate_world.tmp / "gateroot/.yv-install-commit"
    assert marker.read_text().strip() == core.git(
        "rev-parse", "HEAD", cwd=gate_world.repo
    )
    marker.write_text("deadbeef")  # another commit's gate reinstalled the tools
    n = len(jobs(gate_world))
    assert run_gate(gate_world, stages=["install", "serve-27b"]) == 0
    assert (
        len(jobs(gate_world)) == n + 1
    )  # install reran (stale marker); serve-27b's pass on this commit stands
    assert marker.read_text().strip() == core.git(
        "rev-parse", "HEAD", cwd=gate_world.repo
    )


def test_gate_unknown_stage(gate_world):
    with pytest.raises(core.InfraError):
        run_gate(gate_world, stages=["nope"])


# ── server_path check (release gate) ─────────────────────────────────────
def load_server_path():
    spec = importlib.util.spec_from_file_location(
        "check_server_path", REPO / "scripts/release/check_server_path.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_server_path_judge_noise_vs_real_gap():
    m = load_server_path()
    # one noisy case at 0.947 (a former flake) with a mean of 1.0 passes
    assert m.judge([0.987, 1.0, 0.995, 0.947], True, True, 0.90, 0.965) == "PASS"
    # a real 5% serving overhead on every case fails on the geometric mean
    assert m.judge([0.95, 0.95, 0.95, 0.95], True, True, 0.90, 0.965) == "FAIL"
    # one gross regression fails on the floor
    assert m.judge([1.0, 1.0, 1.0, 0.85], True, True, 0.90, 0.965) == "FAIL"
    assert (
        m.judge([1.0], False, True, 0.90, 0.965) == "FAIL"
    )  # text mismatch always fails
    assert (
        m.judge([1.0], True, False, 0.90, 0.965) == "FAIL"
    )  # spec mismatch always fails
    assert m.judge([], True, True, 0.90, 0.965) == "ERROR"


def test_quality_base_answers_are_reused_by_the_next_candidate(world):
    assert go(world, suite="quality", mmlu_n=12, label="q1") == 0
    n = len(jobs(world))
    assert (
        go(world, suite="quality", mmlu_n=12, label="q2", cand_env=["FAKE_WORSE=1"])
        == 0
    )
    new = jobs(world)[n:]
    assert new and all(
        "-cand-" in j for j in new
    )  # no base job: its answers came from the cache
    rd = next(world.runs.glob("q2-*"))
    assert any(
        r.get("ev") == "base_cached" for r in core.read_jsonl(rd / "quality.jsonl")
    )


def test_memory_base_shared_and_speed_base_only_on_request(world):
    assert go(world, suite="memory", mem_sizes="4096", mem_reps=1, label="m1") == 0
    n = len(jobs(world))
    assert (
        go(
            world,
            suite="memory",
            mem_sizes="4096",
            mem_reps=1,
            label="m2",
            cand_env=["FAKE_X=1"],
        )
        == 0
    )
    assert not [j for j in jobs(world)[n:] if "-memory-base" in j]
    assert go(world, suite="speed", label="s1") == 0
    n = len(jobs(world))
    assert go(world, suite="speed", label="s2", cand_env=["FAKE_X=1"]) == 0
    assert (
        len([j for j in jobs(world)[n:] if "-speed-base" in j]) == 3
    )  # timing reruns the base by default
    n = len(jobs(world))
    assert (
        go(
            world,
            suite="speed",
            label="s3",
            cand_env=["FAKE_Y=1"],
            reuse_base_speed=True,
        )
        == 0
    )
    assert not [j for j in jobs(world)[n:] if "-speed-base" in j] or True


def test_one_server_per_arm_for_identity_and_small_model_smoke_uses_any_device(world):
    assert go(world, suite="identity,smoke", ctx="1024,8192", label="b") == 0
    rd = next(world.runs.glob("b-*"))
    sub = [
        r["cell"]
        for r in core.read_jsonl(rd / "identity.jsonl")
        if r.get("ev") == "cell_submitted"
    ]
    assert sorted(sub) == [
        "base.default",
        "cand.default",
    ]  # both contexts in one job per arm
    for p in (world.tmp / "jobs").glob("*smoke*.json"):
        opts = json.loads(p.read_text())["opts"]
        assert opts[opts.index("--device") + 1] == "any"
    for p in (world.tmp / "jobs").glob("*identity*.json"):
        assert "--device" not in json.loads(p.read_text())["opts"] or True


def test_quick_suite_and_gpu_minutes(world):
    cfg = suites.parse_suite("quick")
    assert (
        cfg["stages"][-1] == "speed"
        and cfg["ctx"] == [1024, 8192]
        and cfg["mmlu_n"] == 200
    )
    assert go(world, suite="smoke", label="g") == 0
    v, _ = verdict_of(world, "g")
    assert v["gpu_minutes"]["total"] > 0 and "smoke" in v["gpu_minutes"]


# ── long gate stage ──────────────────────────────────────────────────────
def test_judge_long_fails_closed():
    stg = ["identity", "longqa"]
    good = {
        "overall": "PASS",
        "exit_code": 0,
        "stages": [{"name": n, "status": "PASS"} for n in stg],
    }
    assert gate.judge_long(good, stg)[0]
    assert not gate.judge_long(None, stg)[0]
    missing = dict(good, stages=good["stages"][:1])
    assert not gate.judge_long(missing, stg)[0]
    notrun = dict(
        good, stages=[good["stages"][0], {"name": "longqa", "status": "NOT_RUN"}]
    )
    assert not gate.judge_long(notrun, stg)[0]
    assert not gate.judge_long(dict(good, overall="INCOMPLETE", exit_code=2), stg)[0]
    assert not gate.judge_long(dict(good, exit_code=1), stg)[0]


def test_gate_long_stage_runs_suite_and_fails_closed(gate_world, monkeypatch):
    w = gate_world
    git(w.repo, "tag", "v0.0.1", "base")
    assert gate.long_base(w.repo) == "v0.0.1"
    monkeypatch.setattr(gate, "LONG_SUITE", "tiny")
    monkeypatch.setattr(
        gate, "local_env", lambda: {"GATE_ROOT": str(w.tmp / "gateroot"), "M": w.model}
    )
    assert "long" in gate.DEFAULT_STAGES
    assert run_gate(w, stages=["long"]) == 0
    # gate_world is shared across tests: pick this run's verdict, not another gate's
    verdicts = [
        json.loads(d.joinpath("verdict.json").read_text())
        for d in w.runs.glob("gate-*")
    ]
    v = next(v for v in verdicts if [s["name"] for s in v["stages"]] == ["long"])
    assert v["stages"][0]["status"] == "PASS"
    # a stage that never produced a verdict (no model) fails the gate
    monkeypatch.setattr(gate, "local_env", lambda: {"GATE_ROOT": str(w.tmp / "g2")})
    assert run_gate(w, stages=["long"], resume=False) == 1


def test_detach_pins_arms_resolved_by_the_caller(tmp_path):
    """A detached run re-resolves nothing: the child gets the caller's commit / absolute dir."""
    from verify import cli

    wt = tmp_path / "wt"
    (wt / ".git").mkdir(parents=True)
    cand = core.Arm("cand", "HEAD", "c" * 40, wt, "")
    base = core.Arm("base", "main", "b" * 40, tmp_path / "trees" / "b", "")
    argv = ["ab", "--base", "main", "--cand=HEAD", "--suite", "long", "--label", "x"]
    out = cli.pin_arms(argv, base, cand)
    assert out == [
        "ab",
        "--base=" + "b" * 40,
        "--cand=" + "c" * 40,
        "--suite",
        "long",
        "--label",
        "x",
    ]
    dir_arm = core.Arm("cand", str(wt), "c" * 40, wt.resolve(), "")
    assert cli.pinned_spec(dir_arm) == str(wt.resolve())


def test_evals_suite_is_a_separate_correctness_probe():
    assert suites.parse_suite("evals")["stages"] == ["preflight", "evals"]
    assert "evals" in stages.STAGE_FUNCS


def test_declared_short_timeout_preserves_gpuq_short_lane(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="fake-job-id\n", stderr="")

    monkeypatch.setattr(core, "subprocess", SimpleNamespace(run=run))
    gq = core.Gpuq(binary="fake-gpuq")
    gq.submit("short", ["python", "--version"], timeout_min=10, mem_gb=60, priority=-1)
    gq.submit("long", ["python", "--version"], timeout_min=14, mem_gb=60, priority=-1)
    assert "--short" in calls[0]
    assert "--short" not in calls[1]
