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
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
FAKES = Path(__file__).parent / "yv_fakes"

from verify import analyze, core, gate, runner, stages, suites, verdict  # noqa: E402
from verify.execute import Cell, Executor  # noqa: E402


@pytest.mark.parametrize("stage", [stages.stage_tavily, stages.stage_websearch])
def test_web_cells_apply_recorded_arm_environment(stage, tmp_path):
    from types import SimpleNamespace

    cells = []

    class CapturedError(Exception):
        pass

    class Capture:
        def run_cells(self, submitted):
            cells.extend(submitted)
            raise CapturedError

    ctx = stages.Ctx(
        run=None,
        exe=Capture(),
        base=SimpleNamespace(path=tmp_path / "base"),
        cand=SimpleNamespace(path=tmp_path / "cand"),
        env={
            "COVAUDIT_BIN": "/isolated path/yunshu",
            "SHARED": "common",
            "PYTHONPATH": "wrong-tree",
        },
        base_env={"ARM": "base", "SHARED": "base-override"},
        cand_env={"ARM": "cand", "SHARED": "cand-override"},
        model="tiny",
        model_name="tiny",
        suite={},
        mem_gb=14,
    )
    with pytest.raises(CapturedError):
        stage(ctx)
    assert len(cells) == 2
    for cell in cells:
        interpreter = cell.argv.index(ctx.py)
        assignments = dict(arg.split("=", 1) for arg in cell.argv[1:interpreter])
        assert assignments["COVAUDIT_BIN"] == "/isolated path/yunshu"
        assert assignments["ARM"] == cell.key
        assert assignments["SHARED"] == cell.key + "-override"
        assert assignments["PYTHONPATH"].split(":")[0] == str(
            tmp_path / cell.key / "python"
        )


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
    # The caller's production queue window must not alter fixture expectations.
    monkeypatch.delenv("YV_MAX_PENDING", raising=False)
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
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
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
def test_multimodal_speed_rejects_short_turn_regression_and_keeps_long_cells():
    numbers = {}
    for rep in range(3):
        for arm in ("base", "cand"):
            numbers[f"{arm}-r{rep}"] = [
                dict(
                    size=size,
                    kind=kind,
                    ttft_s=(
                        0.1177
                        if arm == "cand" and kind == "turn2-hit" and size == 46
                        else 0.1
                    ),
                )
                for size in (46, 32768)
                for kind in ("cold", "warm", "turn2-hit")
            ]
    result = stages._multimodal_speed(numbers, 3, 2)
    assert not result["ok"]
    assert result["regressions"][0]["ctx"] == 46
    assert {r["ctx"] for r in result["cells"]} == {46, 32768}
    for rep in range(3):
        for row in numbers[f"cand-r{rep}"]:
            row["ttft_s"] = 0.1
    assert stages._multimodal_speed(numbers, 3, 2)["ok"]


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
        "ttft_profile",
        "ssd_ttft",
        "ssd_ttft_profile",
        "multimodal",
        "snapshot",
        "modelprobe",
        "lane_bandwidth",
        "client_compat",
        "respfeat",
        "websearch",
        "rerank",
        "embedding",
        "priorart",
        "evals",
        "console",
        "deployment",
        "deployment-idle",
        "service_deploy",
        "telemetry",
        "telemetry-tiny",
        "tavily",
        "searchrank",
        "toolparse",
        "qwen4_mtp",
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


def test_untracked_files_change_the_dir_arm_key(world):
    d = world.repo
    a0 = core.resolve_arm("cand", str(d))
    new = d / "python/yunshu_engine/new_mod.py"
    new.write_text("A = 1\n")
    a1 = core.resolve_arm("cand", str(d))
    new.write_text("A = 2\n")
    a2 = core.resolve_arm("cand", str(d))
    assert len({a0.key, a1.key, a2.key}) == 3 and a1.dirty


def test_dirty_pinned_ref_worktree_is_not_keyed_clean(world):
    c = core.resolve_arm("cand", "cand", world.trees)
    assert not c.dirty
    (c.path / "python/yunshu_engine/foo.py").write_text("X = 99\n")
    c2 = core.resolve_arm("cand", "cand", world.trees)
    assert c2.dirty and c2.key != c.key


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


def test_identity_separates_a_changed_input_from_changed_arithmetic():
    """A cell whose prompt token count differs had a different input (a deliberate tokenization
    change, e.g. special-token neutralisation): it fails by default and is listed as input_changed
    only when the run says the change is deliberate; other cells still compare."""
    base = rows("a") + [dict(r, kind="prose", sha="p", pt=8203) for r in rows()]
    cand = rows("a") + [dict(r, kind="prose", sha="q", pt=8206) for r in rows()]
    strict = analyze.compare_identity(base, cand)
    assert not strict["ok"]
    assert {m["why"] for m in strict["mismatches"]} == {"prompt tokens differ"}
    ok = analyze.compare_identity(base, cand, input_change_ok=True)
    assert ok["ok"] and ok["compared"] == 3 and len(ok["input_changed"]) == 3
    # same prompt, different digest still fails with the flag
    worse = rows("b") + [dict(r, kind="prose", sha="q", pt=8206) for r in rows()]
    assert not analyze.compare_identity(base, worse, input_change_ok=True)["ok"]


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


def test_a_slower_follow_up_decode_is_a_regression():
    base = [rows() for _ in range(3)]
    cand = [rows() for _ in range(3)]
    for rep in cand:
        rep[2]["dec_tps"] = 70.0  # turn2 only; the cold reply is as fast
    sp = analyze.speed_compare(base, cand)
    assert not sp["ok"]
    assert [r["metric"] for r in sp["regressions"]] == ["followup_decode_tps"]


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


@pytest.mark.parametrize("failure", [None, "6"])
def test_executor_bounds_pending_cells_and_stops_before_next_window(
    world, monkeypatch, failure
):
    from verify.execute import CellResult

    _, ex = mkexec(world)
    live, submitted, cancelled = set(), [], []
    monkeypatch.setattr(ex, "_cached", lambda c: None)
    monkeypatch.setattr(ex, "_shared", lambda c: None)
    monkeypatch.setattr(ex, "_inflight", lambda c: None)
    monkeypatch.setattr(ex, "_attempts", lambda c: 0)

    def submit(c, attempt):
        live.add(c.key)
        submitted.append(c.key)
        assert len(live) <= 6
        return c.key

    def finish(c, jid, attempt):
        live.remove(c.key)
        return CellResult(c.key, c.key != failure)

    monkeypatch.setattr(ex, "_submit", submit)
    monkeypatch.setattr(ex, "_finish", finish)
    monkeypatch.setattr(ex.gq, "cancel", cancelled.append)
    results = ex.run_cells([Cell("st", str(i), []) for i in range(18)])
    if failure is None:
        assert len(results) == len(submitted) == 18
        assert not live and not cancelled
    else:
        assert submitted == [str(i) for i in range(12)]
        assert len(results) == 7 and not results["6"].ok
        assert cancelled == [str(i) for i in range(7, 12)]


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
    # the first reuse run has nothing to reuse yet: it records the shared base
    assert len([j for j in jobs(world)[n:] if "-speed-base" in j]) == 3
    n = len(jobs(world))
    assert (
        go(
            world,
            suite="speed",
            label="s4",
            cand_env=["FAKE_Z=1"],
            reuse_base_speed=True,
        )
        == 0
    )
    assert not [j for j in jobs(world)[n:] if "-speed-base" in j]


def test_longqa_base_cells_share_evidence_and_candidate_never(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "prose-1024.txt").write_text("one")
    monkeypatch.setattr(stages, "PROMPTS", prompts)
    monkeypatch.setattr(stages, "_check_prompts", lambda *a, **k: None)
    ctx = _share_ctx(tmp_path, monkeypatch)
    ctx.suite = {"needle_ctx": [1024]}
    ctx.run = types.SimpleNamespace(path=tmp_path / "run")
    ctx.observe = False
    ctx.mem_gb = 8
    ctx.big = False
    seen = []

    class CapturedStopError(Exception):
        pass

    class Capture:
        def run_cells(self, cells):
            seen.append({c.key: c.share_key for c in cells})
            raise CapturedStopError

    ctx.exe = Capture()

    def keys():
        with pytest.raises(CapturedStopError):
            stages.stage_longqa(ctx)
        return seen[-1]

    k0 = keys()
    base = {k: v for k, v in k0.items() if k.startswith("base@")}
    assert base and all(base.values())
    assert all(v == "" for k, v in k0.items() if k.startswith("cand@"))
    assert keys() == k0  # stable across runs
    (prompts / "prose-1024.txt").write_text("two")  # the prompt bytes changed
    assert {k: v for k, v in keys().items() if k.startswith("base@")} != base


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
        assert "--device" not in json.loads(p.read_text())["opts"]


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
    original = runner.run_ab
    captured = []

    def pinned(a, **kwargs):
        captured.append(a)
        return original(a, **kwargs)

    monkeypatch.setattr(runner, "run_ab", pinned)
    monkeypatch.setenv("YV_LABEL_PREFIX", "releng015-test")
    assert run_gate(w, stages=["long"]) == 0
    assert captured[0].base == git(w.repo, "rev-parse", "v0.0.1^{commit}")
    assert captured[0].cand == git(w.repo, "rev-parse", "HEAD")
    assert captured[0].base != captured[0].cand
    assert captured[0].label.startswith("releng015-")
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


def test_ttft_profile_is_opt_in_and_requires_complete_spans(tmp_path):
    assert "ttft_profile" not in suites.LADDER
    assert "ttft_profile" not in suites.SUITES["long"]["stages"]
    path = tmp_path / "profile.jsonl"
    rows = [
        dict(
            phase=phase,
            call_spans=[dict(stage="gpu_wait")],
            engaged="dflash",
            cached=1,
            followup_apc_required=True,
            copy_index_gc_engaged=True,
            http=dict(first_sse_submit_ms=10),
        )
        for phase in ("cold", "warm", "warm", "warm", "followup")
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert not stages._ttft_profile_valid(path)[0]
    rows.append(dict(complete=True))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert stages._ttft_profile_valid(path)[0]
    rows[1]["call_spans"] = []
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert not stages._ttft_profile_valid(path)[0]


def test_yv_imports_on_its_macos_system_interpreter():
    import os

    interpreter = Path("/usr/bin/python3")
    if not interpreter.exists():
        pytest.skip("macOS system interpreter unavailable")
    result = subprocess.run(
        [str(interpreter), "-m", "verify", "suites"],
        cwd=REPO,
        env=dict(os.environ, PYTHONPATH=str(REPO / "scripts")),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "long" in result.stdout


def test_yv_identity_compare_runs_on_its_macos_system_interpreter():
    # Import alone is not enough: 3.10+ syntax inside a function (zip(strict=...)) only fails when called.
    import os

    interpreter = Path("/usr/bin/python3")
    if not interpreter.exists():
        pytest.skip("macOS system interpreter unavailable")
    code = (
        "from verify import analyze as a; "
        "r = {'lps': [-0.5, -0.25], 'lp_tokens': ['a', 'b']}; "
        "print(a.logprob_delta(r, dict(r, lps=[-0.5, -0.125])))"
    )
    result = subprocess.run(
        [str(interpreter), "-c", code],
        cwd=REPO,
        env=dict(os.environ, PYTHONPATH=str(REPO / "scripts")),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "(0.125, '')"


def test_executor_batches_at_most_six_cells_and_stops_before_later_batches(
    world, monkeypatch
):
    _, ex = mkexec(world)
    submit, finish = ex._submit, ex._finish
    live, peaks, submitted = set(), [], []

    def track_submit(c, attempt):
        jid = submit(c, attempt)
        live.add(jid)
        submitted.append(c.key)
        peaks.append(len(live))
        return jid

    def track_finish(c, jid, attempt):
        result = finish(c, jid, attempt)
        live.remove(jid)
        return result

    monkeypatch.setattr(ex, "_submit", track_submit)
    monkeypatch.setattr(ex, "_finish", track_finish)
    result = ex.run_cells([cell(str(i), OK_BODY) for i in range(14)])
    assert len(result) == 14 and all(r.ok for r in result.values())
    assert max(peaks) == 6 and not live
    _, broken = mkexec(world, "bounded-fail")
    failed = broken.run_cells(
        [cell(str(i), "raise RuntimeError()" if i == 0 else OK_BODY) for i in range(14)]
    )
    assert not failed["0"].ok
    rows = broken.run.rows("st")
    assert sum(r.get("ev") == "cell_submitted" for r in rows) == 6


@pytest.mark.parametrize("quiet", [True, False])
def test_executor_retries_paused_timing_but_accepts_correctness(
    world, monkeypatch, quiet
):
    rd, ex = mkexec(world)
    wait = world.gq.wait
    calls = []

    def paused_once(jid):
        job = wait(jid)
        calls.append(jid)
        if len(calls) == 1:
            return core.Job({**job.d, "pauses": [[1, 2]], "contended": False})
        return job

    monkeypatch.setattr(world.gq, "wait", paused_once)
    result = ex.run_cells([cell("p", OK_BODY, quiet=quiet)])
    assert result["p"].ok
    assert len(calls) == (2 if quiet else 1)
    done = [r for r in rd.rows("st") if r.get("ev") == "cell_done"]
    assert done[0]["paused"] and not done[0]["contended"]
    assert done[0]["ok"] is (not quiet)
    if quiet:
        assert "paused" in done[0]["reason"]


def test_multimodal_evidence_is_fail_closed(tmp_path):
    path = tmp_path / "mm.jsonl"
    path.write_text('{"complete": true}\n')
    assert not stages._multimodal_valid(path, [1], True)[0]
    rows = []
    for size, kinds in [
        (1, ["cold", "warm", "turn2-hit", "turn2-miss", "other-image"]),
        (
            0,
            [
                "anthropic-cold",
                "anthropic-warm",
                "anthropic-turn2-hit",
                "anthropic-turn2-miss",
            ],
        ),
    ]:
        for kind in kinds:
            hit = kind.endswith("warm") or kind.endswith("hit")
            rows.append(
                dict(
                    event="request",
                    size=size,
                    kind=kind,
                    ids=[7],
                    cached=42 if hit else 0,
                )
            )
    rows.append(dict(complete=True))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert stages._multimodal_valid(path, [1], True)[0]
    baseline = [r for r in rows if not r.get("kind", "").startswith("anthropic-")]
    base_path = tmp_path / "base.jsonl"
    base_path.write_text("".join(json.dumps(r) + "\n" for r in baseline))
    assert stages._multimodal_valid(base_path, [1], False, require_anthropic=False)[0]
    assert not stages._multimodal_valid(base_path, [1], True)[0]
    paired = rows[:-1] + [
        dict(
            event="parity_item",
            i=0,
            cold_ids=[7],
            hit_ids=[7],
            cached=42,
            cold_correct=True,
            hit_correct=True,
        ),
        dict(event="quality", n=1, scores=dict(cold=1, hit=1)),
        dict(complete=True),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in paired))
    assert stages._multimodal_quality_valid(path, [1], 1)[0]
    paired[-3]["hit_ids"] = [8]
    path.write_text("".join(json.dumps(r) + "\n" for r in paired))
    assert not stages._multimodal_quality_valid(path, [1], 1)[0]
    rows[1]["ids"] = [8]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not stages._multimodal_valid(path, [1], True)[0]


def test_gate_owner_prefix_and_timing_admission(gate_world):
    assert (
        run_gate(gate_world, stages=["install", "serve-27b"], label_prefix="sync015")
        == 0
    )
    records = [
        json.loads(p.read_text()) for p in (gate_world.tmp / "jobs").glob("*.json")
    ]
    assert all(r["label"].startswith("sync015-gate-") for r in records)
    for record in records:
        assert ("--quiet" in record["opts"]) == ("serve-27b" in record["label"])


def test_gate_cli_pins_ref_and_forwards_owner_prefix(world, monkeypatch):
    from verify import cli

    commit = git(world.repo, "rev-parse", "base")
    seen = {}
    monkeypatch.setattr(core, "TREES", world.trees)

    def capture(stages, **kwargs):
        seen.update(kwargs)
        assert stages == ["install"]
        return 0

    monkeypatch.setattr(cli.gate_mod, "run_gate", capture)
    assert (
        cli.main(
            [
                "gate",
                "--ref",
                commit,
                "--label-prefix",
                "sync015",
                "--stages",
                "install",
                "--root",
                str(world.tmp / "own-gate"),
            ]
        )
        == 0
    )
    assert git(seen["repo"], "rev-parse", "HEAD") == commit
    assert seen["label_prefix"] == "sync015"
    assert seen["extra_env"] == {"GATE_ROOT": str(world.tmp / "own-gate")}


def test_executor_bounds_submitted_cells_before_waiting(world, monkeypatch):
    monkeypatch.setenv("YV_MAX_PENDING", "2")
    _, ex = mkexec(world)
    submit, finish = ex._submit, ex._finish
    live = peak = 0

    def tracked_submit(*args):
        nonlocal live, peak
        result = submit(*args)
        live += 1
        peak = max(peak, live)
        return result

    def tracked_finish(*args):
        nonlocal live
        result = finish(*args)
        live -= 1
        return result

    monkeypatch.setattr(ex, "_submit", tracked_submit)
    monkeypatch.setattr(ex, "_finish", tracked_finish)
    result = ex.run_cells([cell(str(i), OK_BODY) for i in range(7)])
    assert len(result) == 7 and all(r.ok for r in result.values())
    assert peak <= 2


def test_quality_uses_marked_arm_interpreters(world):
    for name in ("base", "cand"):
        arm = core.resolve_arm(name, name, world.trees)
        (arm.path / ".venv/bin").mkdir(parents=True)
        (arm.path / ".venv/bin/python").symlink_to(sys.executable)
        (arm.path / ".yv-own-venv").touch()
    assert go(world, suite="quality", mmlu_n=2) == 0
    records = [json.loads(p.read_text()) for p in (world.tmp / "jobs").glob("*.json")]
    for record in records:
        command = record["cmd"]
        tree = Path(
            next(x.split("=", 1)[1] for x in command if x.startswith("PAIRED_TREE="))
        )
        expected = str(tree / ".venv/bin/python")
        assert f"PAIRED_PY={expected}" in command
        assert expected in command


def test_marked_runtime_record_changes_arm_identity(tmp_path):
    bindir = tmp_path / ".venv/bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("python")
    (tmp_path / ".yv-own-venv").touch()
    record = tmp_path / ".venv/lib/python3.13/site-packages/mlx-0.32.4.dist-info/RECORD"
    record.parent.mkdir(parents=True)
    record.write_text("mlx/core.so,sha256=old,1\n")
    before = core.Arm("cand", "c", "a" * 40, tmp_path, "")
    key = before.key
    record.write_text("mlx/core.so,sha256=new,1\n")
    after = core.Arm("cand", "c", "a" * 40, tmp_path, "")
    assert after.key != key
    assert before.key == key  # a run freezes its installed-runtime receipt
    (tmp_path / ".yv-own-venv").unlink()
    unmarked = core.Arm("cand", "c", "a" * 40, tmp_path, "")
    assert unmarked.key == "a" * 12


def test_detach_retains_caller_verifier_implementation(monkeypatch, tmp_path):
    from verify import cli

    script = tmp_path / "worker" / "scripts" / "verify" / "cli.py"
    monkeypatch.setattr(cli, "__file__", str(script))
    assert cli.verifier_scripts() == script.parents[1]
    assert cli.verifier_scripts() != cli.REPO / "scripts"


def test_modelprobe_is_explicit_and_not_in_release_suites():
    assert suites.parse_suite("modelprobe")["stages"] == ["modelprobe"]
    assert "modelprobe" in stages.STAGE_FUNCS
    for suite in suites.SUITES.values():
        assert "modelprobe" not in suite["stages"]


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


def test_client_routes_validator_fails_closed(tmp_path):
    names = (
        "agent-custom-tools",
        "agent-shell-search",
        "agent-documents-citations",
        "agent-anthropic-client-tools",
        "agent-continuous-usage",
        "agent-template-props",
        "agent-http-video",
    )
    path = tmp_path / "routes.json"
    assert not stages.client_routes_valid(path)[0]
    data = {
        "complete": True,
        "pass": True,
        "checks": {n: {"status": "pass"} for n in names},
    }
    path.write_text(json.dumps(data))
    assert stages.client_routes_valid(path)[0]
    data["checks"]["agent-custom-tools"]["status"] = "fail"
    path.write_text(json.dumps(data))
    assert not stages.client_routes_valid(path)[0]


def test_client_compat_pilot_is_pinned_and_stops_before_second_model(
    world, monkeypatch
):
    from types import SimpleNamespace

    from verify.execute import CellResult

    calls = []

    class FakeExecutor:
        jobs = []

        def run_cells(self, cells):
            cell = cells[0]
            calls.append(cell)
            return {
                cell.key: CellResult(cell.key, False, "pilot failure", job="job-test")
            }

    ctx = SimpleNamespace(
        cand=SimpleNamespace(path=world.repo, key="cand"),
        suite={"client_compat_device": "m3"},
        exe=FakeExecutor(),
        run=SimpleNamespace(append=lambda *args: None),
        py=sys.executable,
    )
    result = stages.stage_client_compat(ctx)
    assert not result.passed and len(calls) == 1
    assert calls[0].cwd == world.repo and calls[0].device == "m3"
    assert "--tree-sha" in calls[0].argv and "Qwen3.5-0.8B" in " ".join(calls[0].argv)


def test_ttft_profile_32k_cells_use_the_short_lane_and_128k_does_not(tmp_path):
    from types import SimpleNamespace

    class StopError(Exception):
        pass

    cells = []

    class Capture:
        label = "t"

        def run_cells(self, submitted):
            cells.extend(submitted)
            raise StopError

    for size, limit_ok in ((32768, True), (131072, False)):
        cells.clear()
        ctx = stages.Ctx(
            run=None,
            exe=Capture(),
            base=SimpleNamespace(path=tmp_path / "base"),
            cand=SimpleNamespace(path=tmp_path / "cand"),
            env={},
            base_env={},
            cand_env={},
            model="tiny",
            model_name="tiny",
            suite={"ctx": [size], "kinds": ["code"]},
            mem_gb=60,
        )
        with pytest.raises(StopError):
            stages.stage_ttft_profile(ctx)
        assert (max(c.timeout_min for c in cells) <= 10) is limit_ok


def test_qwen4_tiny_verdict_rejects_incomplete_or_unengaged_evidence(tmp_path):
    evidence = tmp_path / "native.jsonl"
    checks = [{"check": f"spec-depth-{d}", "passed": True} for d in (2, 3, 4)]
    checks += [{"check": f"rollback-{k}", "passed": True} for k in (1, 2, 3, 4)]
    good = {
        "complete": True,
        "passed": True,
        "device": "m5",
        "engaged": "qwen4-native",
        "checks": checks,
    }
    evidence.write_text(json.dumps(good) + "\n")
    assert stages._qwen4_mtp_valid(evidence)[0]
    for bad in (
        {"complete": False},
        {"dry_run": True},
        {"device": "m3"},
        {"engaged": "off"},
        {"checks": checks[:-1]},
        {"checks": [dict(c, passed=False) for c in checks]},
    ):
        evidence.write_text(json.dumps({**good, **bad}) + "\n")
        assert not stages._qwen4_mtp_valid(evidence)[0]


@pytest.mark.parametrize("stage", ["identity", "speed"])
def test_spec_mode_guard_rejects_mixed_or_missing_engagement(stage):
    base = [{"part": "session", "engaged_spec_mode": "dflash"}]
    cand = [{"part": "session", "engaged_spec_mode": "mtp"}]
    assert stages.spec_mode_failures(base, cand, {}, stage)
    assert stages.spec_mode_failures(base, [], {}, stage)
    assert not stages.spec_mode_failures(base, base, {}, stage)
    assert not stages.spec_mode_failures(
        base, cand, {"allow_spec_mode_change": True}, stage
    )
    # Explicit opt-in does not waive missing evidence.
    assert stages.spec_mode_failures(base, [], {"allow_spec_mode_change": True}, stage)


@pytest.mark.parametrize("suite", ["identity", "speed"])
def test_real_stage_rejects_cross_spec_modes_with_identical_outputs(world, suite):
    assert go(world, suite=suite, cand_env=["YUNSHU_VLM_DRAFT=off"]) == 1
    v, _ = verdict_of(world)
    assert v["stages"][0]["status"] == "FAIL"
    assert any("engaged spec mode mismatch" in r for r in v["stages"][0]["reasons"])


def test_apc_stage_checks_every_spec_mode(tmp_path):
    import json
    from types import SimpleNamespace

    good, bad = rows(cached=1024), rows(cached=1024)
    bad[1]["sha"] = "zzz"  # MTP warm != cold; the later (good) mode must not mask it
    ev = {"mtp": bad, "dflash": good}

    def cell_path(stage, key):
        p = tmp_path / f"{stage}.{key}.jsonl"
        mode = key.split(".")[1]
        p.write_text("".join(json.dumps(r) + "\n" for r in ev[mode]))
        return p

    ctx = SimpleNamespace(
        suite={"spec_modes": ["mtp", "dflash"], "ctx": [1024], "kinds": ["code"]},
        run=SimpleNamespace(cell_path=cell_path, append=lambda *a: None),
        exe=SimpleNamespace(jobs=[]),
    )
    res = stages.stage_apc(ctx)
    assert not res.passed


# ── fail-closed evidence reuse (gate fingerprint, share keys, cached cells, planned grids) ──
def _model_dir(tmp_path, name="m", cfg="{}"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(cfg)
    (d / "model.safetensors").write_bytes(b"w" * 8)
    return d


def test_gate_reuse_requires_matching_model_env_and_clean_tree(gate_world, tmp_path):
    w = gate_world
    stages_ = ["install", "serve-27b"]
    model = _model_dir(tmp_path)
    env = {"M": str(model), "AGENT_TURNS": "4"}
    assert run_gate(w, stages=stages_, extra_env=env) == 0
    n = len(jobs(w))
    assert run_gate(w, stages=stages_, extra_env=env) == 0
    assert len(jobs(w)) == n  # same fingerprint: reused
    (model / "config.json").write_text('{"changed": 1}')  # replaced checkpoint
    assert run_gate(w, stages=stages_, extra_env=env) == 0
    assert len(jobs(w)) == n + 2
    n = len(jobs(w))
    assert run_gate(w, stages=stages_, extra_env={**env, "AGENT_TURNS": "9"}) == 0
    assert len(jobs(w)) == n + 2  # a gate parameter changed
    n = len(jobs(w))
    (w.repo / "scripts/release/gate.sh").write_text(FAKE_GATE + "\n# edited\n")
    assert run_gate(w, stages=stages_, extra_env={**env, "AGENT_TURNS": "9"}) == 0
    assert len(jobs(w)) == n + 2  # an uncommitted edit reruns
    assert run_gate(w, stages=stages_, extra_env={**env, "AGENT_TURNS": "9"}) == 0
    assert len(jobs(w)) == n + 2  # the same edit is the same tree: reused
    (w.repo / "scripts/release/gate.sh").write_text(FAKE_GATE + "\n# edited twice\n")
    assert run_gate(w, stages=stages_, extra_env={**env, "AGENT_TURNS": "9"}) == 0
    assert len(jobs(w)) == n + 4  # a different edit reruns
    git(w.repo, "checkout", "scripts/release/gate.sh")


def test_gate_partial_run_checks_install_marker(gate_world):
    assert run_gate(gate_world, stages=["install", "serve-27b"]) == 0
    marker = gate_world.tmp / "gateroot/.yv-install-commit"
    marker.write_text("deadbeef")
    n = len(jobs(gate_world))
    assert run_gate(gate_world, stages=["serve-27b"]) == 0
    assert len(jobs(gate_world)) == n + 1  # no reuse on a stale install


def _share_ctx(tmp_path, monkeypatch, py=None):
    from types import SimpleNamespace

    arm = core.Arm("base", "b", "a" * 40, tmp_path / "arm", "")
    return SimpleNamespace(
        tree=lambda a: arm,
        arm_env=lambda a, e=None: dict(e or {}),
        model=str(_model_dir(tmp_path)),
        suite={"kinds": ["prose"]},
        py=py or str(tmp_path / "venv/bin/python"),
        base=arm,
    )


def test_identity_share_key_tracks_prompt_content_checkpoint_and_packages(
    tmp_path, monkeypatch
):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "prose-1024.txt").write_text("one")
    monkeypatch.setattr(stages, "PROMPTS", prompts)
    site = tmp_path / "venv/lib/python3.13/site-packages"
    (site / "mlx-0.1.0.dist-info").mkdir(parents=True)
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/python").write_text("")
    ctx = _share_ctx(tmp_path, monkeypatch)

    def key():
        return stages._share_key(ctx, "base", {}, [1024])

    k0 = key()
    (prompts / "prose-1024.txt").write_text("two")  # regenerated in place
    k1 = key()
    assert k1 != k0
    (Path(ctx.model) / "config.json").write_text('{"v": 2}')
    k2 = key()
    assert k2 != k1
    (site / "mlx-0.2.0.dist-info").mkdir()
    assert key() != k2


def test_memory_share_key_tracks_harness_bytes(tmp_path, monkeypatch):
    harness = tmp_path / "memory_ab.py"
    harness.write_text("v1")
    monkeypatch.setattr(stages, "MEMORY_AB", harness)
    ctx = _share_ctx(tmp_path, monkeypatch)
    ctx.suite = {"mem_reps": 1, "mem_sizes": [1024]}
    ctx.mem_gb = 8
    ctx.big = False
    keys = []

    class CapturedStopError(Exception):
        pass

    class Capture:
        def run_cells(self, cells):
            keys.extend(c.share_key for c in cells if c.share_key)
            raise CapturedStopError

    ctx.exe = Capture()
    for text in ("v1", "v2"):
        harness.write_text(text)
        with pytest.raises(CapturedStopError):
            stages.stage_memory(ctx)
    assert len(keys) == 2 and keys[0] != keys[1]


def test_cached_cell_is_revalidated(world):
    rd, ex = mkexec(world, "cached")
    cell = Cell("s", "c1", ["true"])
    ev = rd.cell_path("s", "c1")
    ev.write_text('{"complete": true}\n')
    rd.append(
        "s", {"ev": "cell_done", "cell": "c1", "ok": True, "sig": cell.sig, "job": "j"}
    )
    assert ex._cached(cell) is not None
    ev.write_text('{"part": "decode"}\n')  # truncated / edited after the fact
    assert ex._cached(cell) is None
    cell2 = Cell("s", "c2", ["true"], validate=lambda p: (False, "bad"))
    rd.cell_path("s", "c2").write_text('{"complete": true}\n')
    rd.append(
        "s", {"ev": "cell_done", "cell": "c2", "ok": True, "sig": cell2.sig, "job": "j"}
    )
    assert ex._cached(cell2) is None


def test_identity_and_speed_check_the_planned_grid():
    planned = analyze.planned_identity([1024, 131072], ["code"])
    only_small = rows(ctx=1024)  # the 128K records are lost from both arms
    assert analyze.compare_identity(only_small, only_small)["ok"]  # observed grid only
    r = analyze.compare_identity(only_small, only_small, planned=planned)
    assert not r["ok"] and any("131072" in str(m["cell"]) for m in r["mismatches"])
    nosha = [{**x, "sha": None, "ct": None} for x in rows()]
    r = analyze.compare_identity(nosha, nosha)
    assert not r["ok"] and "missing" in r["mismatches"][0]["why"]
    base = [rows(ctx=1024) for _ in range(3)]
    assert analyze.speed_compare(base, base)["ok"]
    sp = analyze.speed_compare(base, base, planned=[(1024, "code"), (131072, "code")])
    assert not sp["ok"] and any("131072" in m for m in sp["missing"])


def test_ssd_ttft_profile_requires_tiers_and_cold_ram_ssd_parity(tmp_path):
    path = tmp_path / "ssd.jsonl"
    rows = [
        dict(
            phase=p,
            sha="equal",
            cached=1,
            xy={"cache": {"tier": tier}},
            call_spans=[{"stage": "ssd_read_validate"}],
        )
        for p, tier in (
            ("cold", "none"),
            ("warm", "ram"),
            ("spill", "none"),
            ("ssd", "ssd"),
            ("warm", "ram"),
            ("followup", "ram"),
        )
    ]
    rows[-1]["rehearsal_usage"] = {"prompt_tokens": 10}
    rows.append({"complete": True})

    def valid():
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return stages._ssd_ttft_profile_valid(path)[0]

    assert valid()
    rows[3]["xy"]["cache"]["tier"] = "ram"
    assert not valid()
    rows[3]["xy"]["cache"]["tier"] = "ssd"
    rows[3]["sha"] = "different"
    assert not valid()


def test_ssd_matrix_stage_validator_rejects_rehearsal_drift(tmp_path):
    path = tmp_path / "matrix.jsonl"
    rows = [
        dict(ctx=4096, phase=p, sha="same", cached=10, xy={"cache": {"tier": t}})
        for p, t in (
            ("cold", "none"),
            ("ram_before_restart", "ram"),
            ("ssd", "ssd"),
            ("ram_after_restart", "ram"),
            ("followup", "ram"),
            ("followup_rehearsed", "ram"),
        )
    ]
    rows.insert(
        0,
        dict(
            event="next_turn_proof",
            completed_delta=1,
            cached=2048,
            source_prompt_tokens=1024,
        ),
    )
    rows.append({"complete": True})

    def valid():
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return stages._ssd_ttft_valid(path)[0]

    assert valid()
    rows[-2]["sha"] = "drift"
    assert not valid()


def test_ssd_stages_are_explicitly_selectable_but_not_default_ladder():
    for stage in ("ssd_ttft", "ssd_ttft_profile"):
        assert suites.parse_suite(stage)["stages"] == [stage]
        assert stage not in suites.LADDER


def test_lane_bandwidth_stage_coverage_fails_closed(tmp_path):
    assert suites.parse_suite("lane_bandwidth")["stages"] == ["lane_bandwidth"]
    path = tmp_path / "bandwidth.jsonl"
    cells = [
        dict(
            phase="timing",
            kind=kind,
            name="projection",
            bits=bits,
            rows=count,
            ms=1.0,
            gbps=100.0,
            bytes=100000000,
        )
        for kind, bits in [("target", 4), ("draft", 8)]
        for count in (1, 8, 16, 17, 32)
    ]

    def write(rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write(cells + [{"complete": True}])
    assert stages._lane_bandwidth_valid(path)[0]
    write(cells)
    assert not stages._lane_bandwidth_valid(path)[0]
    write(cells[:-1] + [{"complete": True}])
    assert not stages._lane_bandwidth_valid(path)[0]
    cells[0]["ms"] = float("nan")
    write(cells + [{"complete": True}])
    assert not stages._lane_bandwidth_valid(path)[0]


def test_spec_depth_check_accepts_the_engaged_mode_for_default_suites():
    # console-identity runs spec_modes ["default"]; the server reports the mode it engaged (dflash).
    spec = {
        "mode": "dflash",
        "drafted": 4,
        "accepted": 2,
        "rounds": 2,
        "per_depth": [{"position": 0, "drafted": 4, "accepted": 2}],
    }
    rows = [{"xy": {"speculative": spec}}]
    assert stages._spec_depth_valid(rows, "default")[0]
    assert not stages._spec_depth_valid(rows, "mtp")[0]
    assert not stages._spec_depth_valid(
        [{"xy": {"speculative": dict(spec, rounds=0)}}], "default"
    )[0]


def test_model_size_comes_from_the_weights_not_the_name(tmp_path):
    # Flash-Next has no "27B" in its name but 77 GB of weights; it must be sized as a big model
    model = tmp_path / "Qwen3.8-Flash-Next"
    model.mkdir()
    with open(model / "model-00001.safetensors", "wb") as f:
        f.truncate(25 * 10**9)  # sparse file: no disk use
    assert runner._weights_gb(str(model)) == pytest.approx(25.0)
    assert runner._weights_gb(str(tmp_path / "missing")) == 0.0
