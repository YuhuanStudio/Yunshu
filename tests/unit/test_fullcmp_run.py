"""Unit tests for scripts/research/fullcmp_run.py (batch driver: grouping, rotation, gates, rep 3).

Fakes only: no gpuq call, no GPU, no subprocess. Manifest rows and the pin are synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fullcmp_pin_io import stub_pin_io  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_io(monkeypatch):
    stub_pin_io(monkeypatch)


@pytest.fixture(autouse=True)
def _provenance_gate_passes(monkeypatch):
    """The fake PIN has no real Yunshu tree or binaries, so the census/binary provenance gate would
    refuse every batch here. The gate itself (real census, refusal naming the engine) is tested in
    test_fullcmp_release_dates.py."""
    import fullcmp_provenance as _fp

    monkeypatch.setattr(_fp, "dry_run_problems", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _never_reach_gpuq(monkeypatch):
    """No test in this module may queue a job or run the real gpuq: any such call fails the test."""
    import fullcmp_run as _run

    def _refuse(name):
        def _fail(*a, **k):
            pytest.fail(f"{name} reached the real gpuq from a unit test")

        return _fail

    for name in ("swap_gate", "submit_all", "wait_all", "run_audit", "run_reps"):
        monkeypatch.setattr(_run, name, _refuse(name))


ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_run", ROOT / "scripts" / "research" / "fullcmp_run.py"
)
run = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_run"] = run
_SPEC.loader.exec_module(run)


@pytest.fixture(autouse=True)
def _no_end_of_batch_rerun(monkeypatch):
    """The fake submits of these tests write no cell files, so every cell would count as failed and be rerun. The
    end-of-batch rerun itself is tested below with real files (test_rerun_failed_cells_*)."""
    monkeypatch.setattr(
        run, "rerun_failed_cells", lambda plan, pin, submit=None: ([], 0)
    )


@pytest.fixture(autouse=True)
def quiet_host_stub(monkeypatch):
    """The batch-level quiet gate samples the real host (10 s of ps). Stub it: quiet by default;
    tests that need a busy host override gate themselves."""
    monkeypatch.setattr(
        run.fullcmp_quiet,
        "gate",
        lambda **kw: {"passed": True, "total_foreign_pct": 0.0, "holders": []},
    )


PIN = {
    "sha": "0f2960047855546a9f13b2631b680352b1c57cfe",
    "version": "0.1.7.dev0",
    "python_executable": "/fake/venv/bin/python",
    "launch": {
        "harness_env": {
            "TFB_YUNSHU_BIN": "/fake/venv/bin/yunshu",
            "TFB_YUNSHU_SRC": "/fake/tree/python",
            "PYTHONPATH": "/fake/tree/python",
        }
    },
}


def row(
    engine,
    cid,
    est=2.0,
    reps=2,
    batch="B1",
    status="unverified",
    ctx=1024,
    ctx_tokens=1024,
    conc=1,
    kind="code",
    out=512,
    thinking="off",
    table="decode",
    media="none",
    turns="single",
    sampling="greedy",
    reason="ok",
):
    return {
        "batch": batch,
        "conc": conc,
        "ctx": ctx,
        "ctx_tokens": ctx_tokens,
        "engine": engine,
        "engine_version": "x",
        "est_gpu_min": est,
        "id": cid,
        "media": media,
        "model": "q27-4bit",
        "out_len": out,
        "phases": ["cold", "warm", "fu"],
        "prompt_kind": kind,
        "reason": reason,
        "reps": reps,
        "sampling": sampling,
        "status": status,
        "sweep": "core",
        "table": table,
        "thinking": thinking,
        "turns": turns,
    }


@pytest.fixture
def pin_file(tmp_path, monkeypatch):
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(PIN))
    monkeypatch.setattr(run, "PIN_PATH", p)
    monkeypatch.setattr(run, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(run, "PREFLIGHT_DIR", tmp_path / "preflight")
    return p


# --- grouping and the 20-minute split -------------------------------------


def test_group_splits_at_the_70_percent_limit_per_rep():
    # every chunk's itemized job_budget total stays within 14 min (70% of the 20 min cap)
    rows = [row("yunshu", f"B1.c{i}", est=10.0) for i in range(6)]
    chunks = run.pack_chunks(rows)
    assert len(chunks) >= 2
    assert sorted(r["id"] for c in chunks for r in c) == sorted(r["id"] for r in rows)
    assert all(run.job_budget(c)["total"] <= run.JOB_LIMIT_MIN for c in chunks)
    assert run.JOB_LIMIT_MIN == 0.7 * run.JOB_CAP_MIN == 14.0


def test_group_whose_measured_sum_exceeds_the_limit_is_split():
    # a group whose summed budget exceeds the limit is split; a group under it is not
    small = [row("yunshu", f"B1.a{i}", est=2.0) for i in range(2)]
    assert len(run.pack_chunks(small)) == 1
    big = [row("yunshu", f"B1.b{i}", est=6.0) for i in range(6)]
    assert run.job_budget(big)["total"] > run.JOB_LIMIT_MIN
    chunks = run.pack_chunks(big)
    assert len(chunks) > 1
    assert all(run.job_budget(c)["total"] <= run.JOB_LIMIT_MIN for c in chunks)


def test_yunshu_220358_shape_splits_into_chunks_within_the_limit():
    # the 10 core cells of job 1103-220358 (yunshu, 27B, conc 1) from the real manifest: the old
    # single job (13.5 min estimate, no per-cell start) was killed at the 20 min cap
    import fullcmp_manifest as fm_

    rows, _meta = fm_.build_rows(engines=["yunshu"])
    core = [
        r
        for r in rows
        if r["batch"] == "B1"
        and r["ctx"] in fm_.CORE_CTX
        and r["status"] == "unverified"
    ]
    assert len(core) == 10
    chunks = run.pack_chunks(core)
    assert len(chunks) >= 2
    assert sorted(r["id"] for c in chunks for r in c) == sorted(r["id"] for r in core)
    for c in chunks:
        b = run.job_budget(c)
        assert b["total"] <= 0.7 * run.JOB_CAP_MIN, (b["total"], [r["id"] for r in c])
        assert b["timeout"] <= run.JOB_CAP_MIN


def test_timeout_is_1_3_x_total_floored_by_the_same_kind_and_capped():
    rows = [row("yunshu", "B1.t", est=2.0)]
    b = run.job_budget(rows)
    assert b["timeout"] == pytest.approx(
        min(
            run.SINGLE_CELL_CAP_MIN,
            __import__("math").ceil(round(1.3 * b["total"] * 10, 6)) / 10,
        )
    )
    # floored by the longest same-kind run (the killed 1103-220358 job ran 20 min)
    # a single cell's timeout follows 1.3 x prior up to SINGLE_CELL_CAP_MIN; a multi-cell job stays at the cap
    assert run.job_budget(rows, prior_min=20.0)["timeout"] == 26.0
    assert run.job_budget(rows * 2, prior_min=20.0)["timeout"] == run.JOB_CAP_MIN
    assert run.job_budget(rows, prior_min=0.0)["timeout"] < 20.0
    assert run.prior_wall_min("B1", "yunshu") >= 20.0
    assert run.prior_wall_min("B9", "yunshu") == 0.0


def test_plan_labels_split_groups_and_keeps_one_group_per_server_config():
    rows = [row("yunshu", f"B1.c{i}", est=10.0) for i in range(6)]
    rows.append(row("yunshu", "B1.think", est=2.0, thinking="on"))
    plan = run.plan_batch(rows, "B1", PIN)
    # thinking=on is a runnable tfbench cell now (the family path): planned, never skipped
    assert "B1.think" in {r["id"] for r in plan.runnable}
    assert not any(r["id"] == "B1.think" for r, _why in plan.skipped)
    rep0 = [j for j in plan.jobs if j.rep == 0]
    labels = [j.label for j in rep0]
    assert "fullcmp-B1-yunshu-q27-4bit-th-off-p1-r0" in labels
    assert "fullcmp-B1-yunshu-q27-4bit-th-off-p2-r0" in labels
    assert all(j.est_min <= 20.0 for j in rep0)


def test_single_cell_above_cap_is_flagged():
    rows = [
        row("yunshu", "B1.big", est=80.0)
    ]  # 40 min per rep: 1.3 x 40 > SINGLE_CELL_CAP_MIN
    plan = run.plan_batch(rows, "B1", PIN)
    assert plan.jobs and all(j.over_cap for j in plan.jobs)


def test_single_cell_between_job_cap_and_single_cell_cap_runs_alone():
    """real-1104 B2: mixed 128K c8 cells budget ~22 min; one unsplittable cell runs with its own 1.3x timeout."""
    rows = [
        row("yunshu", "B1.mixed128", est=45.0),
        row("yunshu", "B1.short", est=2.0),
    ]  # 22.5 min per rep
    plan = run.plan_batch(rows, "B1", PIN)
    rep0 = [j for j in plan.jobs if j.rep == 0]
    big = next(j for j in rep0 if "B1.mixed128" in j.cell_ids)
    assert big.cell_ids == ["B1.mixed128"] and not big.over_cap
    assert run.JOB_CAP_MIN < big.timeout_min <= run.SINGLE_CELL_CAP_MIN
    assert not any(j.over_cap for j in plan.jobs)


def test_single_cell_above_the_margin_runs_alone_and_is_flagged():
    rows = [row("yunshu", "B1.long", est=30.0), row("yunshu", "B1.short", est=2.0)]
    plan = run.plan_batch(rows, "B1", PIN)
    rep0 = [j for j in plan.jobs if j.rep == 0]
    long_job = next(j for j in rep0 if "B1.long" in j.cell_ids)
    assert long_job.cell_ids == ["B1.long"]  # alone: the next cell does not fit with it
    assert long_job.over_margin and not long_job.over_cap
    assert long_job.timeout_min == pytest.approx(
        26.1
    )  # alone: its own 1.3 x budget, not the 20 min cap


# --- rotated order across reps ---------------------------------------------


def test_rotated_order_rep_r_starts_at_engine_r_mod_n():
    engines = ["yunshu", "tf-1.2.1", "splash-1.5.0"]
    rows = [row(e, f"B1.{e}", reps=3) for e in engines]
    plan = run.plan_batch(rows, "B1", PIN)
    assert plan.engine_order == engines
    for rep in range(3):
        firsts = [j.engine for j in plan.jobs if j.rep == rep]
        assert firsts[0] == engines[rep % 3]
        assert firsts == run.rotate(engines, rep)


def test_no_back_to_back_same_engine_across_reps_when_three_engines():
    engines = ["yunshu", "tf-1.2.1", "splash-1.5.0"]
    rows = [row(e, f"B1.{e}", reps=3) for e in engines]
    plan = run.plan_batch(rows, "B1", PIN)
    assert run.back_to_back(plan.jobs) == 0


def test_rotate_is_left_rotation():
    assert run.rotate(["A", "B", "C", "D"], 1) == ["B", "C", "D", "A"]
    assert run.rotate(["A", "B"], 2) == ["A", "B"]


# --- skip unsupported and unmapped rows -----------------------------------


def test_harness_stub_engine_row_is_skipped_not_planned():
    """lmstudio is 'unverified' in the manifest but a harness stub (bench_engines.UNSUPPORTED_ENGINES):
    planning used to call tfbench_engine on it and crash the whole dry run with a KeyError."""
    rows = [row("yunshu", "B1.ok"), row("lmstudio", "B1.lms")]
    plan = run.plan_batch(rows, "B1", PIN)
    assert [r["id"] for r in plan.runnable] == ["B1.ok"]
    reasons = {r["id"]: why for r, why in plan.skipped}
    assert reasons["B1.lms"].startswith("unsupported (harness:")
    assert {c for j in plan.jobs for c in j.cell_ids} == {"B1.ok"}


def test_unsupported_rows_are_skipped_and_listed():
    rows = [
        row("ollama", "B1.ollama", status="unsupported", reason="n/a: no checkpoint"),
        row("yunshu", "B1.ok"),
        row("yunshu", "B1.img", media="img1"),
        row("yunshu", "B1.json", kind="json-schema"),
    ]
    plan = run.plan_batch(rows, "B1", PIN)
    # image and json-schema rows are runnable tfbench cells now; only the unsupported row is skipped
    assert [r["id"] for r in plan.runnable] == ["B1.ok", "B1.img", "B1.json"]
    reasons = {r["id"]: why for r, why in plan.skipped}
    assert "unsupported" in reasons["B1.ollama"]
    assert set(reasons) == {"B1.ollama"}
    assert {c for j in plan.jobs for c in j.cell_ids} == {"B1.ok", "B1.img", "B1.json"}


def test_other_batches_are_ignored():
    rows = [row("yunshu", "B2.x", batch="B2")]
    plan = run.plan_batch(rows, "B1", PIN)
    assert plan.runnable == [] and plan.jobs == []


# --- command lines ----------------------------------------------------------


def test_command_lines_carry_fullcmp_env_and_pin():
    rows = [row("yunshu", "B1.ok")]
    plan = run.plan_batch(rows, "B1", PIN)
    job = plan.jobs[0]
    text = job.script
    assert "TFB_FULLCMP=1" in text
    assert "TFB_YUNSHU_BIN=/fake/venv/bin/yunshu" in text
    assert "TFB_YUNSHU_SRC=/fake/tree/python" in text
    assert "PYTHONPATH=/fake/tree/python" in text
    assert f"TFB_TELEMETRY_DIR={run.RAW_ROOT}" in text
    assert f"HOME={run.HOME_ROOT}/yunshu" in text
    assert "--engine yunshu-main" in text and "--part decode" in text
    assert "GPUQ_JOB_ID" not in text  # gpuq exports it to the job env
    assert f"{run.RAW_ROOT}/yunshu/B1.ok-r0.jsonl" in text


def test_telemetry_dir_is_the_cell_file_root():
    """The harness telemetry path for a cell is the audit's telemetry_path beside the cell file."""
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "fullcmp_audit_t", ROOT / "scripts" / "research" / "fullcmp_audit.py"
    )
    audit = ilu.module_from_spec(spec)
    sys.modules["fullcmp_audit_t"] = audit
    spec.loader.exec_module(audit)
    raw = run.raw_path("yunshu", "B1.ok", 0)
    root = run.RAW_ROOT
    tdir = [
        t for t in run.cell_env("yunshu", PIN) if t.startswith("TFB_TELEMETRY_DIR=")
    ][0]
    tdir = tdir.split("=", 1)[1]
    assert Path(tdir) == root
    harness_path = (
        Path(root) / "yunshu" / (raw.name[: -len(".jsonl")] + ".telemetry.jsonl")
    )
    assert harness_path == audit.telemetry_path(raw)


def test_submit_argv_form():
    rows = [row("yunshu", "B1.ok")]
    job = run.plan_batch(rows, "B1", PIN).jobs[0]
    argv = job.submit_argv()
    assert argv[1:3] == ["submit", "--label"]
    assert argv[3] == "fullcmp-B1-yunshu-q27-4bit-th-off-r0"
    assert argv[4:6] == ["--timeout", f"{job.timeout_min:g}"]
    assert argv[6:8] == ["--stall", f"{job.stall_min:g}"]
    assert argv[8:12] == ["--priority", "1", "--quiet", "--"]
    assert argv[12:] == job.cmd


# --- dry run touches nothing ------------------------------------------------


def test_dry_run_touches_no_gpu_or_queue(pin_file, tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r) for r in [row("yunshu", "B1.ok")]) + "\n"
    )

    def boom(*_a, **_k):
        raise AssertionError("dry-run must not call gpuq or subprocess")

    monkeypatch.setattr(run, "gpuq_status_text", boom)
    monkeypatch.setattr(run, "submit_all", boom)
    monkeypatch.setattr(run, "swap_gate", boom)
    monkeypatch.setattr(run, "wait_all", boom)
    monkeypatch.setattr(run.subprocess, "run", boom)
    # the engine-table check builds the pins (its own subprocesses); this test is about the queue
    import fullcmp_engine_tables as fet

    monkeypatch.setattr(fet, "check_all", lambda ids, ctx=None: [])
    rc = run.main(["--batch", "B1", "--manifest", str(manifest), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "JOB fullcmp-B1-yunshu-q27-4bit-th-off-r0" in captured.out
    out = captured.out
    assert "TOTAL B1: 2 jobs" in out
    assert not (tmp_path / "audit").exists()


# --- exclusive GPU refusal --------------------------------------------------

STATUS_FOREIGN = """daemon: up
running  device=m5  100s  p1  1102-120000-00-wait_fn-ple-memory
pending  device=m5  5s  p1  1102-120100-00-fullcmp-B1-yunshu-q27-4bit-th-off-r0
"""

STATUS_OWN = """daemon: up
running  device=m5  100s  p1  fullcmp-B1-yunshu-q27-4bit-th-off-r0
done  device=m5  9s  p1  fullcmp-B1-yunshu-q27-4bit-th-off-r1
"""


def test_parse_status_and_foreign_detection():
    assert run.foreign_jobs(STATUS_FOREIGN) == [
        ("running", "1102-120000-00-wait_fn-ple-memory")
    ]
    assert run.foreign_jobs(STATUS_OWN) == []


def test_refuses_with_foreign_job(pin_file, tmp_path, monkeypatch):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row("yunshu", "B1.ok")) + "\n")
    submitted = []
    monkeypatch.setattr(run, "gpuq_status_text", lambda: STATUS_FOREIGN)
    # a refusal must come before any GPU work: the swap gate would queue a real job
    monkeypatch.setattr(run, "swap_gate", lambda b: pytest.fail("swap gate ran"))
    monkeypatch.setattr(
        run, "submit_all", lambda jobs: submitted.append(jobs) or ([], 0)
    )
    rc = run.main(["--batch", "B1", "--manifest", str(manifest)])
    assert rc == 2
    assert submitted == []


def test_allow_foreign_overrides_refusal(pin_file, tmp_path, monkeypatch):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row("yunshu", "B1.ok")) + "\n")
    monkeypatch.setattr(run, "gpuq_status_text", lambda: STATUS_FOREIGN)
    monkeypatch.setattr(run, "swap_gate", lambda b: 0)
    monkeypatch.setattr(run, "_submit_and_wait", lambda job: ("id-1", 0, "done"))
    monkeypatch.setattr(run, "run_audit", lambda root, mpath: 0)
    monkeypatch.setattr(run, "run_reps", lambda root, mpath: [])
    assert (
        run.main(["--batch", "B1", "--manifest", str(manifest), "--allow-foreign"]) == 0
    )


# --- rep-3 queue from fake reps output --------------------------------------


def test_parse_need_rep3_ignores_other_status_lines():
    out = "\n".join(
        [
            json.dumps({"engine": "yunshu", "cell": "B1.a", "status": "need_rep3"}),
            json.dumps({"engine": "omlx-0.7.3", "cell": "B1.b", "status": "stable"}),
            "not json",
            json.dumps({"engine": "omlx-0.7.3", "cell": "B1.c", "status": "need_rep3"}),
        ]
    )
    assert run.parse_need_rep3(out) == [("yunshu", "B1.a"), ("omlx-0.7.3", "B1.c")]


def test_rep3_jobs_use_rotated_slots_and_flag(pin_file):
    engines = ["yunshu", "tf-1.2.1", "splash-1.5.0"]
    rows = [row(e, f"B1.{e}") for e in engines]
    subset = [r for r in rows if r["engine"] in ("tf-1.2.1", "splash-1.5.0")]
    jobs = run.plan_jobs(
        {run.REP3_INDEX: subset}, engines, PIN, "B1", label_suffix="-rep3"
    )
    assert run.rotate(engines, 2) == ["splash-1.5.0", "yunshu", "tf-1.2.1"]
    assert [j.engine for j in jobs] == ["splash-1.5.0", "tf-1.2.1"]
    assert all(j.label.endswith("-r2-rep3") for j in jobs)
    assert all(j.rep == 2 for j in jobs)
    assert all("-r2.jsonl" in j.script for j in jobs)


def test_full_run_queues_rep3_for_flagged_cells_and_fails_on_bad_job(
    pin_file, tmp_path, monkeypatch
):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(row("yunshu", "B1.a"))
        + "\n"
        + json.dumps(row("omlx-0.7.3", "B1.b"))
        + "\n"
    )
    calls = {"submit": [], "wait": [], "reps": 0}
    monkeypatch.setattr(run, "gpuq_status_text", lambda: "")
    monkeypatch.setattr(run, "swap_gate", lambda b: 0)

    def fake_one(job):
        calls["submit"].append([job.label])
        calls["wait"].append(job.label)
        rc = calls["rc"]
        return f"id-{len(calls['submit'])}", rc, ("done" if rc == 0 else "failed")

    calls["rc"] = 0
    monkeypatch.setattr(run, "_submit_and_wait", fake_one)
    monkeypatch.setattr(run, "run_audit", lambda root, mpath: 0)

    def fake_reps(root, mpath):
        calls["reps"] += 1
        return [("yunshu", "B1.a")] if calls["reps"] == 1 else []

    monkeypatch.setattr(run, "run_reps", fake_reps)
    rc = run.main(["--batch", "B1", "--manifest", str(manifest)])
    assert rc == 0
    assert len(calls["submit"]) == 5  # 2 engines x 2 reps, then the rep-3 cell last
    assert calls["submit"][-1] == ["fullcmp-B1-yunshu-q27-4bit-th-off-r2-rep3"]
    manifest_out = json.loads((run.AUDIT_DIR / "B1_audit_manifest.json").read_text())
    assert {c["cell"] for c in manifest_out["cells"]} == {"B1.a", "B1.b"}

    # a failed wait (nonzero rc) makes the whole batch exit nonzero
    calls["reps"] = 0
    calls["rc"] = 1
    assert run.main(["--batch", "B1", "--manifest", str(manifest)]) == 1


def test_audit_manifest_carries_census_pins_for_every_engine(pin_file):
    rows = [row("yunshu", "B1.a"), row("omlx-0.7.3", "B1.b")]
    m = run.audit_manifest(rows, ["B1"], PIN, ["yunshu", "omlx-0.7.3"])
    assert {"yunshu", "omlx-0.7.3"} <= set(m["pins"])
    assert m["pins"]["omlx-0.7.3"]["install_path"]
    assert m["groups"]["B1.1024.code.c1"]["order"] == ["yunshu", "omlx-0.7.3"]
    assert all(isinstance(c["expected_records"], int) for c in m["cells"])  # never None


def test_real_matrix_rows_have_expected_record_counts():
    # regression 2026-11-03: real-matrix rows returned None, so record_count failed every cell
    # ("expected None"); a B1 decode cell has one record per phase per rep (cold, warm, turn2)
    rows, _meta = run.fm.build_rows(engines=["yunshu"])
    b1 = next(
        r
        for r in rows
        if r["batch"] == "B1" and r["ctx"] == "1K" and r["prompt_kind"] == "code"
    )
    assert run.expected_records(b1) == 3  # per rep file
    conc = next(
        r
        for r in rows
        if r["batch"] == "B2" and r["sweep"] == "conc" and r["conc"] == 4
    )
    assert run.expected_records(conc) == 4 * run.CONC_TRIALS


# --- itemized job budget: startup, gate, cells, post, finalize, shutdown; stall; lane ----------


def test_slow_start_engine_gets_a_long_timeout_and_no_short_lane(monkeypatch):
    # ready_s 90 s (longest seen): startup = 1.5 x 90 s = 2.25 min per cell, two 1K cells
    monkeypatch.setitem(run.JB["startup_seen_s"], "mlx-vlm-server", {"value": 90.0})
    rows = [row("mlx-vlm-server", f"B1.s{i}", est=2.0) for i in range(2)]
    b = run.job_budget(rows)
    assert b["startup"] == pytest.approx(2 * 2.25)
    assert b["timeout"] >= 1.3 * (2 * 2.25 + run.GATE_PRE_MIN) - 0.05
    assert b["timeout"] >= 1.3 * b["total"] - 0.05
    assert b["lane"] == "normal" and b["timeout"] > run.SHORT_LANE_MIN
    assert b["stall"] >= max(5.0, 2 * 2.25)


def test_engine_default_is_a_floor_over_the_seen_ready(monkeypatch):
    # ready_s 8 s on disk still gets the mlx-vlm default floor (90 s); 1.5 x seen is only above it
    monkeypatch.setitem(run.JB["startup_seen_s"], "mlx-vlm-server", {"value": 8.0})
    assert run.startup_s("mlx-vlm-server") == 90.0
    monkeypatch.setitem(run.JB["startup_seen_s"], "mlx-vlm-server", {"value": 100.0})
    assert run.startup_s("mlx-vlm-server") == 150.0
    monkeypatch.delitem(run.JB["startup_seen_s"], "llama-cpp-b12061", raising=False)
    assert run.startup_s("llama-cpp-b12061") == 60.0
    assert run.startup_s("lmstudio") == 60.0
    assert run.startup_s("splash-1.5.0") == max(1.5 * 100.7, 120.0)


def test_fast_cells_still_budget_gate_post_finalize_and_shutdown():
    # cells that finish in ~8 s: the timeout still covers the gate, post, finalize and shutdown
    rows = [row("omlx-0.7.3", "B1.fast", est=0.27)]
    b = run.job_budget(rows)
    floor = run.GATE_PRE_MIN + run.POST_MIN + run.FINALIZE_MIN + run.SHUTDOWN_MIN
    assert b["timeout"] >= 1.3 * floor - 0.05
    assert b["lane"] == ("short" if b["timeout"] <= 10 else "normal")
    assert b["stall"] >= 5.0 and b["stall"] >= 2 * run.GATE_PRE_MIN


def test_dry_run_prints_itemized_budget_timeout_stall_and_lane(capsys):
    rows = [row("yunshu", "B1.p", est=4.0)]
    plan = run.plan_batch(rows, "B1", PIN)
    run.print_plan(plan)
    out = capsys.readouterr().out
    for key in (
        "startup",
        "gate_pre",
        "cells",
        "post",
        "finalize",
        "shutdown",
        "total",
        "timeout_min",
        "stall_min",
        "lane:",
    ):
        assert key in out, key


# --- one gpuq lane per batch: gpuq keeps submission order only inside one lane ----------------


def test_a_batch_with_a_short_job_and_a_normal_job_runs_in_one_lane():
    # a tiny cell (short budget) next to a 27B cell: the batch must not mix lanes
    rows = [row("yunshu", "B1.a-tiny", est=0.05), row("yunshu", "B1.b-big", est=30.0)]
    plan = run.plan_batch(rows, "B1", PIN)
    assert {j.lane for j in plan.jobs} == {"normal"}
    assert all(j.timeout_min > run.SHORT_LANE_MIN for j in plan.jobs)


def _gpuq_module():
    spec = importlib.util.spec_from_file_location(
        "gpuq_under_test", ROOT / "scripts" / "dev" / "gpuq.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_gpuq_jobs(plan):
    """gpuq job dicts for the plan's jobs in submission order (same owner, priority, device)."""
    return [
        {
            "id": f"j{i:03d}",
            "label": j.label,
            "state": "pending",
            "submitted": float(i),
            "priority": 1,
            "device": "m5",
            "timeout_s": j.timeout_min * 60,
            "short": False,
            "cwd": "/w",
            "env": {"GPUQ_OWNER": "fullcmp"},
        }
        for i, j in enumerate(plan.jobs)
    ]


def test_gpuq_picker_keeps_the_planned_order_when_one_lane(monkeypatch):
    gq = _gpuq_module()
    rows = [row("yunshu", "B1.a-tiny", est=0.05), row("yunshu", "B1.b-big", est=30.0)]
    plan = run.plan_batch(rows, "B1", PIN)  # one lane after the fix
    jobs = _fake_gpuq_jobs(plan)
    order = []
    while True:
        pick = gq._pick(jobs)
        if pick is None:
            break
        pick["state"] = "running"
        pick["started"] = float(len(order))
        order.append(pick["label"])
    assert order == [j.label for j in plan.jobs]


def test_gpuq_picker_reorders_a_mixed_lane_batch_so_the_guard_must_fail(monkeypatch):
    # the cause of wire-1106: a short-named job (submitted later) is picked before a normal job
    gq = _gpuq_module()
    jobs = [
        {
            "id": "a",
            "label": "fullcmp-B1-x-r0",
            "state": "pending",
            "submitted": 0.0,
            "priority": 1,
            "device": "m5",
            "timeout_s": 20 * 60,
            "short": False,
            "cwd": "/w",
            "env": {"GPUQ_OWNER": "fullcmp"},
        },
        {
            "id": "b",
            "label": "fullcmp-tiny-y-r0",
            "state": "pending",
            "submitted": 1.0,
            "priority": 1,
            "device": "m5",
            "timeout_s": 20 * 60,
            "short": False,
            "cwd": "/w",
            "env": {"GPUQ_OWNER": "fullcmp"},
        },
    ]
    assert gq._pick(jobs)["id"] == "b"  # the short-lane job jumps the queue


# --- mixed cells: their own work model (long prefill, short prefills, concurrent decode) -------


def _mixed(engine, ctx_tokens, conc, out):
    r = row(
        engine,
        f"B2.mix.{engine}.{ctx_tokens}",
        ctx_tokens=ctx_tokens,
        conc=conc,
        out=out,
        turns="mixed",
        batch="B2",
    )
    r["ctx"] = f"{ctx_tokens // 1024}K"
    return r


def test_mixed_128k_cell_term_exceeds_the_cold_prefill_and_decode():
    for engine in ("yunshu", "tf-1.2.1"):
        r = _mixed(engine, 131072, 8, 512)
        cold_prefill_min = 131072 / run.fm.prefill_tps(engine, 131072) / 60.0
        decode_min = (
            512
            * run.fm.slowdown(engine, 131072, 8)
            / run.fm.decode_tps(engine, 131072)
            / 60.0
        )
        assert run.cell_budget(r)["cells"] > cold_prefill_min + decode_min, engine
        # the 128K cell's trials: the long prefill is in every trial
        assert run.cell_budget(r)["cells"] >= run.CONC_TRIALS * cold_prefill_min


def test_yunshu_8k_plus_short_mixed_term_is_within_30_percent_of_the_measured_phase():
    # wire-1106c yunshu SMOKE mixed 8K c4 o64: measured mixed phase 22.9 s (fullcmp_cell_timing.json)
    measured_min = (
        run.fm.TIMING["job_budget"]["mixed_check_yunshu_8k_c4"]["value"] / 60.0
    )
    r = _mixed("yunshu", 8192, 4, 64)
    term = run.cell_budget(r)["cells"]
    assert abs(term - measured_min) / measured_min <= 0.30, (term, measured_min)


def test_fast_engine_mixed_term_is_its_model_without_the_probe():
    r = _mixed("yunshu", 131072, 8, 512)
    assert abs(run.cell_budget(r)["cells"] - run.mixed_work_min(r)) < 1e-9
    slow = _mixed("mlx-vlm-server", 131072, 8, 512)
    assert (
        abs(
            run.cell_budget(slow)["cells"]
            - run.mixed_work_min(slow)
            - run.MIXED_PROBE_MIN
        )
        < 1e-9
    )


def test_slow_mixed_jobs_get_a_ten_minute_stall_for_every_manifest_engine():
    # every manifest engine id: the slow-mixed set gets >= 10 min, the others keep the default rule
    for engine in run.fm.ENGINE_IDS:
        if engine in ("ollama",):
            continue
        r = _mixed(engine, 8192, 4, 64)
        b = run.job_budget([r])
        if engine in run.MIXED_SLOW_ENGINES:
            assert b["stall"] >= run.MIXED_SLOW_STALL_MIN, (engine, b["stall"])
        assert b["stall"] >= run.STALL_FLOOR_MIN, engine
    assert set(run.MIXED_SLOW_ENGINES) <= set(run.fm.ENGINE_IDS)


# --- --profile-engines: -prof jobs and the audit's profile requirement for listed engines only ----


def test_profile_engines_limits_the_prof_jobs_to_the_listed_engine():
    rows = [row("yunshu", "B1.a", est=2.0), row("omlx-0.7.3", "B1.b", est=2.0)]
    plan = run.plan_batch(rows, "B1", PIN, profile_rep=True, profile_engines=["yunshu"])
    prof = [j for j in plan.jobs if "-prof" in j.label]
    assert len(prof) == 1
    assert prof[0].engine == "yunshu" and prof[0].label.startswith("fullcmp-B1-yunshu")
    assert not any(j.engine == "omlx-0.7.3" and "-prof" in j.label for j in plan.jobs)


def test_profile_default_is_every_engine():
    rows = [row("yunshu", "B1.a", est=2.0), row("omlx-0.7.3", "B1.b", est=2.0)]
    plan = run.plan_batch(rows, "B1", PIN, profile_rep=True)
    assert {j.engine for j in plan.jobs if "-prof" in j.label} == {
        "yunshu",
        "omlx-0.7.3",
    }


def test_audit_manifest_names_the_profile_required_engines():
    rows = [row("yunshu", "B1.a"), row("omlx-0.7.3", "B1.b")]
    m = run.audit_manifest(rows, ["B1"], PIN, ["yunshu", "omlx-0.7.3"], ["yunshu"])
    assert m["profile_required_engines"] == ["yunshu"]
    assert (
        run.audit_manifest(rows, ["B1"], PIN, ["yunshu", "omlx-0.7.3"])[
            "profile_required_engines"
        ]
        is None
    )


def test_audit_requires_profile_rows_only_for_the_listed_engines():
    import fullcmp_audit as fa

    run_ = fa.Run({"groups": {}}, [], [], [], [])
    run_.profile_required = {"yunshu"}
    assert fa.profile_required(run_, "yunshu") is True
    assert fa.profile_required(run_, "omlx-0.7.3") is False
    run_.profile_required = None  # no list in the manifest: every engine, as before
    assert fa.profile_required(run_, "omlx-0.7.3") is True


# --- quiet-gate refusals (rc 3) are retried in the same slot ----------------------------------


@pytest.fixture
def retry_world(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WRAP_ROOT", tmp_path / "wrap")
    monkeypatch.setattr(run, "QUIET_DIR", tmp_path / "quiet")
    monkeypatch.setattr(run, "RAW_ROOT", tmp_path / "raw")
    return tmp_path


def _two_slots():
    rows = [
        row("yunshu", "B1.a", est=2.0),
        row("yunshu", "B1.b", est=2.0, ctx=8192, ctx_tokens=8192),
    ]
    return run.plan_batch(rows, "B1", PIN).jobs


def _scripted(outcomes):
    """run_one stand-in: outcomes[label] is the list of (rc, state) answers, one per attempt."""
    calls, left = [], {k: list(v) for k, v in outcomes.items()}

    def run_one(job):
        calls.append(job.label)
        base = job.label.split("-retry")[0]
        rc, state = (
            left.get(base, [(0, "done")]).pop(0) if left.get(base) else (0, "done")
        )
        return f"jid-{len(calls)}", rc, state

    return run_one, calls


def test_quiet_refusals_are_retried_in_the_same_slot_then_the_slot_passes(retry_world):
    jobs = _two_slots()
    first = jobs[0].label
    run_one, calls = _scripted({first: [(3, "failed"), (3, "failed"), (0, "done")]})
    sleeps = []
    ids, failed = run.submit_sequential(
        jobs,
        PIN,
        run_one=run_one,
        sleep=sleeps.append,
        retry_log=retry_world / "retries.jsonl",
    )
    assert failed == 0
    assert calls == [
        first,
        f"{first}-retry1",
        f"{first}-retry2",
        jobs[1].label,
    ]  # same slot, in order
    assert sleeps == [5 * 60, 10 * 60]  # the back-off table
    log = [
        json.loads(x) for x in (retry_world / "retries.jsonl").read_text().splitlines()
    ]
    assert [r["attempt"] for r in log] == [0, 1] and all(
        r["slot"] == first for r in log
    )


def test_exhausted_quiet_refusals_fail_the_slot_and_the_batch(retry_world):
    jobs = _two_slots()
    first = jobs[0].label
    run_one, calls = _scripted({first: [(3, "failed")] * 6})
    ids, failed = run.submit_sequential(
        jobs, PIN, run_one=run_one, sleep=lambda s: None
    )
    assert failed == 1
    assert (
        calls.count(first) + sum(1 for c in calls if c.startswith(f"{first}-retry"))
        == run.MAX_TRIES
    )


def test_other_failures_are_not_retried(retry_world):
    jobs = _two_slots()
    first = jobs[0].label
    run_one, calls = _scripted({first: [(2, "failed")]})
    ids, failed = run.submit_sequential(
        jobs, PIN, run_one=run_one, sleep=lambda s: None
    )
    assert failed == 1 and calls.count(first) == 1


def test_a_stalled_or_timed_out_job_stays_a_failure(retry_world):
    jobs = _two_slots()
    first = jobs[0].label
    run_one, calls = _scripted({first: [(-2, "stalled")]})
    ids, failed = run.submit_sequential(
        jobs, PIN, run_one=run_one, sleep=lambda s: None
    )
    assert failed == 1 and calls.count(first) == 1


def test_retry_log_names_the_foreign_holders(retry_world):
    jobs = _two_slots()
    first = jobs[0].label
    (retry_world / "quiet").mkdir()
    (retry_world / "quiet" / f"{first}.jsonl").write_text(
        json.dumps(
            {
                "ok": False,
                "holders": [{"name": "XProtectRemediatorPirrit", "pct": 95.0}],
            }
        )
        + "\n"
    )
    run_one, _calls = _scripted({first: [(3, "failed"), (0, "done")]})
    run.submit_sequential(
        jobs,
        PIN,
        run_one=run_one,
        sleep=lambda s: None,
        retry_log=retry_world / "retries.jsonl",
    )
    rec = json.loads((retry_world / "retries.jsonl").read_text().splitlines()[0])
    assert rec["holders"][0]["name"] == "XProtectRemediatorPirrit"


def test_retry_job_is_a_new_wrapper_with_its_own_gate_log(retry_world):
    job = _two_slots()[0]
    again = run.retry_job(job, 1, PIN)
    assert again.label == f"{job.label}-retry1"
    assert f"{job.label}-retry1" in again.script and "retry" in again.flags
    # the gate runs before any cell in the retry too: a refused attempt writes no raw
    assert again.script.index("fullcmp_quiet") < again.script.index("mkdir")


# --- per-cell quiet refusal, rerun of the unfinished cells, quiet-flagged requeue -------------


def test_a_failed_per_cell_quiet_snapshot_refuses_the_cell():
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "tfbench_q", ROOT / "scripts" / "research" / "tfbench.py"
    )
    tb = ilu.module_from_spec(spec)
    sys.modules["tfbench_q"] = tb
    spec.loader.exec_module(tb)
    bad = {
        "ok": False,
        "total_foreign_pct": 112.8,
        "holders": [{"name": "mds", "pct": 48.7}],
    }
    assert tb.quiet_refused(bad) is True
    assert tb.quiet_refused({"ok": True}) is False
    assert (
        tb.quiet_refused(None) is False
    )  # not a fullcmp cell: no snapshot, no refusal
    assert tb.QUIET_REFUSED_EXIT == 3


def _part_done(path, complete=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"part": "decode"}\n'
        + ('{"part": "part_done", "complete": true}\n' if complete else "")
    )


def test_a_quiet_retry_runs_only_the_unfinished_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "RAW_ROOT", tmp_path / "raw")
    rows = [
        row("yunshu", "B1.first", est=2.0),
        row("yunshu", "B1.second", est=2.0, ctx=8192, ctx_tokens=8192),
    ]
    job = run.plan_batch(rows, "B1", PIN).jobs[0]
    job = run.dataclasses.replace(job, cells=[dict(r) for r in rows], rep=0)
    _part_done(
        run.raw_path("yunshu", "B1.first", 0)
    )  # the first cell finished before the refusal
    assert [c["id"] for c in run.remaining_cells(job)] == ["B1.second"]
    retry = run.retry_job(job, 1, PIN, cells=run.remaining_cells(job))
    assert [c["id"] for c in retry.cells] == ["B1.second"]
    assert retry.label.endswith("-retry1")


def test_a_quiet_flagged_cell_is_superseded_and_rerun_in_its_slot(
    monkeypatch, tmp_path
):
    import fullcmp_audit as fa

    raw = tmp_path / "raw"
    monkeypatch.setattr(run, "RAW_ROOT", raw)
    rows = [row("yunshu", "B1.a", est=2.0)]
    plan = run.plan_batch(rows, "B1", PIN)
    rec = {
        "ok": False,
        "total_foreign_pct": 60.0,
        "max_single_pct": 20.0,
        "max_total_pct": 150.0,
        "holders": [{"pid": 9, "name": "stray", "pct": 43.0}],
    }
    head = {
        "part": "session",
        "quiet_pre": {**rec, "ok": True, "total_foreign_pct": 1.0, "holders": []},
        "quiet_post": rec,
    }
    cell_file = run.raw_path("yunshu", "B1.a", 0)
    cell_file.parent.mkdir(parents=True, exist_ok=True)
    cell_file.write_text(
        json.dumps(head) + "\n" + '{"part": "part_done", "complete": true}\n'
    )
    manifest = run.audit_manifest(rows, ["B1"], PIN, ["yunshu"])
    submitted = []
    monkeypatch.setattr(
        run,
        "_submit_and_wait",
        lambda job: submitted.append(job.label) or ("id", 0, "done"),
    )
    failures = run.requeue_quiet_flagged(plan, PIN, raw, manifest)
    assert failures == 0
    assert not cell_file.exists()  # the flagged file moved aside, as evidence
    assert cell_file.with_name(
        cell_file.name.replace(".jsonl", ".superseded-1.jsonl")
    ).exists()
    assert submitted and submitted[0].endswith("-qretry1")
    # the audit does not count the superseded file as an unplanned file
    assert (
        fa.SUPERSEDED_TAG
        in cell_file.with_name(
            cell_file.name.replace(".jsonl", ".superseded-1.jsonl")
        ).name
    )


def test_a_cell_flagged_twice_fails_the_batch(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    monkeypatch.setattr(run, "RAW_ROOT", raw)
    rows = [row("yunshu", "B1.a", est=2.0)]
    plan = run.plan_batch(rows, "B1", PIN)
    rec = {
        "ok": False,
        "total_foreign_pct": 60.0,
        "max_single_pct": 20.0,
        "max_total_pct": 150.0,
        "holders": [{"pid": 9, "name": "stray", "pct": 43.0}],
    }
    cell_file = run.raw_path("yunshu", "B1.a", 0)
    cell_file.parent.mkdir(parents=True, exist_ok=True)
    cell_file.write_text(json.dumps({"part": "session", "quiet_post": rec}) + "\n")
    cell_file.with_name(
        cell_file.name.replace(".jsonl", ".superseded-1.jsonl")
    ).write_text("x\n")
    submitted = []
    monkeypatch.setattr(
        run,
        "_submit_and_wait",
        lambda job: submitted.append(job.label) or ("id", 0, "done"),
    )
    assert (
        run.requeue_quiet_flagged(
            plan, PIN, raw, run.audit_manifest(rows, ["B1"], PIN, ["yunshu"])
        )
        == 1
    )
    assert submitted == []


def test_a_real_batch_manifest_carries_the_ctl_1103_controls(tmp_path, monkeypatch):
    """Failing-before (wire-1108 audit): the driver's audit manifest had controls {} so fairness_controls
    reported 'A/A delta None' and 'injected delay not detected'. A real batch now reads the ctl-1103
    verdict files; a smoke batch carries none; a failing control is reported as an error, never passed."""
    import fullcmp_audit_manifest as fam

    assert run.batch_controls(True) == {}
    monkeypatch.setattr(
        fam,
        "load_controls",
        lambda: (
            {"aa_delta_pct": 0.4, "aa_noise_pct": 2.0, "injected_delay_detected": True},
            {},
        ),
    )
    c = run.batch_controls(False)
    assert c["aa_delta_pct"] == 0.4 and c["injected_delay_detected"] is True

    def boom():
        raise fam.ControlError("injected-delay control not detected")

    monkeypatch.setattr(fam, "load_controls", boom)
    assert "not detected" in run.batch_controls(False)["error"]


def test_the_real_ctl_1103_controls_pass_the_fairness_guard():
    """The existing ctl-1103 runs (A/A inside noise, slow2 detected, flip3 detected) satisfy the audit's
    fairness_controls guard through the driver's own plumbing. Skipped where the verify runs are absent."""
    import fullcmp_audit as fa
    import fullcmp_audit_manifest as fam

    if not (fam.VERIFY_RUNS / fam.CONTROL_AA / "verdict.json").is_file():
        pytest.skip("ctl-1103 verify runs not on this machine")
    c = run.batch_controls(False)
    assert "error" not in c, c
    assert (
        c["injected_delay_detected"] is True
        and c["evidence"]["slow"] == fam.CONTROL_SLOW
    )
    res = fa.check_fairness_controls(
        fa.Run(
            manifest={"controls": c}, files=[], missing=[], extra=[], parse_errors=[]
        )
    )
    assert [v for v in res.violations if ": controls:" in v] == [], res.violations


# --- a failed (non-quiet) cell is rerun once at the end of the batch, from the same snapshot ---------------------------


def _complete(engine, cell, rep=0):
    p = run.raw_path(engine, cell, rep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"part": "session"})
        + "\n"
        + json.dumps({"part": "part_done", "complete": True})
        + "\n"
    )


def _failed(engine, cell, rep=0):
    p = run.raw_path(engine, cell, rep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"part": "session"})
        + "\n"
        + json.dumps({"part": "error", "error": "boom"})
        + "\n"
    )
    p.with_name(p.name.replace(".jsonl", ".server.log")).write_text(
        "log of the failed attempt"
    )


def test_rerun_failed_cells_appends_one_rerun_job_per_slot_and_keeps_the_failed_record(
    pin_file, tmp_path, monkeypatch
):
    monkeypatch.undo()  # the autouse stubs above are not wanted here
    monkeypatch.setattr(run, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(run, "RETRY_LOG", tmp_path / "retry.log")
    rows = [
        {**row(e, c), "reps": 1}
        for e, c in (("yunshu", "B1.a"), ("yunshu", "B1.b"), ("omlx-0.7.3", "B1.c"))
    ]
    plan = run.plan_batch(rows, "B1", json.loads(pin_file.read_text()))
    _complete("yunshu", "B1.a")
    _failed("yunshu", "B1.b")  # failed: an error record and no part_done
    # omlx B1.c never ran (no file)
    submitted = []

    def fake_submit(jobs, pin, retry_log=None):
        submitted.extend(jobs)
        for j in jobs:
            for c in j.cells:
                _complete(c["engine"], c["id"], j.rep)  # the rerun succeeds
        return [f"id{i}" for i in range(len(jobs))], 0

    ids, still = run.rerun_failed_cells(
        plan, json.loads(pin_file.read_text()), submit=fake_submit
    )
    assert still == 0 and len(ids) == 2
    assert len(submitted) == 2 and all(
        j.label.endswith("-rerun1") for j in submitted
    )  # one per slot, after the others
    assert {j.label.removesuffix("-rerun1") for j in submitted} <= {
        j.label for j in plan.jobs
    }
    done_cells = {c["id"] for j in submitted for c in j.cells}
    assert done_cells == {"B1.b", "B1.c"}  # B1.a was complete: not rerun
    kept = run.raw_path("yunshu", "B1.b", 0).with_name(
        "B1.b-r0.superseded-fail1.server.log"
    )
    assert (
        kept.read_text() == "log of the failed attempt"
    )  # the failed first record is evidence, not deleted
    assert (
        run.raw_path("yunshu", "B1.b", 0)
        .with_name("B1.b-r0.superseded-fail1.jsonl")
        .exists()
    )


def test_a_cell_still_incomplete_after_the_rerun_is_counted_and_there_is_no_third_attempt(
    pin_file, tmp_path, monkeypatch
):
    monkeypatch.undo()
    monkeypatch.setattr(run, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(run, "RETRY_LOG", tmp_path / "retry.log")
    plan = run.plan_batch(
        [{**row("yunshu", "B1.a"), "reps": 1}], "B1", json.loads(pin_file.read_text())
    )
    calls = []

    def fail_again(jobs, pin, retry_log=None):
        calls.append(len(jobs))
        return ["id0"], 1  # nothing written: the rerun failed too

    ids, still = run.rerun_failed_cells(
        plan, json.loads(pin_file.read_text()), submit=fail_again
    )
    assert still == 1 and calls == [1]


def test_nothing_is_rerun_when_every_cell_is_complete(pin_file, tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(run, "RAW_ROOT", tmp_path / "raw")
    plan = run.plan_batch(
        [{**row("yunshu", "B1.a"), "reps": 1}], "B1", json.loads(pin_file.read_text())
    )
    _complete("yunshu", "B1.a")

    def never(jobs, pin, retry_log=None):
        raise AssertionError("no rerun expected")

    assert run.rerun_failed_cells(
        plan, json.loads(pin_file.read_text()), submit=never
    ) == ([], 0)


# --- profile-only batch: only the -prof jobs, from an existing snapshot, beside a finished timed batch -----------------


def _profile_rows():
    return [
        row("yunshu", "B1.a", reps=1),
        row("yunshu-pb2", "B1.b", reps=1),
        row("omlx-0.7.3", "B1.c", reps=1),
    ]


def test_plan_batch_profile_only_has_only_profile_jobs(pin_file):
    pin = json.loads(pin_file.read_text())
    both = run.plan_batch(
        _profile_rows(),
        "B1",
        pin,
        profile_rep=True,
        profile_engines=["yunshu", "yunshu-pb2"],
    )
    only = run.plan_batch(
        _profile_rows(),
        "B1",
        pin,
        profile_rep=True,
        profile_engines=["yunshu", "yunshu-pb2"],
        profile_only=True,
    )
    assert any(
        "profile" not in j.flags for j in both.jobs
    )  # the normal plan keeps the timed jobs
    assert only.jobs and all("profile" in j.flags for j in only.jobs)
    assert {j.engine for j in only.jobs} == {
        "yunshu",
        "yunshu-pb2",
    }  # omlx has no profile job here
    assert all(j.label.endswith("-prof") or "-prof" in j.label for j in only.jobs)


def test_profile_only_needs_profile_rep_and_engines(
    pin_file, tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in _profile_rows()) + "\n")
    assert (
        run.main(
            [
                "--batch",
                "B1",
                "--manifest",
                str(manifest),
                "--profile-only",
                "--dry-run",
            ]
        )
        == 2
    )
    assert (
        run.main(
            [
                "--batch",
                "B1",
                "--manifest",
                str(manifest),
                "--profile-only",
                "--profile-rep",
                "--dry-run",
            ]
        )
        == 2
    )
    assert (
        "--profile-only needs --profile-rep and --profile-engines"
        in capsys.readouterr().err
    )


def test_profile_only_dry_run_prints_only_profile_jobs(
    pin_file, tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in _profile_rows()) + "\n")
    args = [
        "--batch",
        "B1",
        "--manifest",
        str(manifest),
        "--profile-rep",
        "--profile-engines",
        "yunshu,yunshu-pb2",
        "--dry-run",
    ]
    assert run.main([*args, "--profile-only"]) == 0
    out = capsys.readouterr().out
    jobs = [ln for ln in out.splitlines() if ln.startswith("JOB ")]
    assert jobs and all(ln.rstrip().endswith("-prof") or "-prof" in ln for ln in jobs)
    assert run.main(args) == 0
    full = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("JOB ")]
    assert len(full) > len(jobs)  # the normal plan also has the timed jobs


def test_profile_only_run_skips_the_audit_reps_and_requeues(
    pin_file, tmp_path, monkeypatch
):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in _profile_rows()) + "\n")
    monkeypatch.setattr(run, "gpuq_status_text", lambda: "")
    monkeypatch.setattr(run, "swap_gate", lambda b: 0)
    submitted = []

    def fake_one(job):
        submitted.append(job.label)
        return f"id{len(submitted)}", 0, "done"

    def forbidden(*a, **k):
        raise AssertionError(
            "a profile-only batch must not audit, schedule reps or requeue"
        )

    monkeypatch.setattr(run, "_submit_and_wait", fake_one)
    monkeypatch.setattr(run, "run_audit", forbidden)
    monkeypatch.setattr(run, "run_reps", forbidden)
    monkeypatch.setattr(run, "requeue_quiet_flagged", forbidden)
    rc = run.main(
        [
            "--batch",
            "B1",
            "--manifest",
            str(manifest),
            "--profile-rep",
            "--profile-engines",
            "yunshu,yunshu-pb2",
            "--profile-only",
        ]
    )
    assert rc == 0 and submitted and all("-prof" in lbl for lbl in submitted)


def test_cells_filter_selects_only_matching_cells_and_refuses_an_empty_match(
    tmp_path, capsys
):
    """real-1104 B2: restart cells (SIGKILL before the disk write) and oMLX c4/c8 cells (batch-parked parse)
    need a targeted re-run with the fixed harness; --cells selects them by id."""
    rows = [
        {"batch": "B2", "id": "B2.restart.yunshu.32K", "engine": "yunshu"},
        {"batch": "B2", "id": "B2.conc.omlx-0.7.3.8K.c8", "engine": "omlx-0.7.3"},
        {"batch": "B2", "id": "B2.ctx.yunshu.4K", "engine": "yunshu"},
    ]
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    rc = run.main(
        ["--batch", "B2", "--manifest", str(m), "--cells", r"^B2\.nomatch", "--dry-run"]
    )
    assert rc == 2 and "matches no B2 cell" in capsys.readouterr().err


def test_resume_reps_skips_cells_whose_rep3_is_on_disk(tmp_path):
    """real-1104 B2 (2026-11-07): the driver died with its parent during rep 3; --resume-reps runs only the
    rep-3 cells still missing (a non-empty <raw>/<engine>/<cell>-r2.jsonl counts as present)."""
    row = {"engine": "kapitsa-0.2.26", "id": "B2.conc.kapitsa-0.2.26.8K.c2"}
    d = tmp_path / row["engine"]
    d.mkdir()
    assert not run.rep3_present(tmp_path, row)
    (d / f"{row['id']}-r{run.REP3_INDEX}.jsonl").write_text("")
    assert not run.rep3_present(
        tmp_path, row
    )  # an empty file is a dead job, not a result
    (d / f"{row['id']}-r{run.REP3_INDEX}.jsonl").write_text('{"part": "conc"}\n')
    assert run.rep3_present(tmp_path, row)
    assert "--resume-reps" in Path(run.__file__).read_text(encoding="utf-8")


def test_supersede_never_overwrites_and_cells_rerun_moves_existing_records(
    tmp_path, monkeypatch
):
    """real-1104 B2 fix (2026-11-07): the --cells re-run hit 'exists; a fullcmp cell writes a fresh file' on every
    job, and rerun_failed_cells' rename could overwrite an earlier superseded-fail1 copy. Existing records are now
    superseded with the first free number before the re-run."""
    d = tmp_path / "yunshu"
    d.mkdir()
    cell = d / "B2.restart.yunshu.32K-r0.jsonl"
    for n in (1,):
        (d / f"B2.restart.yunshu.32K-r0.superseded-fail{n}.jsonl").write_text(
            "old fail1\n"
        )
    for sfx in (".jsonl", ".telemetry.jsonl", ".server.log"):
        (d / f"B2.restart.yunshu.32K-r0{sfx}").write_text("current\n")
    moved = run.supersede_cell_files(cell, "fail")
    assert (
        d / "B2.restart.yunshu.32K-r0.superseded-fail1.jsonl"
    ).read_text() == "old fail1\n"
    assert {p.name for p in moved} == {
        "B2.restart.yunshu.32K-r0.superseded-fail2.jsonl",
        "B2.restart.yunshu.32K-r0.superseded-fail2.telemetry.jsonl",
        "B2.restart.yunshu.32K-r0.superseded-fail2.server.log",
    }
    assert not cell.exists()
    monkeypatch.setattr(run, "RAW_ROOT", tmp_path)
    cell.write_text("again\n")

    class J:
        flags, rep = [], 0
        cells = [{"engine": "yunshu", "id": "B2.restart.yunshu.32K"}]

    class P:
        jobs = [J()]

    assert run.supersede_selected(P()) == 1
    assert (
        d / "B2.restart.yunshu.32K-r0.superseded-fix1.jsonl"
    ).read_text() == "again\n"
    assert run.supersede_selected(P()) == 0
