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
        if r["batch"] == "B1" and r["ctx"] in fm_.CORE_CTX and r["status"] == "unverified"
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
        min(20.0, __import__("math").ceil(round(1.3 * b["total"] * 10, 6)) / 10)
    )
    # floored by the longest same-kind run (the killed 1103-220358 job ran 20 min)
    assert run.job_budget(rows, prior_min=20.0)["timeout"] == 20.0
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
    rows = [row("yunshu", "B1.big", est=60.0)]  # 30 min per rep
    plan = run.plan_batch(rows, "B1", PIN)
    assert plan.jobs and all(j.over_cap for j in plan.jobs)


def test_single_cell_above_the_margin_runs_alone_and_is_flagged():
    rows = [row("yunshu", "B1.long", est=30.0), row("yunshu", "B1.short", est=2.0)]
    plan = run.plan_batch(rows, "B1", PIN)
    rep0 = [j for j in plan.jobs if j.rep == 0]
    long_job = next(j for j in rep0 if "B1.long" in j.cell_ids)
    assert long_job.cell_ids == ["B1.long"]  # alone: the next cell does not fit with it
    assert long_job.over_margin and not long_job.over_cap
    assert long_job.timeout_min == run.JOB_CAP_MIN  # 1.3 x 16 min, capped at 20


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
    assert "TFB_TELEMETRY_DIR=/Volumes/Micron/yunshu-fullcmp/raw" in text
    assert "HOME=/Volumes/Micron/yunshu-fullcmp/home/yunshu" in text
    assert "--engine yunshu-main" in text and "--part decode" in text
    assert "GPUQ_JOB_ID" not in text  # gpuq exports it to the job env
    assert "/Volumes/Micron/yunshu-fullcmp/raw/yunshu/B1.ok-r0.jsonl" in text


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
    rc = run.main(["--batch", "B1", "--manifest", str(manifest), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "JOB fullcmp-B1-yunshu-q27-4bit-th-off-r0" in out
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
    monkeypatch.setattr(run, "submit_all", lambda jobs: (["id-1"], 0))
    monkeypatch.setattr(run, "wait_all", lambda ids: 0)
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

    def fake_submit(jobs):
        calls["submit"].append([j.label for j in jobs])
        return [f"id{len(calls['submit'])}-{i}" for i in range(len(jobs))], 0

    monkeypatch.setattr(run, "submit_all", fake_submit)
    monkeypatch.setattr(run, "wait_all", lambda ids: calls["wait"].append(ids) or 0)
    monkeypatch.setattr(run, "run_audit", lambda root, mpath: 0)

    def fake_reps(root, mpath):
        calls["reps"] += 1
        return [("yunshu", "B1.a")] if calls["reps"] == 1 else []

    monkeypatch.setattr(run, "run_reps", fake_reps)
    rc = run.main(["--batch", "B1", "--manifest", str(manifest)])
    assert rc == 0
    assert len(calls["submit"]) == 2
    assert calls["submit"][1] == ["fullcmp-B1-yunshu-q27-4bit-th-off-r2-rep3"]
    manifest_out = json.loads((run.AUDIT_DIR / "B1_audit_manifest.json").read_text())
    assert {c["cell"] for c in manifest_out["cells"]} == {"B1.a", "B1.b"}

    # a failed wait (nonzero rc) makes the whole batch exit nonzero
    calls["reps"] = 0
    monkeypatch.setattr(run, "wait_all", lambda ids: 1)
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
    b1 = next(r for r in rows if r["batch"] == "B1" and r["ctx"] == "1K" and r["prompt_kind"] == "code")
    assert run.expected_records(b1) == 3  # per rep file
    conc = next(r for r in rows if r["batch"] == "B2" and r["sweep"] == "conc" and r["conc"] == 4)
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
    for key in ("startup", "gate_pre", "cells", "post", "finalize", "shutdown", "total",
                "timeout_min", "stall_min", "lane:"):
        assert key in out, key
