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


def test_group_splits_at_20_min_per_rep():
    # reps=2, est 10 per row -> 5 min per rep per cell; 4 cells fill 20 min exactly.
    rows = [row("yunshu", f"B1.c{i}", est=10.0) for i in range(6)]
    chunks = run.pack_chunks(rows)
    assert [len(c) for c in chunks] == [4, 2]
    assert all(sum(run.cell_est(r) for r in c) <= 20.0 for c in chunks)


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
    assert argv[4:8] == ["--priority", "1", "--quiet", "--"]
    assert argv[8:] == job.cmd


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
    assert all(c["expected_records"] is None for c in m["cells"])
