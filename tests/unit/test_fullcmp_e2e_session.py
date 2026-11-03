"""E2E cell types through the REAL tfbench code path against the session stub (no GPU, no model).

Chain per test (the same one test_fullcmp_e2e.py runs): fullcmp_manifest rows -> fullcmp_run plans the
job and writes the wrapper -> the wrapper runs real tfbench (real Srv launch path, TFB_STUB_SERVER
replaces only the engine process) -> real telemetry sampler -> real fullcmp_audit: zero violations.
Each test then reads the raw records and checks the cell's own fields.

Cell types covered (COVERED_CELL_TYPES below): conc (2/4/8 concurrent streams), follow-up (turn2
immediately after cold), idle follow-up (the turn2 after an idle gap; the gap is patched to 0.05 s),
restart-with-disk-cache (a second stub process on the same cache dir). Engines: yunshu and tf-1.2.1.
The audit needs >= 2 prompts per (model, ctx, kind), so every row runs prompt variants 0 and 1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import fullcmp_stub_session as sess  # noqa: E402

from .test_fullcmp_e2e import FAKE_JOB_ID, _run_audit  # noqa: E402
from .test_fullcmp_e2e_coverage import cell_type_key  # noqa: E402

ENGINES_UNDER_TEST = ("yunshu", "tf-1.2.1")
VARIANTS = (0, 1)

COVERED_CELL_TYPES = {
    "conc": "test_conc_cells_run_through_tfbench_and_audit_is_clean",
    "follow-up": "test_follow_up_turn2_cell_hits_the_prefix_cache",
    "idle-follow-up": "test_idle_follow_up_cell_runs_turn2_after_the_idle_gap",
    # the manifest's restart row shape (B2 restart at 32K): key = cell_type_key of that row
    "restart-disk-cache": "test_restart_with_disk_cache_serves_prefix_hits",
}
# keys are cell_type_key()s (test_fullcmp_e2e_coverage); the restart row is the manifest's own
COVERED_CELL_TYPES = {
    **{k: v for k, v in COVERED_CELL_TYPES.items() if k != "restart-disk-cache"},
    cell_type_key(
        fm._cell_row(
            "yunshu",
            fm.SMOKE_BATCH,
            fm.spec(
                sweep="restart", ctx="32K", turns="restart", phases=("cold", "restart")
            ),
        )
    ): "test_restart_with_disk_cache_serves_prefix_hits",
}


@pytest.fixture(params=ENGINES_UNDER_TEST)
def engine(request):
    return request.param


def _rows(engine, specs):
    """Runnable manifest rows: each spec at prompt variants 0 and 1, one rep, the expected count
    carried on the row (smoke-style)."""
    rows = []
    for variant in VARIANTS:
        for sp in specs:
            row = fm._cell_row(engine, fm.SMOKE_BATCH, sp)
            row["id"] = f"{row['id']}.v{variant}"
            row["reps"] = 1
            row["prompt_variant"] = variant
            row["est_gpu_min"] = 0.0
            rows.append(row)
    return rows


def _run_cell_rows(tmp: Path, rows, *, stub_module: str = "fullcmp_stub_session"):
    """Plan the rows as one batch, run the wrapper against the session stub, return (raw, plan, pin)."""
    raw = tmp / "raw"
    fr.configure(raw, tmp / "work")
    pin = json.loads(fr.PIN_PATH.read_text())
    plan = fr.plan_batch(rows, fm.SMOKE_BATCH, pin)
    assert not plan.skipped, [(r["id"], why) for r, why in plan.skipped]
    assert len(plan.jobs) == 1, [j.label for j in plan.jobs]
    job = plan.jobs[0]
    wrapper = fr.write_wrapper(job)
    env = dict(os.environ)
    env["GPUQ_JOB_ID"] = FAKE_JOB_ID
    env["TFB_STUB_SERVER"] = str(sess.shim_dir(tmp / "shim", module=stub_module))
    proc = subprocess.run(
        ["/bin/sh", str(wrapper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return raw, plan, pin


def _audit_clean(raw: Path, rows, plan, pin, tmp: Path):
    manifest = fr.audit_manifest(rows, [fm.SMOKE_BATCH], pin, plan.engine_order)
    manifest["stub_engine"] = True  # test only: the engine process is the session stub
    res = _run_audit(raw, manifest, tmp)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert "0 violations" in res.stdout, res.stdout[-2000:]


def _records(raw: Path, engine: str, cell_id: str):
    path = raw / engine / f"{cell_id}-r0.jsonl"
    assert path.exists(), f"cell file missing: {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows[0]["part"] == "session"
    return rows[1:]


def _decode(raw, engine, row):
    return [r for r in _records(raw, engine, row["id"]) if r.get("part") == "decode"]


def test_conc_cells_run_through_tfbench_and_audit_is_clean(tmp_path, engine):
    specs = [
        fm.spec(
            sweep="smoke-conc", ctx="1K", kind="code", out=64, conc=n, phases=("conc",)
        )
        for n in (2, 4, 8)
    ] + [
        fm.spec(
            sweep="smoke-conc", ctx="1K", kind="prose", out=64, conc=2, phases=("conc",)
        )
    ]
    rows = _rows(engine, specs)
    raw, plan, pin = _run_cell_rows(tmp_path, rows)
    for row in rows:
        n = row["conc"]
        recs = [r for r in _records(raw, engine, row["id"]) if r.get("part") == "conc"]
        # record_count exact: one conc record per trial, each holding n requests
        assert len(recs) == fr.CONC_TRIALS, f"{row['id']}: conc records {len(recs)}"
        assert sum(r["n"] for r in recs) == n * fr.CONC_TRIALS
        for rec in recs:
            assert rec["n"] == n
            assert len(rec["per_req_dec"]) == len(rec["ttfts"]) == len(rec["cts"]) == n
            assert all(isinstance(t, (int, float)) and t > 0 for t in rec["ttfts"])
            assert all(
                isinstance(d, (int, float)) and d > 0 for d in rec["per_req_dec"]
            )
            assert all(c == row["out_len"] for c in rec["cts"]), rec["cts"]
            assert rec["total_tokens"] == sum(rec["cts"])
            assert rec["agg_tps"] > 0 and rec["wall_s"] > 0
            assert rec["dec_min"] == min(rec["per_req_dec"])
            assert rec["dec_max"] == max(rec["per_req_dec"])
            assert rec["dec_min"] <= rec["dec_median"] <= rec["dec_max"]
            assert rec["fair_max_min"] == round(rec["dec_max"] / rec["dec_min"], 3)
            assert len(rec["cached"]) == n and all(
                isinstance(c, int) for c in rec["cached"]
            )
    _audit_clean(raw, rows, plan, pin, tmp_path)


def test_follow_up_turn2_cell_hits_the_prefix_cache(tmp_path, engine):
    sp = fm.spec(
        sweep="smoke-core", ctx="1K", kind="code", out=64, phases=("cold", "turn2")
    )
    rows = _rows(engine, [sp])
    raw, plan, pin = _run_cell_rows(tmp_path, rows)
    for row in rows:
        cold, turn2 = _decode(raw, engine, row)
        assert (cold["phase"], turn2["phase"]) == ("cold", "turn2")
        assert cold["cached"] == 0, "a first sighting after an empty cache is a miss"
        assert turn2["cached"] > 0, (
            "the follow-up re-sends the cold prompt: a prefix hit"
        )
        assert turn2["ct"] == 64 and turn2["dec_tps"] > 0 and turn2["ttft_s"] > 0
    _audit_clean(raw, rows, plan, pin, tmp_path)


def test_idle_follow_up_cell_runs_turn2_after_the_idle_gap(
    tmp_path, engine, monkeypatch
):
    """The manifest's idle5 cell (cold, then the follow-up after 5 s idle). The gap is patched to
    0.05 s in the runner so the test stays fast; the job's argv carries --think-s 0.05."""
    monkeypatch.setattr(fr, "IDLE_THINK_S", 0.05)
    sp = fm.spec(
        sweep="smoke-idle",
        ctx="1K",
        kind="code",
        out=64,
        turns="idle5",
        phases=("cold", "fu_idle"),
    )
    rows = _rows(engine, [sp])
    raw, plan, pin = _run_cell_rows(tmp_path, rows)
    assert "--think-s 0.05" in plan.jobs[0].script
    for row in rows:
        cold, turn2 = _decode(raw, engine, row)
        assert (cold["phase"], turn2["phase"]) == ("cold", "turn2")
        assert turn2["cached"] > 0, "the idle follow-up still hits the cached prefix"
        assert turn2["ct"] == 64
    _audit_clean(raw, rows, plan, pin, tmp_path)


def test_restart_with_disk_cache_serves_prefix_hits(tmp_path, engine):
    """The restart cell in one run: cold prefill, then tfbench stops the server (keeping its HOME)
    and relaunches it on the same HOME (restart_server, a real Srv launch). The relaunched stub
    serves the cold prompt from the disk tier; the audit requires cached > 0 on the restart row."""
    sp = fm.spec(
        sweep="restart",
        ctx="1K",
        kind="code",
        out=64,
        turns="restart",
        phases=("cold", "restart"),
    )
    rows = _rows(engine, [sp])
    raw, plan, pin = _run_cell_rows(tmp_path, rows)
    for row in rows:
        cold, restart = _decode(raw, engine, row)
        assert (cold["phase"], restart["phase"]) == ("cold", "restart")
        assert cold["cached"] == 0, "first sighting after an empty HOME is a miss"
        assert restart["cached"] > 0, (
            "the relaunched server reads the same HOME's disk tier"
        )
        assert restart["ct"] == 64
    _audit_clean(raw, rows, plan, pin, tmp_path)


# ---------------------------------------------------------------- fast checks (no run)
def test_manifest_follow_up_phase_names_reach_the_audit_as_turn2():
    """Regression: the manifest's 'fu' / 'fu_idle' phases reached the audit unmapped, so every
    follow-up cell failed record_count ('turn2' recorded, 'fu' planned) and fu_idle crashed the
    audit manifest builder."""
    row = _rows("yunshu", [fm.spec(sweep="smoke-core", ctx="1K", kind="code", out=64)])[
        0
    ]
    assert row["phases"] == ["cold", "warm", "fu"]
    pin = json.loads(fr.PIN_PATH.read_text())
    cell = fr.audit_manifest([row], [fm.SMOKE_BATCH], pin, ["yunshu"])["cells"][0]
    assert cell["phases"] == ["cold", "warm", "turn2"]


def test_idle_row_is_a_decode_run_with_the_idle_gap_as_think_s(monkeypatch):
    """Regression: idle5 rows were skipped by harness_part (turns not in the tfbench CLI), so the
    manifest's idle cells never ran. They map to cold + turn2 with --think-s = the idle gap."""
    monkeypatch.setattr(fr, "IDLE_THINK_S", 5.0)
    sp = fm.spec(
        sweep="idle5",
        ctx="8K",
        kind="prose",
        out=64,
        turns="idle5",
        phases=("cold", "fu_idle"),
    )
    part, extra, reason = fr.harness_part(_rows("yunshu", [sp])[0])
    assert reason is None and part == "decode"
    assert extra[extra.index("--phases") + 1] == "cold,turn2"
    assert extra[extra.index("--think-s") + 1] == "5.0"


def test_stub_usage_shapes_match_the_tfbench_reader():
    import tfbench as tb

    details = sess.usage_object("details", 100, 8, 40)
    assert tb.cached_tokens(details) == 40
    assert details["prompt_tokens"] == 100 and details["completion_tokens"] == 8
    none = sess.usage_object("unavailable", 100, 8, 40)
    assert tb.cached_tokens(none) is None
    assert set(sess.USAGE_SHAPE) >= set(ENGINES_UNDER_TEST)
