"""End-to-end CPU chain for engine yunshu-main (no gpuq daemon, no GPU, no real model).

smoke manifest (fullcmp_manifest --smoke) -> fullcmp_run builds the job argv and wrapper script
-> gpuq's own preflight on the argv and the job body -> the wrapper runs locally against the stub
server (TFB_STUB_SERVER) with the env the runner exports (gpuq exports GPUQ_JOB_ID; the test sets
it to a well-formed id) -> real tfbench (real telemetry sampler) -> real fullcmp_audit: zero violations.

Only the stub replaces the engine. The quiet-machine gate runs for real inside the wrapper.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))

import fullcmp_audit as fa  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import fullcmp_attribution as fa_attr  # noqa: E402
import gpuq_preflight  # noqa: E402

STUB_DIR = ROOT / "tests" / "unit"
FAKE_JOB_ID = "2026-110301-01-fullcmp-SMOKE"


def _smoke_rows_under_test():
    """The --smoke decode rows (smoke-core, 1K code) for yunshu-main: prompt variants 0 and 1, so the
    >= 2 prompts per (model, ctx, kind) guard passes. The smoke conc rows are not used here: the
    audit's attribution guard covers decode rows, and the conc path has its own tests."""
    rows = fm.build_smoke_rows()
    keep = [r for r in rows if r["sweep"] == "smoke-core" and r["prompt_kind"] == "code"]
    assert {r["prompt_variant"] for r in keep} == {0, 1}
    return keep


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    work = tmp_path_factory.mktemp("fullcmp-e2e")
    raw = work / "raw"
    fr.configure(raw, work / "work")
    pin = json.loads(fr.PIN_PATH.read_text())
    rows = _smoke_rows_under_test()
    manifest = work / "smoke-manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    return {"work": work, "raw": raw, "pin": pin, "rows": rows, "manifest": manifest}


def _audit_manifest_for(chain, *, stub_engine=True):
    plan = fr.plan_batch(chain["rows"], fm.SMOKE_BATCH, chain["pin"])
    manifest = fr.audit_manifest(chain["rows"], [fm.SMOKE_BATCH], chain["pin"], plan.engine_order)
    if stub_engine:
        # test only: the engine process is the stub (see fullcmp_audit.check_pinned_versions)
        manifest["stub_engine"] = True
    return manifest


def _run_audit(root, manifest, work):
    mpath = work / "audit-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return subprocess.run(
        [
            fr.HARNESS_PY,
            str(SCRIPTS / "fullcmp_audit.py"),
            "--root",
            str(root),
            "--manifest",
            str(mpath),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )


@pytest.fixture(scope="module")
def good_run(chain):
    """One real pass of the chain (wrapper -> stub -> tfbench -> telemetry). The mutation tests copy
    its output and break one thing each."""
    plan = fr.plan_batch(chain["rows"], fm.SMOKE_BATCH, chain["pin"])
    job = plan.jobs[0]
    wrapper = fr.write_wrapper(job)
    # The runner's job env: gpuq exports GPUQ_JOB_ID to the running job; the stub replaces the
    # engine process only. Everything else comes from the wrapper's own `env ...` assignments.
    env = dict(os.environ)
    env["GPUQ_JOB_ID"] = FAKE_JOB_ID
    env["TFB_STUB_SERVER"] = str(STUB_DIR)
    proc = subprocess.run(
        ["/bin/sh", str(wrapper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return {"plan": plan, "job": job, "raw": chain["raw"], "work": chain["work"]}


def _cell_files(root):
    return sorted(
        c
        for c in (root / "yunshu").glob("*-r*.jsonl")
        if not c.name.endswith(".telemetry.jsonl")
    )


def test_wrapper_runs_tfbench_against_stub_and_audit_is_clean(chain, good_run):
    cells = _cell_files(chain["raw"])
    assert len(cells) == len(chain["rows"])
    for c in cells:
        assert fa.telemetry_path(c).exists(), f"telemetry beside {c.name}"
    res = _run_audit(chain["raw"], _audit_manifest_for(chain), chain["work"])
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert re.search(r"TOTAL: \d+/\d+ checks passed, 0 violations", res.stdout), res.stdout[-2000:]


def test_smoke_manifest_plans_one_job_with_the_gpuq_argv(chain):
    plan = fr.plan_batch(chain["rows"], fm.SMOKE_BATCH, chain["pin"])
    assert not plan.skipped
    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.engine == "yunshu"
    assert job.cell_ids == [r["id"] for r in chain["rows"]]
    assert job.cmd[0] == "/bin/sh" and job.cmd[1].endswith(".sh")
    assert job.submit_argv()[0] == fr.GPUQ


def test_gpuq_preflight_and_job_body_pass_for_the_smoke_job(chain):
    job = fr.plan_batch(chain["rows"], fm.SMOKE_BATCH, chain["pin"]).jobs[0]
    fr.write_wrapper(job)
    assert gpuq_preflight.preflight(job.cmd, str(ROOT), dict(os.environ)) == []
    assert fr.script_problems(job.script) == []


def test_runner_manifest_carries_expected_spec_mode_and_weight_group(chain):
    """Regression (e2e, 2026-11-03): the runner's audit manifest left spec_mode and weight_group
    null for every cell, so every production run failed closed on engaged mode."""
    manifest = fr.audit_manifest(chain["rows"], [fm.SMOKE_BATCH], chain["pin"], ["yunshu"])
    cells = manifest["cells"]
    assert cells
    for c in cells:
        assert c["spec_mode"] == "dflash"
        assert c["weight_group"] == "qwen3.8-27b-4bit-jundot"


def test_engaged_mode_without_an_expected_mode_is_a_violation_not_a_crash():
    """Regression: check_engaged_mode called .lower() on a None expected mode (AttributeError)."""
    import types

    cell = {"cell": "c0", "spec_mode": None}
    header = {
        "requested_spec_mode": "dflash",
        "spec_mode_expected": "dflash",
        "engaged_spec_mode": "dflash",
        "engaged_log_line": "VLM batch runner: draft=dflash",
    }
    run = types.SimpleNamespace(
        files=[types.SimpleNamespace(header=header, cell=cell, rep=0)]
    )
    res = fa.check_engaged_mode(run)
    assert res.violations, "a missing expected mode must be reported"
    assert any("no expected spec mode" in str(v) for v in res.violations)


def test_attribution_schema_resolves_for_the_bench_id_yunshu_main():
    """Regression (e2e, 2026-11-03): tfbench looked the attribution schema up by the bench id
    (yunshu-main), which has no schema; the decode path raised KeyError on the first request."""
    import bench_engines as be

    assert be.manifest_id("yunshu-main") == "yunshu"
    assert fa_attr.engine_attr(be.manifest_id("yunshu-main")).engine == "yunshu"
    assert be.manifest_id("not-an-engine") == "not-an-engine"


def test_memory_part_accepts_a_zero_gib_delta_and_rejects_no_samples():
    """Regression (e2e, 2026-11-03): a real 0.0 GiB delta was reported as 'no data'."""
    import os

    import pytest as _pytest

    import tfbench as tb

    mem = tb.MemSampler(os.getpid(), baseline_bytes=1, used_fn=lambda: 1)
    idle = mem.sample()
    assert idle == 0.0
    assert tb.memory_peak_or_raise(mem, idle) == 0.0
    def unreadable():
        raise OSError("vm_stat unavailable")

    empty = tb.MemSampler(os.getpid(), baseline_bytes=1, used_fn=unreadable)
    with _pytest.raises(RuntimeError, match="no data"):
        tb.memory_peak_or_raise(empty, empty.sample())


# ---------------------------------------------------------------- mutations (step 2)
# Each one breaks the good chain output (or its env) in one place and must fail closed with the
# named message. The good run is shared (good_run); the audit runs on a copy.


def _env_pairs(cell_env_list):
    return dict(item.split("=", 1) for item in cell_env_list if "=" in item)


def _cell_env_for(chain):
    row = chain["rows"][0]
    return _env_pairs(fr.cell_env(row["engine"], chain["pin"], "fullcmp-x", group=fr.comparison_group(row)))


def test_mutation_drop_run_id_fails_closed_before_any_cell(chain):
    import fullcmp_provenance as fp

    env = _cell_env_for(chain)
    env.pop("TFB_RUN_ID")
    env["GPUQ_JOB_ID"] = FAKE_JOB_ID
    with pytest.raises(fp.ProvenanceError, match="requires TFB_RUN_ID"):
        fp.check_required_env(env, jobs_dir=chain["work"])


def test_mutation_drop_group_fails_closed_before_any_cell(chain):
    import fullcmp_provenance as fp

    env = _cell_env_for(chain)
    env.pop("TFB_GROUP")
    env["GPUQ_JOB_ID"] = FAKE_JOB_ID
    with pytest.raises(fp.ProvenanceError, match="requires TFB_GROUP"):
        fp.check_required_env(env, jobs_dir=chain["work"])


def _copy_raw(good, tmp_path):
    import shutil

    dst = tmp_path / "raw"
    shutil.copytree(good["raw"], dst)
    return dst


def test_mutation_drop_header_field_is_a_violation(chain, good_run, tmp_path):
    root = _copy_raw(good_run, tmp_path)
    for c in _cell_files(root):
        lines = c.read_text().splitlines()
        header = json.loads(lines[0])
        header.pop("release_date")
        lines[0] = json.dumps(header)
        c.write_text("\n".join(lines) + "\n")
    res = _run_audit(root, _audit_manifest_for(chain), tmp_path)
    assert res.returncode != 0
    assert "release_date not recorded" in res.stdout


def test_mutation_wrong_binary_path_is_a_violation(chain, good_run, tmp_path):
    """Without the test-only stub flag the server must be the pinned venv python. The stub's command
    line is not, so the audit must reject it (the wrong binary is caught, not waved through)."""
    res = _run_audit(good_run["raw"], _audit_manifest_for(chain, stub_engine=False), tmp_path)
    assert res.returncode != 0
    assert "Yunshu not started by its pinned venv python" in res.stdout


def test_mutation_missing_telemetry_is_a_violation(chain, good_run, tmp_path):
    root = _copy_raw(good_run, tmp_path)
    victim = _cell_files(root)[0]
    fa.telemetry_path(victim).unlink()
    res = _run_audit(root, _audit_manifest_for(chain), tmp_path)
    assert res.returncode != 0
    assert "missing telemetry file" in res.stdout


@pytest.mark.xfail(
    strict=True,
    reason="open gap: a single-engine group has no reference prompt sha. prompt_identity compares "
    "engines within a group, so a wrong prompt sha in a one-engine smoke passes. Needs a per-prompt "
    "expected sha in the manifest.",
)
def test_mutation_wrong_prompt_sha_is_a_violation(chain, good_run, tmp_path):
    root = _copy_raw(good_run, tmp_path)
    victim = _cell_files(root)[0]
    lines = victim.read_text().splitlines()
    for i in range(1, len(lines)):
        row = json.loads(lines[i])
        if "prompt_ids_sha" in row:
            row["prompt_ids_sha"] = "0" * 16
        lines[i] = json.dumps(row)
    victim.write_text("\n".join(lines) + "\n")
    res = _run_audit(root, _audit_manifest_for(chain), tmp_path)
    assert res.returncode != 0
