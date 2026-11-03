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


def test_wrapper_runs_tfbench_against_stub_and_audit_is_clean(chain, monkeypatch):
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

    cells = sorted((chain["raw"] / "yunshu").glob("*-r*.jsonl"))
    cells = [c for c in cells if not c.name.endswith(".telemetry.jsonl")]
    assert len(cells) == len(chain["rows"])
    for c in cells:
        tel = fa.telemetry_path(c)
        assert tel.exists(), f"telemetry beside {c.name}"

    manifest = fr.audit_manifest(chain["rows"], [fm.SMOKE_BATCH], chain["pin"], plan.engine_order)
    mpath = chain["work"] / "audit-manifest.json"
    # test only: the engine process is the stub (see fullcmp_audit.check_pinned_versions)
    manifest["stub_engine"] = True
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    rc = subprocess.run(
        [
            fr.HARNESS_PY,
            str(SCRIPTS / "fullcmp_audit.py"),
            "--root",
            str(chain["raw"]),
            "--manifest",
            str(mpath),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )
    assert rc.returncode == 0, rc.stdout[-4000:] + rc.stderr[-4000:]
    assert re.search(r"TOTAL: \d+/\d+ checks passed, 0 violations", rc.stdout), rc.stdout[-2000:]


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
