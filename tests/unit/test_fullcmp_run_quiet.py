"""Quiet-machine wiring in scripts/research/fullcmp_run.py: every job gates first (in-job, with
retries and a log per label); a busy host refuses the batch before anything is submitted.

Fakes only: no gpuq call, no GPU, no real host sampling (gate is stubbed).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_run_quiet_t", ROOT / "scripts" / "research" / "fullcmp_run.py"
)
run = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_run_quiet_t"] = run
_SPEC.loader.exec_module(run)

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


def row(engine, cid):
    return {
        "id": cid,
        "batch": "B1",
        "engine": engine,
        "model": "q27-4bit",
        "thinking": "off",
        "ctx": "1K",
        "ctx_tokens": 1024,
        "prompt_kind": "code",
        "conc": 1,
        "out_len": 128,
        "status": "unverified",
        "reason": "",
        "reps": 1,
        "est_gpu_min": 2.0,
        "media": "none",
        "sampling": "greedy",
        "turns": "single",
        "table": "decode",
    }


def test_every_job_runs_the_quiet_gate_before_any_cell():
    job = run.plan_batch([row("yunshu", "B1.a")], "B1", PIN).jobs[0]
    script = job.cmd[2]
    steps = script.split(" && ")
    assert "fullcmp_quiet.py" in steps[0] and " gate " in steps[0]
    assert steps[1].startswith("mkdir -p")
    assert "tfbench.py" in steps[2]
    assert sum("fullcmp_quiet.py" in s for s in steps) == 1


def test_gate_log_is_named_by_the_job_label():
    job = run.plan_batch([row("yunshu", "B1.a")], "B1", PIN).jobs[0]
    first = job.cmd[2].split(" && ")[0]
    assert "--log" in first
    assert f"{job.label}.jsonl" in first
    assert str(run.QUIET_DIR) in first


def test_gate_argv_uses_the_main_venv_python_and_the_worktree_script():
    argv = run.quiet_gate_argv("fullcmp-B1-x-r0")
    assert argv[0] == run.MAIN_PY
    assert argv[1] == str(run.SCRIPTS / "fullcmp_quiet.py")
    assert argv[2] == "gate"


def test_rep3_jobs_are_gated_too():
    rows = {2: [row("yunshu", "B1.a")]}
    jobs = run.plan_jobs(rows, ["yunshu"], PIN, "B1", label_suffix="-rep3")
    assert "fullcmp_quiet.py" in jobs[0].cmd[2].split(" && ")[0]


def test_busy_host_refuses_the_batch_before_anything_is_submitted(
    tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row("yunshu", "B1.a")) + "\n")
    submitted = []
    monkeypatch.setattr(run, "PIN_PATH", tmp_path / "pin.json")
    (tmp_path / "pin.json").write_text(json.dumps(PIN))
    monkeypatch.setattr(run, "gpuq_status_text", lambda: "")
    monkeypatch.setattr(
        run.fullcmp_quiet,
        "gate",
        lambda **kw: {
            "passed": False,
            "total_foreign_pct": 31.5,
            "holders": [{"pid": 600, "name": "mds_stores", "pct": 31.5}],
        },
    )
    monkeypatch.setattr(
        run, "swap_gate", lambda b: pytest.fail("swap gate must not run")
    )
    monkeypatch.setattr(
        run, "submit_all", lambda jobs: submitted.append(jobs) or ([], 0)
    )
    rc = run.main(["--batch", "B1", "--manifest", str(manifest)])
    assert rc == 2
    assert submitted == []
    err = capsys.readouterr().err
    assert "host busy before batch B1" in err and "mds_stores" in err


def test_every_cell_command_carries_its_job_label_and_gpuq_dir():
    rows = [row("yunshu", "B1.a"), row("yunshu", "B1.b")]
    job = run.plan_batch(rows, "B1", PIN).jobs[0]
    cell_cmds = [s for s in job.cmd[2].split(" && ") if "tfbench.py" in s]
    assert len(cell_cmds) == 2
    for cmd in cell_cmds:
        assert f"TFB_JOB_LABEL={job.label}" in cmd
        assert f"GPUQ_DIR={run.GPUQ_DIR_DEFAULT}" in cmd
        assert "GPUQ_JOB_ID" not in cmd  # never set by the driver: gpuq exports it (or it resolves)


def test_cell_env_without_a_label_sets_no_job_label():
    env = run.cell_env("yunshu", PIN)
    assert not any(e.startswith("TFB_JOB_LABEL=") for e in env)
    assert f"GPUQ_DIR={run.GPUQ_DIR_DEFAULT}" in env
