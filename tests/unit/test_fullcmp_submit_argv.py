"""The gpuq submit argv of every B1 job passes gpuq's own preflight (CPU only, no queue, no GPU).

Regression for the 2026-11-03 smoke: the job command was `/bin/sh -c "<main .venv python> fullcmp_quiet.py
gate ..."`, so gpuq preflight read the whole script as an input path and refused every submit. The job
now runs `/bin/sh <wrapper file>`; the body lives in the wrapper and every path it names is checked by
fullcmp_run.script_problems (gpuq preflight cannot see inside a job body).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))

import bench_engines as be  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import gpuq_preflight  # noqa: E402

MAIN_CHECKOUT = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu"

B1_ENGINES = sorted(e for e in be.MANIFEST_ENGINES if e not in be.UNSUPPORTED_ENGINES)


def _row(rid, engine, ctx=1024, kind="code", conc=1, group="27B"):
    return {
        "id": rid,
        "engine": engine,
        "model": group,
        "batch": "B1",
        "status": "unverified",
        "thinking": "off",
        "turns": "single",
        "media": "none",
        "sampling": "greedy",
        "prompt_kind": kind,
        "ctx": f"{ctx // 1024}K",
        "ctx_tokens": ctx,
        "out_len": 64,
        "conc": conc,
        "reps": 3,
        "est_gpu_min": 3.0,
        "table": "perf",
        "phases": ["conc"] if conc > 1 else ["decode"],
    }


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    """Every input the job names exists under tmp_path, every output root is under tmp_path too.
    Real engine dirs do not exist on a dev checkout, so the model and the pinned tree are fakes."""
    model = tmp_path / "models" / "fake-27b"
    model.mkdir(parents=True)
    tree = tmp_path / "tree" / "python"
    tree.mkdir(parents=True)
    bin_yunshu = tmp_path / "venv" / "bin" / "yunshu"
    bin_yunshu.parent.mkdir(parents=True)
    bin_yunshu.write_text("#!/bin/sh\n")
    py = tmp_path / "venv" / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    harness_py = tmp_path / "harness" / ".venv" / "bin" / "python"
    harness_py.parent.mkdir(parents=True)
    harness_py.write_text("#!/bin/sh\n")
    monkeypatch.setattr(fr, "MODEL_PATH", str(model))
    monkeypatch.setattr(fr, "HARNESS_PY", str(harness_py))
    monkeypatch.setattr(fr, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(fr, "HOME_ROOT", tmp_path / "home")
    monkeypatch.setattr(fr, "WRAP_ROOT", tmp_path / "wrap")
    monkeypatch.setattr(fr, "QUIET_DIR", tmp_path / "quiet")
    monkeypatch.setattr(fr, "PREFLIGHT_DIR", tmp_path / "preflight")
    monkeypatch.setattr(fr, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(fr, "TFOUT_ROOT", tmp_path / "tfout")
    pin = {
        "version": "0.1.7.dev0",
        "sha": "0f2960047855546a9f13b2631b680352b1c57cfe",
        "python_executable": str(py),
        "launch": {
            "harness_env": {
                "TFB_YUNSHU_BIN": str(bin_yunshu),
                "TFB_YUNSHU_SRC": str(tree),
                "PYTHONPATH": str(tree),
            }
        },
    }
    return {"pin": pin, "tmp": tmp_path}


def _b1_rows():
    rows = []
    for i, eng in enumerate(B1_ENGINES):
        rows.append(_row(f"c{i}a", eng))
        rows.append(_row(f"c{i}b", eng, conc=2))
    return rows


def _argv_problems(job, cwd):
    problems = list(gpuq_preflight.preflight(job.cmd, str(cwd), {}))
    return problems


def test_b1_covers_every_manifest_engine():
    # the test is only meaningful if it exercises every engine the batch can contain
    assert len(B1_ENGINES) >= 10, B1_ENGINES
    assert "yunshu" in B1_ENGINES  # manifest id; the tfbench id is yunshu-main


@pytest.mark.parametrize("profile_rep", [False, True])
def test_every_b1_submit_argv_passes_gpuq_preflight(fake_world, profile_rep):
    plan = fr.plan_batch(_b1_rows(), "B1", fake_world["pin"], profile_rep=profile_rep)
    assert plan.jobs, "B1 plan is empty"
    for job in plan.jobs:
        fr.write_wrapper(job)
        problems = _argv_problems(job, fr.REPO_ROOT)
        assert problems == [], f"{job.label}: {problems}"
        assert fr.script_problems(job.script) == [], job.label


def test_submit_argv_never_names_the_main_checkout(fake_world):
    plan = fr.plan_batch(_b1_rows(), "B1", fake_world["pin"], profile_rep=True)
    for job in plan.jobs:
        argv = job.submit_argv()[1:]  # argv[0] is the literal gpuq path (user rule)
        for el in argv:
            assert "/Yunshu/.venv" not in el, (job.label, el)
            assert MAIN_CHECKOUT not in el, (job.label, el)
        assert job.cmd[0] == "/bin/sh"
        assert Path(job.cmd[1]).is_absolute()
        assert MAIN_CHECKOUT not in job.script, job.label
        assert "/Yunshu/.venv" not in job.script, job.label


def test_gate_is_a_step_inside_the_wrapper_not_the_argv(fake_world):
    plan = fr.plan_batch(_b1_rows(), "B1", fake_world["pin"])
    for job in plan.jobs:
        assert "fullcmp_quiet.py" not in " ".join(job.cmd)
        first = job.script.splitlines()[1].split(" && ")[0]
        assert "fullcmp_quiet.py" in first and " gate " in first


def test_written_wrapper_is_an_executable_shell_script(fake_world):
    plan = fr.plan_batch(_b1_rows()[:2], "B1", fake_world["pin"])
    job = plan.jobs[0]
    path = fr.write_wrapper(job)
    text = path.read_text()
    assert text.startswith("#!/bin/sh\nset -e\n")
    assert text.rstrip().endswith(job.script.rstrip().splitlines()[-1])
    assert path.stat().st_mode & 0o111


def test_self_check_is_clean_for_every_b1_job(fake_world):
    plan = fr.plan_batch(_b1_rows(), "B1", fake_world["pin"], profile_rep=True)
    assert fr.self_check(plan.jobs) == {}


def test_self_check_refuses_a_missing_model(fake_world, monkeypatch):
    monkeypatch.setattr(fr, "MODEL_PATH", str(fake_world["tmp"] / "nope"))
    plan = fr.plan_batch(_b1_rows()[:2], "B1", fake_world["pin"])
    bad = fr.self_check(plan.jobs)
    assert bad, "a job with a missing model must not pass the self-check"
    assert any("input path does not exist" in p for ps in bad.values() for p in ps)


def test_job_body_naming_the_main_checkout_is_refused():
    assert fr.script_problems(
        "set -e\n/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python x.py"
    )
