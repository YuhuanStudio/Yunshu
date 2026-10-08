"""Recovery must not turn an interrupted benchmark into a successful run."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def gpuq(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/dev/gpuq.py"
    spec = importlib.util.spec_from_file_location("gpuq_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOGS", tmp_path)
    monkeypatch.setattr(module, "_alive", lambda pid: False)
    return module


@pytest.mark.parametrize(
    "exit_status,expected",
    [(None, "lost"), ("invalid", "lost"), ("0", "done"), ("1", "failed")],
)
def test_adoption_requires_exit_status(gpuq, tmp_path, exit_status, expected):
    job = {"id": "interrupted", "pid": 123, "started": 1, "state": "running"}
    (tmp_path / "interrupted.log").write_text("139/1892 correct=True\n")
    if exit_status is not None:
        (tmp_path / "interrupted.rc").write_text(exit_status)
    gpuq._adopt(job, tmp_path / "interrupted.json")
    assert job["state"] == expected
    assert job["adopted"] is True


def test_failing_command_is_recorded_failed(tmp_path, monkeypatch):
    """A command that crashes must be 'failed' with its own exit status, not 'done' / 0
    (the shell wrapper ends in `echo`, whose status used to be taken as the job's)."""
    import importlib
    import sys
    from pathlib import Path

    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dev"))
    import gpuq

    gpuq = importlib.reload(gpuq)
    gpuq.JOBS.mkdir(parents=True, exist_ok=True)
    gpuq.LOGS.mkdir(parents=True, exist_ok=True)
    job = {
        "id": "crash",
        "cmd": [sys.executable, "-c", "raise KeyError('agent')"],
        "cwd": str(tmp_path),
        "env": dict(__import__("os").environ),
        "timeout_s": 60,
        "stall_s": 60,
        "priority": 0,
        "submitted": 0.0,
        "state": "pending",
    }
    path = gpuq.JOBS / "crash.json"
    gpuq._write(path, job)
    gpuq._run_one(job, path)
    assert job["state"] == "failed"
    assert job["rc"] == 1


def test_gate_job_runs_before_same_priority_backlog(tmp_path, monkeypatch):
    """A --gate verdict job must not wait behind a long p0 backlog (2026-10-08: the
    release gate waited behind 40 agentbench p0 jobs), but never beats higher priority."""
    import importlib
    import sys

    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dev"))
    import gpuq

    gpuq = importlib.reload(gpuq)

    def job(i, t, **kw):
        return {
            "id": i,
            "label": i,
            "state": "pending",
            "submitted": t,
            "priority": 0,
            **kw,
        }

    backlog = [job(f"bench{n}", n) for n in range(5)]
    gate = job("release-gate", 100, gate=True)
    assert gpuq._pick(backlog + [gate])["id"] == "release-gate"
    assert gpuq._pick(backlog)["id"] == "bench0"
    higher = job("urgent", 200, priority=1)
    assert gpuq._pick(backlog + [gate, higher])["id"] == "urgent"
    low_gate = job("lowgate", __import__("time").time(), gate=True, priority=-1)
    assert gpuq._pick(backlog + [low_gate])["id"] == "bench0"
