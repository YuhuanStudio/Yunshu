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
