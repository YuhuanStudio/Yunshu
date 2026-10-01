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
