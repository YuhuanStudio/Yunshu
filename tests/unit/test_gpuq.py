"""Recovery must not turn an interrupted benchmark into a successful run."""

import importlib.util
import time
import types
from pathlib import Path

import pytest


@pytest.fixture
def gpuq(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/dev/gpuq.py"
    spec = importlib.util.spec_from_file_location("gpuq_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A private clock: patching sleep/monotonic on the shared time module would also patch other threads.
    module.time = types.SimpleNamespace(**vars(time))
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


@pytest.mark.serial
def test_adopted_job_ignoring_sigint_is_killed_after_grace(gpuq, tmp_path, monkeypatch):
    import signal

    clock = [1000.0]
    sent = []
    monkeypatch.setattr(gpuq, "_now", lambda: clock[0])
    monkeypatch.setattr(
        gpuq.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 5)
    )
    monkeypatch.setattr(gpuq.os, "killpg", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(
        gpuq, "_alive", lambda pid: signal.SIGKILL not in sent and clock[0] < 1500
    )
    job = {
        "id": "stuck",
        "pid": 123,
        "started": 900,
        "state": "running",
        "cancel": True,
    }
    path = tmp_path / "stuck.json"
    gpuq._write(path, job)
    gpuq._adopt(job, path)
    assert sent[0] == signal.SIGINT
    assert sent[-1] == signal.SIGKILL


def test_cancel_and_patch_take_the_job_lock(gpuq, tmp_path):
    """Job-file read-modify-write must hold the per-job lock (lost cancel / state updates)."""
    import fcntl
    import threading

    path = tmp_path / "j.json"
    gpuq._write(path, {"id": "j", "state": "pending"})
    with open(str(path) + ".lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        t = threading.Thread(
            target=gpuq._patch_job, args=(path,), kwargs={"cancel": True}
        )
        t.start()
        t.join(0.5)
        blocked = t.is_alive()
        fcntl.flock(held, fcntl.LOCK_UN)
        t.join(5)
    assert blocked
    assert gpuq._read(path)["cancel"] is True


def test_job_temp_files_stay_off_the_system_disk(tmp_path, monkeypatch):
    """A job's TMPDIR is a per-job dir under GPUQ_DIR, removed when the job ends (Core ML compiles in the default
    system-disk TMPDIR piled up to 480 GB, 2026-10-25)."""
    import importlib
    import os
    import sys

    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dev"))
    import gpuq

    gpuq = importlib.reload(gpuq)
    gpuq.JOBS.mkdir(parents=True, exist_ok=True)
    gpuq.LOGS.mkdir(parents=True, exist_ok=True)
    seen = tmp_path / "seen"
    code = f"import os,tempfile;open(tempfile.mktemp(),'w').write('x');open({str(seen)!r},'w').write(os.environ['TMPDIR'])"
    job = {
        "id": "tmpjob",
        "cmd": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "env": {**os.environ, "TMPDIR": "/private/var/folders/system-disk"},
        "timeout_s": 60,
        "stall_s": 60,
        "priority": 0,
        "submitted": 0.0,
        "state": "pending",
    }
    path = gpuq.JOBS / "tmpjob.json"
    gpuq._write(path, job)
    gpuq._run_one(job, path)
    assert job["state"] == "done"
    assert seen.read_text() == str(tmp_path / "tmp" / "tmpjob")
    assert not (tmp_path / "tmp" / "tmpjob").exists()


def test_finished_job_records_its_result_summary(tmp_path, monkeypatch):
    """A finished job keeps its last meaningful output in its record, so its result survives a dead reader."""
    import importlib
    import os
    import sys

    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dev"))
    import gpuq

    gpuq = importlib.reload(gpuq)
    gpuq.JOBS.mkdir(parents=True, exist_ok=True)
    gpuq.LOGS.mkdir(parents=True, exist_ok=True)
    code = "print('loading'); print('{\"tps\": 41.5, \"complete\": true}')"
    job = {
        "id": "sumjob",
        "cmd": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "env": dict(os.environ),
        "timeout_s": 60,
        "stall_s": 60,
        "priority": 0,
        "submitted": 0.0,
        "state": "pending",
    }
    path = gpuq.JOBS / "sumjob.json"
    gpuq._write(path, job)
    gpuq._run_one(job, path)
    stored = gpuq._read(path)
    assert '"tps": 41.5' in stored["summary"]
    assert "gpuq:" not in stored["summary"]
