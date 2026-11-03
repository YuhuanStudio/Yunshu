"""wait_all summary (scripts/research/fullcmp_run.py): after the one blocking gpuq wait it reads every
job's own record and fails if ANY job failed or timed out, naming the engines.

Fake gpuq only: the wait is a stub, the job records are JSON files in a temp GPUQ_DIR. No GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_run as frun  # noqa: E402


class _Done:
    returncode = 0


def _record(jobs_dir: Path, jid: str, state: str, rc):
    (jobs_dir / f"{jid}.json").write_text(json.dumps({"id": jid, "state": state, "rc": rc}))


@pytest.fixture
def fake_gpuq(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(frun, "gpuq_env", lambda: {"GPUQ_DIR": str(tmp_path)})
    frun.JOB_OF_ID.clear()
    return jobs


def _jobs(engines):
    return [
        frun.Job(label=f"lbl-{e}", batch="B1", engine=e, group="g", rep=0, cells=[], est_min=1.0, cmd=[])
        for e in engines
    ]


def test_mixed_results_fail_and_name_every_failing_engine(fake_gpuq, monkeypatch, capsys):
    engines = ["tf-1.2.1", "omlx-0.7.3", "mlx-serve-26.11.1", "kapitsa-0.2.26", "splash-1.5.0", "yunshu"]
    ids = [f"1104-00000{i}-00-x" for i in range(len(engines))]
    for jid, job in zip(ids, _jobs(engines)):
        frun.JOB_OF_ID[jid] = job
    # the single wait returns 0 (it saw one failure as 'done'): the records must still decide
    monkeypatch.setattr(frun.subprocess, "run", lambda *a, **k: _Done())
    _record(fake_gpuq, ids[0], "done", 0)
    _record(fake_gpuq, ids[1], "failed", 1)
    _record(fake_gpuq, ids[2], "failed", 2)
    _record(fake_gpuq, ids[3], "timeout", None)
    _record(fake_gpuq, ids[4], "failed", 127)
    _record(fake_gpuq, ids[5], "done", 0)

    rc = frun.wait_all(ids)

    out = capsys.readouterr()
    assert rc == 1
    for jid, engine, rc_val, state in [
        (ids[0], "tf-1.2.1", 0, "done"),
        (ids[1], "omlx-0.7.3", 1, "failed"),
        (ids[3], "kapitsa-0.2.26", None, "timeout"),
    ]:
        assert f"{jid} | {engine} | {rc_val} | {state}" in out.out
    assert "4 of 6 job(s) failed or timed out" in out.err
    for engine in ("omlx-0.7.3", "mlx-serve-26.11.1", "kapitsa-0.2.26", "splash-1.5.0"):
        assert engine in out.err
    assert "tf-1.2.1" not in out.err.split("engines:")[1]


def test_all_done_exits_zero(fake_gpuq, monkeypatch, capsys):
    ids = ["1104-000001-00-a", "1104-000002-00-b"]
    for jid, job in zip(ids, _jobs(["tf-1.2.1", "yunshu"])):
        frun.JOB_OF_ID[jid] = job
    monkeypatch.setattr(frun.subprocess, "run", lambda *a, **k: _Done())
    for jid in ids:
        _record(fake_gpuq, jid, "done", 0)
    assert frun.wait_all(ids) == 0
    assert "FAILED" not in capsys.readouterr().err


def test_missing_record_counts_as_failed(fake_gpuq, monkeypatch, capsys):
    ids = ["1104-000009-00-gone"]
    frun.JOB_OF_ID[ids[0]] = _jobs(["tf-1.1.8"])[0]
    monkeypatch.setattr(frun.subprocess, "run", lambda *a, **k: _Done())
    assert frun.wait_all(ids) == 1
    assert "tf-1.1.8" in capsys.readouterr().err


def test_nonzero_wait_exit_alone_still_fails(fake_gpuq, monkeypatch):
    ids = ["1104-000010-00-w"]
    frun.JOB_OF_ID[ids[0]] = _jobs(["yunshu"])[0]
    _record(fake_gpuq, ids[0], "done", 0)

    class _Bad:
        returncode = 3

    monkeypatch.setattr(frun.subprocess, "run", lambda *a, **k: _Bad())
    assert frun.wait_all(ids) == 1
