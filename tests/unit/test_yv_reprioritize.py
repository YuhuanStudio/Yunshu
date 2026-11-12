"""yv reprioritize: a live run's later cells and its queued gpuq jobs take the new priority."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "scripts/dev"
sys.path.insert(0, str(REPO / "scripts"))


class FakeGpuq:
    def __init__(self, pending):
        self.pending = set(pending)
        self.submits = []
        self.reprio = []

    def submit(self, label, argv, *, priority=0, **kw):
        jid = f"new{len(self.submits)}"
        self.submits.append((jid, priority))
        return jid

    def reprioritize(self, jid, priority):
        self.reprio.append((jid, priority))
        return jid in self.pending


@pytest.fixture
def run(tmp_path):
    from verify.core import RunDir

    rd = RunDir(tmp_path / "run")
    rd.append("speed", {"ev": "cell_submitted", "cell": "a", "job": "j1", "attempt": 1})
    rd.append(
        "speed",
        {
            "ev": "cell_submitted",
            "cell": "b",
            "job": "j2",
            "attempt": 1,
            "inherit": False,
        },
    )
    rd.append("speed", {"ev": "cell_submitted", "cell": "c", "job": "j3", "attempt": 1})
    rd.append("speed", {"ev": "cell_done", "cell": "c", "job": "j3", "ok": True})
    return rd


def test_reprioritize_moves_queued_inheriting_jobs_and_later_cells(run):
    from verify.execute import Cell, Executor, reprioritize_run

    gq = FakeGpuq(pending={"j1", "j2"})
    moved = reprioritize_run(run, gq, 3)

    assert moved == ["j1"]  # j2 has an explicit cell priority; j3 already finished
    assert gq.reprio == [("j1", 3)]

    ex = Executor(run, gq, "lbl", priority=0)
    assert ex.priority == 3
    ex._submit(Cell(stage="speed", key="d", argv=["x"]), 1)
    assert gq.submits == [("new0", 3)]


def test_executor_keeps_base_priority_without_override(tmp_path):
    from verify.core import RunDir
    from verify.execute import Executor

    rd = RunDir(tmp_path / "run")
    assert Executor(rd, FakeGpuq([]), "lbl", priority=2).priority == 2


def test_gpuq_reprioritize_only_touches_pending_jobs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("gpuq_reprio_test", DEV / "gpuq.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "p.json").write_text('{"id": "p", "state": "pending", "priority": 0}')
    (jobs / "r.json").write_text('{"id": "r", "state": "running", "priority": 0}')

    assert q._reprioritize(jobs / "p.json", 3) is True
    assert q._read(jobs / "p.json")["priority"] == 3
    assert q._reprioritize(jobs / "r.json", 3) is False
    assert q._read(jobs / "r.json")["priority"] == 0
