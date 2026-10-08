"""Daemon CPU cost with a large job history, and the filler for a CPU-quiet-blocked head."""

import importlib
import json
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts" / "dev"
NOW = 1_000_000.0


@pytest.fixture
def gpuq(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(DEV))
    import gpuq as module

    module = importlib.reload(module)
    monkeypatch.setattr(module.time, "time", lambda: NOW)
    monkeypatch.setattr(module, "_now", lambda: NOW)
    module.JOBS.mkdir(parents=True, exist_ok=True)
    return module


def mk(gpuq, i, **kw):
    j = {
        "id": i,
        "label": i,
        "state": "pending",
        "submitted": NOW - 3600,
        "priority": 0,
        "timeout_s": 3600,
        "cwd": "/x/wt-a",
        "env": {},
        "mem_gb": 0.0,
        **kw,
    }
    gpuq._write(gpuq.JOBS / f"{i}.json", j)
    return j


# ---------------------------------------------------------------- daemon CPU


def test_job_json_reads_per_poll_are_bounded(gpuq, monkeypatch):
    for n in range(300):
        mk(gpuq, f"old{n}", state="done", started=1.0, ended=2.0)
    mk(gpuq, "act", state="pending")
    first = gpuq._jobs()
    assert len(first) == 301
    reads = []
    real = gpuq._read
    monkeypatch.setattr(gpuq, "_read", lambda p: reads.append(p) or real(p))
    for _ in range(10):  # ten polls with nothing changed: no job file is parsed again
        rows = gpuq._jobs()
    assert reads == [] or all("priority_caps" in str(p) for p in reads)
    assert len(rows) == 301


def test_jobs_returns_private_copies(gpuq):
    mk(gpuq, "act", state="pending", env={"a": 1})
    mk(gpuq, "fin", state="done", started=1.0, ended=2.0)
    a = {j["id"]: j for j in gpuq._jobs()}
    a["act"]["env"]["a"] = 99
    a["act"]["waiting"] = "x"
    a["fin"]["state"] = "pending"
    b = {j["id"]: j for j in gpuq._jobs()}
    assert b["act"]["env"]["a"] == 1 and "waiting" not in b["act"]
    assert b["fin"]["state"] == "done"


def test_changed_file_is_picked_up(gpuq):
    mk(gpuq, "a", priority=0)
    assert [j["id"] for j in gpuq._jobs()] == ["a"]
    mk(gpuq, "b", priority=1, state="pending")
    assert [j["id"] for j in gpuq._jobs()] == ["b", "a"]
    path = gpuq.JOBS / "a.json"
    d = json.loads(path.read_text())
    d.update(state="running", extra="x" * 50)
    gpuq._write(path, d)
    assert next(j for j in gpuq._jobs() if j["id"] == "a")["state"] == "running"


# ---------------------------------------------------------------- filler


class Gate:
    enabled = False
    reserve_gb = 0.0
    cpu = object()  # present: quiet jobs go through _cpu_blocker

    def may_start(self):
        return True


@pytest.fixture
def blocked_head(gpuq, monkeypatch):
    monkeypatch.setattr(
        gpuq, "_cpu_blocker", lambda job, gate: "cpu" if job.get("quiet") else None
    )
    monkeypatch.setattr(gpuq, "mem_blocked", lambda job, gate, free=None: False)
    return gpuq


def head(gpuq, **kw):
    history(gpuq)
    return mk(
        gpuq,
        "head",
        quiet=True,
        quiet_wait_started=NOW - 100,
        quiet_hold_started=NOW - 100,
        quiet_hold_limit_s=300,
        **kw,
    )


def history(gpuq):
    """An hour of long-job time: the short-lane budget allows 9 interleaved minutes."""
    mk(gpuq, "hist", state="done", started=NOW - 3700, ended=NOW - 100)


def admit(gpuq):
    return gpuq._admit(gpuq._jobs(), Gate())


def test_filler_runs_while_head_waits_for_quiet(blocked_head):
    g = blocked_head
    head(g)
    mk(
        g, "long-cell", timeout_s=3600, priority=-1, submitted=NOW - 10
    )  # not bounded: never a filler
    mk(g, "smoke", timeout_s=120, priority=-1, submitted=NOW - 10)
    mk(g, "quiet-other", quiet=True, timeout_s=120, priority=-1, submitted=NOW - 10)
    job = admit(g)
    assert job["id"] == "smoke" and job["filler"] is True


def test_head_waits_for_its_window_before_any_filler(blocked_head):
    g = blocked_head
    history(g)
    mk(
        g,
        "head",
        quiet=True,
        quiet_wait_started=NOW - 5,
        quiet_hold_started=NOW - 5,
        quiet_hold_limit_s=300,
    )  # fresh window
    mk(g, "smoke", timeout_s=120, priority=-1, submitted=NOW - 10)
    assert admit(g) is None


def test_head_runs_first_after_filler_ends(blocked_head, monkeypatch):
    g = blocked_head
    head(g)
    mk(g, "smoke1", timeout_s=120, priority=-1, submitted=NOW - 10)
    mk(g, "smoke2", timeout_s=120, priority=-1, submitted=NOW - 10)
    f = admit(g)
    assert f["filler"]
    # filler ran and ended; the head's window restarted (poll gap) so it has waited ~0 s
    done = json.loads((g.JOBS / f"{f['id']}.json").read_text())
    done.update(state="done", started=NOW - 90, ended=NOW - 5, interleaved=True)
    g._write(g.JOBS / f"{f['id']}.json", done)
    h2 = json.loads((g.JOBS / "head.json").read_text())
    h2.update(quiet_wait_started=NOW - 1, quiet_hold_started=NOW - 1)
    g._write(g.JOBS / "head.json", h2)
    assert admit(g) is None  # second filler held back: the head owns the next window
    # quiet window satisfied -> the head itself is admitted
    blocked_head_cpu = g._cpu_blocker
    g._cpu_blocker = lambda job, gate: None
    assert admit(g)["id"] == "head"
    g._cpu_blocker = blocked_head_cpu


def test_no_filler_when_head_blocked_by_memory(blocked_head, monkeypatch):
    g = blocked_head
    head(g, mem_gb=100.0)
    mk(g, "smoke", timeout_s=120, priority=-1, submitted=NOW - 10, mem_gb=1.0)
    monkeypatch.setattr(
        g, "mem_blocked", lambda job, gate, free=None: job["mem_gb"] > 50
    )
    job = admit(g)
    # the ordinary pick moves past a memory-blocked job; it is not a filler
    assert job is None or not job.get("filler")


def test_no_filler_when_head_is_not_cpu_blocked(blocked_head, monkeypatch):
    g = blocked_head
    head(g)
    mk(g, "smoke", timeout_s=120, priority=-1, submitted=NOW - 10)
    monkeypatch.setattr(g, "_cpu_blocker", lambda job, gate: None)
    assert admit(g)["id"] == "head"


def test_filler_respects_short_budget_and_excludes_unbounded(blocked_head):
    g = blocked_head
    head(g)
    mk(g, "used", state="done", started=NOW - 3000, ended=NOW - 2400, interleaved=True)
    mk(
        g, "smoke", timeout_s=120, priority=-1, submitted=NOW - 10
    )  # short, budget (9 min) already used
    assert admit(g) is None
    mk(
        g, "mid", timeout_s=20 * 60, priority=-1, submitted=NOW - 10
    )  # bounded non-short: allowed
    assert admit(g)["id"] == "mid"
