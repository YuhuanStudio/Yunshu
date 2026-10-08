"""Short-job lane: short verification jobs interleave between the cells of long suites."""

import contextlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts" / "dev"
NOW = 1_000_000.0
MIN = 60.0


@pytest.fixture
def gpuq(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(DEV))
    import gpuq as module

    module = importlib.reload(module)
    monkeypatch.setattr(module.time, "time", lambda: NOW)
    return module


def job(i, **kw):
    base = {
        "id": i,
        "label": i,
        "state": "pending",
        "submitted": NOW - 3600,
        "priority": 0,
        "timeout_s": 3600,
        "cwd": "/x/wt-a",
    }
    return {**base, **kw}


def done(i, start_ago, dur, **kw):
    return job(
        i,
        state="done",
        started=NOW - start_ago,
        ended=NOW - start_ago + dur,
        **kw,
    )


def history(minutes=60):
    """A long job that ran `minutes` minutes just before now."""
    return [done("hist", minutes * MIN + 60, minutes * MIN)]


def short(i, **kw):
    return job(i, **{"priority": -1, "timeout_s": 5 * MIN, **kw})


def test_short_runs_before_waiting_long_p0(gpuq):
    jobs = history() + [job("long0"), short("s1")]
    assert gpuq._pick(jobs, interleave=True)["id"] == "s1"
    assert gpuq._pick(jobs, interleave=True)["interleaved"] is True
    assert gpuq._pick(jobs)["id"] == "long0"  # strict priority without the lane


def test_short_must_have_waited_five_minutes(gpuq):
    jobs = history() + [job("long0"), short("s1", submitted=NOW - 60)]
    assert gpuq._pick(jobs, interleave=True)["id"] == "long0"


def test_alternation_short_long_short(gpuq):
    jobs = history(120) + [job("long0"), short("s1"), short("s2")]
    assert gpuq._pick(jobs, interleave=True)["id"] == "s1"
    # s1 ran (interleaved): the next pick must be the long job even with s2 waiting
    first = next(j for j in jobs if j["id"] == "s1")
    first.update(state="done", started=NOW - 300, ended=NOW - 240, interleaved=True)
    assert gpuq._pick(jobs, interleave=True)["id"] == "long0"
    # after a long job started more recently, a short may go again
    jobs.append(done("long-ran", 120, 60))
    jobs[jobs.index(next(j for j in jobs if j["id"] == "long0"))]["state"] = "pending"
    assert gpuq._pick(jobs, interleave=True)["id"] == "s2"


def test_budget_exhaustion_falls_back_to_strict_priority(gpuq):
    # 60 long min in the window -> 9 min allowed; 9 interleaved min already used
    jobs = history(60) + [
        done("old-s", 100 * MIN, 9 * MIN, interleaved=True, priority=-1),
        job("long0"),
        short("s1"),
    ]
    # the most recent started job is the long history job, so only the budget blocks
    used, long_, allowed = gpuq.interleave_budget(jobs, NOW)
    assert used >= allowed
    assert gpuq._pick(jobs, interleave=True)["id"] == "long0"
    jobs[0]["ended"] -= 10 * MIN  # less long time -> even less budget; sanity
    assert gpuq._pick(jobs, interleave=True)["id"] == "long0"


def test_no_long_history_means_no_budget(gpuq):
    jobs = [job("long0"), short("s1")]
    assert gpuq._pick(jobs, interleave=True)["id"] == "long0"


def test_budget_numbers(gpuq):
    jobs = history(60) + [done("s", 30 * MIN, 5 * MIN, interleaved=True)]
    used, long_, allowed = gpuq.interleave_budget(jobs, NOW)
    assert (used, long_) == (5 * MIN, 60 * MIN)
    assert allowed == pytest.approx(0.15 * 60 * MIN)


def test_gate_job_still_first(gpuq):
    jobs = history() + [job("gate0", gate=True, priority=0), short("s1")]
    assert gpuq._pick(jobs, interleave=True)["id"] == "gate0"
    jobs = history() + [job("long0"), short("s1"), short("gs", gate=True)]
    assert gpuq._pick(jobs, interleave=True)["id"] in {"long0", "s1"}


def test_p_minus_2_not_eligible(gpuq):
    jobs = history() + [job("long0"), short("big", priority=-2)]
    assert gpuq._pick(jobs, interleave=True)["id"] == "long0"


def test_ineligible_short_is_skipped(gpuq):
    jobs = history() + [job("long0"), short("s1"), short("s2")]
    assert gpuq._pick(jobs, lambda j: j["id"] != "s1", interleave=True)["id"] == "s2"


def test_short_flag_counts_as_short_and_caps_timeout(gpuq, tmp_path, monkeypatch):
    monkeypatch.setattr(gpuq.time, "time", time.time)  # real clock for submit
    jid = gpuq.submit(["true"], "x-short", 60, -1, short=True)
    j = json.loads((gpuq.JOBS / f"{jid}.json").read_text())
    assert j["short"] is True and j["timeout_s"] == gpuq.SHORT_MIN * 60
    assert gpuq._lane_short(j)
    jid2 = gpuq.submit(["true"], "x-long", 60, -1)
    j2 = json.loads((gpuq.JOBS / f"{jid2}.json").read_text())
    assert not gpuq._lane_short(j2)


def test_priority_preemption_never_interleaves(gpuq):
    """_admit(preempt=True) only considers p>=0 pending jobs and never the lane."""
    seen = []
    orig = gpuq._pick
    gpuq._pick = lambda jobs, el=None, interleave=False: (
        seen.append(interleave) or orig(jobs, el, interleave)
    )
    gate = gpuq.ServingGate()
    gpuq._admit([job("p0", timeout_s=3600)], gate, preempt=True)
    gpuq._pick = orig
    assert seen == [False]


def test_timeout_kills_short_job_and_logs_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(DEV))
    import gpuq

    gpuq = importlib.reload(gpuq)
    monkeypatch.setattr(gpuq, "POLL_S", 0.2)
    gpuq.JOBS.mkdir(parents=True)
    gpuq.LOGS.mkdir(parents=True)
    j = job(
        "sleeper",
        timeout_s=1,
        short=True,
        cmd=["sleep", "30"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        submitted=time.time(),
        stall_s=60,
    )
    path = gpuq.JOBS / "sleeper.json"
    gpuq._write(path, j)
    t0 = time.time()
    gpuq._run_one(j, path)
    assert j["state"] == "timeout" and time.time() - t0 < 20
    assert "short-lane job exceeded" in (gpuq.LOGS / "sleeper.log").read_text()


def test_end_to_end_real_daemon_start_order(tmp_path):
    """Isolated GPUQ_DIR, real daemon, `sleep` jobs: p0 long-declared cells with p-1 shorts
    that waited past GPUQ_SHORT_WAIT_S interleave; a running cell is never interrupted."""
    d = tmp_path / "q"
    d.mkdir()
    env = {
        **os.environ,
        "GPUQ_DIR": str(d),
        "GPUQ_NO_PREFLIGHT": "1",
        "GPUQ_SHORT_WAIT_S": "1",
        "GPUQ_SHORT_SHARE": "1.0",
        "GPUQ_DIR_ISOLATED": "1",
    }
    g = [sys.executable, str(DEV / "gpuq.py")]

    def submit(label, prio, tmin, secs):
        r = subprocess.run(
            [*g, "submit", "--label", label, "--priority", str(prio), "--timeout", str(tmin),
             "--serving-ok", "--mem-gb", "0", "--", "sleep", str(secs)],
            env=env, capture_output=True, text=True, cwd=tmp_path,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        return r.stdout.strip().splitlines()[-1]

    ids = []
    try:
        ids.append(submit("e2e-long0", 0, 60, 4))
        ids += [submit(f"e2e-long{n}", 0, 60, 4) for n in (1, 2, 3)]
        ids += [submit(f"e2e-short{n}", -1, 5, 1) for n in (1, 2)]
        deadline = time.time() + 120
        rows = {}
        while time.time() < deadline:
            rows = {
                i: json.loads(p.read_text())
                for p in (d / "jobs").glob("*.json")
                for i in [p.stem]
            }
            if len(rows) == 6 and all(
                r["state"] in ("done", "failed") for r in rows.values()
            ):
                break
            time.sleep(1)
        assert all(r["state"] == "done" for r in rows.values()), rows
        order = [r["label"] for r in sorted(rows.values(), key=lambda r: r["started"])]
        assert order[0] == "e2e-long0"
        # shorts waited >1 s while long0 ran, so one runs right after it, alternating
        assert order[1].startswith("e2e-short"), order
        assert order[2].startswith("e2e-long"), order
        assert order[3].startswith("e2e-short"), order
        for r in rows.values():  # nothing was preempted/paused
            assert not r.get("pauses")
        assert sum(1 for r in rows.values() if r.get("interleaved")) == 2
    finally:
        pid = (
            (d / "daemon.lock").read_text().strip()
            if (d / "daemon.lock").exists()
            else ""
        )
        if pid.isdigit():
            with contextlib.suppress(OSError):
                os.kill(int(pid), 9)
