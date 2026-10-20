"""Line holding: big jobs run line by line instead of all lines advancing a cell at a time."""

import importlib
import sys
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


def job(i, wt, **kw):
    base = {
        "id": i,
        "label": i,
        "state": "pending",
        "submitted": NOW - 3600,
        "priority": 0,
        "timeout_s": 3600,
        "cwd": f"/x/Yunshu-wt-{wt}",
    }
    return {**base, **kw}


def ran(i, wt, start_ago, dur, **kw):
    return job(
        i, wt, state="done", started=NOW - start_ago, ended=NOW - start_ago + dur, **kw
    )


def test_holder_keeps_lane_while_it_has_long_jobs(gpuq):
    # Round-robin alone would pick line b (it ran longest ago); the holding line a goes on.
    hist = [ran("b0", "b", 120 * MIN, 30 * MIN), ran("a0", "a", 60 * MIN, 59 * MIN)]
    pending = [job("b1", "b", submitted=NOW - 7200), job("a1", "a", submitted=NOW - 60)]
    assert gpuq._pick(hist + pending)["id"] == "a1"


def test_holder_gap_waits_for_next_batch_but_short_jobs_still_run(gpuq):
    hist = [ran("a0", "a", 31 * MIN, 30 * MIN)]  # ended 60 s ago
    other_long = job("b1", "b")
    assert gpuq._pick(hist + [other_long]) is None
    smoke = job("b-smoke", "b", timeout_s=5 * MIN)
    assert gpuq._pick(hist + [other_long, smoke])["id"] == "b-smoke"


def test_holder_released_after_gap(gpuq):
    hist = [ran("a0", "a", 40 * MIN, 30 * MIN)]  # ended 10 min ago, no new job from a
    assert gpuq._pick(hist + [job("b1", "b")])["id"] == "b1"


def test_tenure_cap_hands_lane_over(gpuq):
    hist = [
        ran(f"a{i}", "a", (5 - i) * 60 * MIN, 60 * MIN) for i in range(5)
    ]  # 5 h streak
    pending = [job("a9", "a"), job("b1", "b")]
    assert gpuq._pick(hist + pending)["id"] == "b1"


def test_higher_priority_still_wins(gpuq):
    hist = [ran("a0", "a", 31 * MIN, 30 * MIN)]
    pending = [job("a1", "a"), job("b1", "b", priority=2)]
    assert gpuq._pick(hist + pending)["id"] == "b1"


def test_interleaved_short_does_not_take_holding(gpuq):
    hist = [
        ran("a0", "a", 40 * MIN, 30 * MIN),
        ran("b-short", "b", 10 * MIN, 2 * MIN, timeout_s=5 * MIN, interleaved=True),
    ]
    pending = [job("b1", "b"), job("a1", "a")]
    assert gpuq._pick(hist + pending)["id"] == "a1"
