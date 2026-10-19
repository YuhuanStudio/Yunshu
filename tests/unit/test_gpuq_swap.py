"""CPU-only Darwin swap attribution and recorded historical policy regressions."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/dev"))
import gpuq_contention as h
from gpuq_swap import memory_verdict, process_memory


@pytest.mark.parametrize(
    "host,own,free,expected",
    [
        (400.7, 0, 119000, []),  # recorded bench016 27B false host-only rejection
        (3977.7, 0, 119000, ["swap"]),  # recorded fnmerge genuine pressure
        (0, 65, 119000, ["job_swap"]),
        (0, 0, 4095, ["memory_pressure"]),
        (None, 0, 119000, ["swap_unreadable"]),
        (0, None, 119000, ["job_swap_unreadable"]),
        (0, 0, None, ["memory_pressure_unreadable"]),
    ],
)
def test_memory_policy(host, own, free, expected):
    assert memory_verdict(host, own, free, h.DEFAULTS) == expected


@pytest.mark.parametrize("priority", [2, 1, 0, -1, -2, -3, -9])
@pytest.mark.parametrize("quiet", [False, True])
def test_monitor_policy_independent_of_priority(tmp_path, monkeypatch, priority, quiet):
    job = dict(id="cpu-only", pid=10, priority=priority, quiet=quiet, env={})
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))

    def write(p, data):
        p.write_text(json.dumps(data))

    m = h.ContentionMonitor(
        job, path, tmp_path, write, lambda p: json.loads(p.read_text())
    )
    monkeypatch.setattr(
        h, "host_available_bytes", lambda: 119_000_000_000, raising=False
    )
    monkeypatch.setattr(
        h,
        "process_memory",
        lambda pid: dict(
            pid=pid, start=1, pageins=99999999, swapped_bytes=0, regions=1
        ),
        raising=False,
    )
    values = iter([0, 400_700_000, 4_000_000_000])
    monkeypatch.setattr(h, "swapout_bytes", lambda: next(values))
    m.sampler.owned = {10, 11}
    monkeypatch.setattr(
        m.sampler,
        "sample",
        lambda *a, **kw: dict(time=kw["now"], since=kw["now"] - 2, foreign_cpu_pct=0),
    )
    m.record("start", 100)
    m.record("poll", 102)
    assert not job["contended"]  # huge file pageins don't imply anonymous swap
    assert job["host_swap_mb"] == 400.7 and job["job_swap_mb"] == 0
    m.record("poll", 104)
    assert job["contended"] and job["contention_reasons"] == ["swap"]
    assert m.events == [[102, 104]]
    assert json.loads(path.read_text())["memory_policy_version"] == 2


def test_real_cpu_process_memory():
    import os

    if sys.platform != "darwin":
        pytest.skip("Darwin API")
    row = process_memory(os.getpid())
    assert row["regions"] > 0 and row["start"] > 0 and row["swapped_bytes"] >= 0


def test_historical_missing_evidence_cannot_be_promoted():
    from gpuq_swap_review import review

    records = json.loads(
        (Path(__file__).parents[1] / "fixtures/gpuq/swap_1018_1019.json").read_text()
    )
    small, large = [review(r) for r in records]
    assert small["host_rejection_clears"] and small["verdict"] == "UNVERIFIABLE"
    assert large["verdict"] == "REJECT" and not large["host_rejection_clears"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_memory_evidence_fails_closed(bad):
    assert memory_verdict(bad, 0, 99999, h.DEFAULTS) == ["swap_unreadable"]


def test_footprint_incomplete_output_fails_closed(monkeypatch):
    import gpuq_swap as s

    if sys.platform != "darwin":
        pytest.skip("Darwin API")
    import os

    monkeypatch.setattr(
        s.subprocess,
        "run",
        lambda *a, **kw: type("R", (), {"stdout": "truncated TOTAL"})(),
    )
    with pytest.raises(ValueError, match="incomplete footprint"):
        process_memory(os.getpid())


def test_tree_peak_retained_after_child_exit_and_read_errors(tmp_path, monkeypatch):
    job = dict(id="tree", pid=10, env={})
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    m = h.ContentionMonitor(
        job,
        path,
        tmp_path,
        lambda p, d: p.write_text(json.dumps(d)),
        lambda p: json.loads(p.read_text()),
    )
    monkeypatch.setattr(h, "host_available_bytes", lambda: 100_000_000_000)
    monkeypatch.setattr(h, "swapout_bytes", lambda: 0)
    m.sampler.owned = {10, 11}
    monkeypatch.setattr(
        m.sampler,
        "sample",
        lambda *a, **kw: dict(time=kw["now"], since=kw["now"] - 2, foreign_cpu_pct=0),
    )
    monkeypatch.setattr(
        h,
        "process_memory",
        lambda pid: dict(
            pid=pid, start=1, pageins=0, swapped_bytes=40_000_000, regions=1
        ),
    )
    m.record("start", 100)
    assert job["job_swap_mb"] == 80 and "job_swap" in job["contention_reasons"]
    m.sampler.owned = {10}
    m.record("poll", 102)
    assert job["job_swap_mb"] == 80

    def denied(pid):
        raise PermissionError("denied")

    monkeypatch.setattr(h, "process_memory", denied)
    monkeypatch.setattr(h.os, "kill", lambda *a: None)  # still alive
    m.record("poll", 104)
    assert (
        job["job_swap_mb"] is None
        and "job_swap_unreadable" in job["contention_reasons"]
    )


def test_review_preserves_unreadable_interval():
    from gpuq_swap_review import review

    row = review(
        dict(
            id="unreadable",
            host_swap_mb=0,
            job_swap_mb=0,
            available_min_mb=100000,
            contended=True,
            contention_reasons=["swap_unreadable"],
        )
    )
    assert row["verdict"] == "UNVERIFIABLE"


def test_memory_budget_boundaries():
    assert memory_verdict(h.DEFAULTS["host_swap_mb"], 64, 4096, h.DEFAULTS) == []
