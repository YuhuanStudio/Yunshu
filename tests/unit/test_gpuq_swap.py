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
    monkeypatch.setattr(h, "_process_gone", lambda pid: False)  # still alive
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
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


def _monitor(tmp_path, monkeypatch):
    job = dict(id="exit", pid=10, env={})
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
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        m.sampler,
        "sample",
        lambda *a, **kw: dict(time=kw["now"], since=kw["now"] - 2, foreign_cpu_pct=0),
    )
    return job, m


def _row(pid):
    return dict(pid=pid, start=pid, pageins=0, swapped_bytes=0, regions=1)


def test_child_exiting_during_read_is_not_unreadable(tmp_path, monkeypatch):
    # 1019 deploy27b-heap-27b-1019h: a short-lived child exited (zombie) between the ps
    # snapshot and footprint; policy 2 latched job_swap_unreadable with 0.0 MB swapped.
    job, m = _monitor(tmp_path, monkeypatch)
    m.sampler.owned = {10, 11}

    def read(pid):
        if pid == 11:
            raise ValueError("incomplete footprint for 11")
        return _row(pid)

    monkeypatch.setattr(h, "process_memory", read)
    monkeypatch.setattr(h, "_process_gone", lambda pid: pid == 11)
    m.record("start", 100)
    m.record("poll", 140)
    assert job["job_swap_mb"] == 0 and not job.get("contention_reasons")


def test_transient_read_failure_retries(tmp_path, monkeypatch):
    job, m = _monitor(tmp_path, monkeypatch)
    m.sampler.owned = {10}
    calls = []

    def flaky(pid):
        calls.append(pid)
        if len(calls) == 1:
            raise h.subprocess.TimeoutExpired("footprint", 5)
        return _row(pid)

    monkeypatch.setattr(h, "process_memory", flaky)
    monkeypatch.setattr(h, "_process_gone", lambda pid: False)
    m.record("start", 100)
    assert len(calls) == 2 and job["job_swap_mb"] == 0
    assert not job.get("contention_reasons")


def test_zombie_counts_as_gone(monkeypatch):
    monkeypatch.setattr(h.os, "kill", lambda *a: None)
    monkeypatch.setattr(
        h.subprocess,
        "run",
        lambda *a, **kw: h.subprocess.CompletedProcess(a, 0, stdout="Z+\n"),
    )
    assert h._process_gone(4242)


def test_job_swap_is_concurrent_peak_not_sum_of_process_peaks(tmp_path, monkeypatch):
    job = dict(id="restarts", pid=10, env={})
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
    monkeypatch.setattr(
        m.sampler,
        "sample",
        lambda *a, **kw: dict(time=kw["now"], since=kw["now"] - 2, foreign_cpu_pct=0),
    )
    # Three server restarts, each peaking at 50 MB while alone: 50 MB, not 150.
    for i, pid in enumerate((11, 12, 13)):
        m.sampler.owned = {10, pid}
        monkeypatch.setattr(
            h,
            "process_memory",
            lambda p, pid=pid: dict(
                pid=p,
                start=pid,
                pageins=0,
                swapped_bytes=50_000_000 if p == pid else 0,
                regions=1,
            ),
        )
        m.record("start" if i == 0 else "poll", 100 + 2 * i)
    assert job["job_swap_mb"] == 50
