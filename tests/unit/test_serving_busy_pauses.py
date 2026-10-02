"""GPU busy-time accounting (fake clock) and the gpuq pause-overlap contract."""

from __future__ import annotations

import json
import math

import pytest

from yunshu_engine.serving import busy_time, pauses


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_busy_meter_accumulates_and_idle_fraction():
    c = Clock()
    m = busy_time.BusyMeter(c)
    c.t += 2.0  # idle
    with m.span():
        c.t += 3.0
    c.t += 5.0  # idle
    with m.span():
        c.t += 2.0
    snap = m.snapshot()
    assert snap["busy_seconds"] == 5.0 and snap["uptime_seconds"] == 12.0
    assert snap["idle_fraction"] == pytest.approx(1 - 5 / 12, abs=1e-4)
    assert snap["spans"] == 2


def test_nested_spans_counted_once_and_open_span_included():
    c = Clock()
    m = busy_time.BusyMeter(c)
    with m.span():
        c.t += 1.0
        with m.span():  # a driver step inside a slice
            c.t += 2.0
        c.t += 1.0
    assert m.busy_seconds() == 4.0
    m.enter()
    c.t += 2.5
    assert m.busy_seconds() == 6.5  # still running
    m.exit()
    m.exit()  # unbalanced: ignored, never negative
    assert m.busy_seconds() == 6.5


def test_span_closes_on_exception():
    c = Clock()
    m = busy_time.BusyMeter(c)
    with pytest.raises(RuntimeError), m.span():
        c.t += 1.0
        raise RuntimeError
    c.t += 4.0
    assert m.busy_seconds() == 1.0


def test_idle_fraction_missing_data_is_none():
    assert busy_time.idle_fraction(1.0, 0.0) is None
    assert (
        busy_time.window_idle_fraction(5, 10, 4, 20) is None
    )  # counter went backwards
    assert busy_time.window_idle_fraction(5, 10, 6, 10) is None  # clock did not advance
    assert busy_time.window_idle_fraction(5, 10, 8, 20) == pytest.approx(0.7)


def test_runner_slice_and_driver_step_are_metered():
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    c = Clock()
    runner = VLMBatchRunner(model=None, processor=None)
    runner.busy_meter = busy_time.BusyMeter(c)
    runner.driver_busy_meter = busy_time.BusyMeter(c)

    class Driver:
        def step(self):
            c.t += 0.5
            return []

    runner.driver = Driver()
    runner._driver_jobs = {}
    runner._note_driver_prefill = lambda: None
    runner._driver_jobs[1] = type(
        "J", (), {"abandoned": False, "cancel_event": None, "stats": None}
    )()
    runner._step_driver()
    assert runner.driver_busy_meter.busy_seconds() == 0.5
    # a whole slice with nothing pending costs (fake) zero but is counted as a span
    runner._driver_jobs.clear()
    runner._drive_slice(resubmit=False)
    snap = runner.busy_snapshot()
    assert snap["slices"]["spans"] == 1 and snap["round_driver"]["busy_seconds"] == 0.5


# ── pause contract ──────────────────────────────────────────────────────


def test_overlaps_semantics():
    p = [(10.0, 20.0), (50.0, 60.0)]
    assert pauses.overlaps(0, 5, p) is False
    assert pauses.overlaps(5, 10, p) is True  # touching counts (closed intervals)
    assert pauses.overlaps(12, 15, p) is True  # inside
    assert pauses.overlaps(5, 70, p) is True  # contains
    assert pauses.overlaps(20, 30, p) is True
    assert pauses.overlaps(20.001, 49.999, p) is False
    assert pauses.overlaps(61, 62, p) is False
    assert pauses.overlaps(0, 100, []) is False
    assert (
        pauses.overlaps(55, 1e9, [(50.0, None)]) is True
    )  # open pause extends forever
    with pytest.raises(ValueError):
        pauses.overlaps(5, 1, p)
    with pytest.raises(ValueError):
        pauses.overlaps(math.nan, 1, p)


def test_parse_open_interval_ends_now():
    got = pauses.parse_pauses({"pauses": [[1, 2], [5, None]]}, now=9.0)
    assert got == [(1.0, 2.0), (5.0, 9.0)]
    assert pauses.parse_pauses({}, now=1.0) == []
    assert pauses.parse_pauses([[1, 2]]) == [(1.0, 2.0)]


@pytest.mark.parametrize(
    "bad",
    [
        {"pauses": "x"},
        {"pauses": [[1]]},
        {"pauses": [["a", 2]]},
        {"pauses": [[5, 1]]},
        {"pauses": [[1, math.inf]]},
        {"pauses": [[1, 2, 3]]},
        {"pauses": 5},
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(pauses.PauseDataError):
        pauses.parse_pauses(bad, now=0.0)


def test_pause_file_reading_and_fail_closed(tmp_path, monkeypatch):
    f = tmp_path / "j.pauses.json"
    f.write_text(json.dumps({"pauses": [[10, 20], [30, None]]}))
    assert pauses.read_pause_file(str(f), now=40.0) == [(10.0, 20.0), (30.0, 40.0)]
    assert pauses.was_paused(15, 16, str(f), now=40.0) is True
    assert pauses.was_paused(21, 29, str(f), now=40.0) is False
    # unreadable / malformed file: paused (fail closed)
    assert pauses.was_paused(1, 2, str(tmp_path / "missing.json")) is True
    f.write_text("{not json")
    assert pauses.was_paused(1, 2, str(f)) is True
    # no file configured: never paused
    monkeypatch.delenv(pauses.ENV_VAR, raising=False)
    assert pauses.was_paused(1, 2) is False
    monkeypatch.setenv(pauses.ENV_VAR, str(f))
    assert pauses.was_paused(1, 2) is True


def test_paused_seconds_merges_overlaps():
    assert pauses.paused_seconds(0, 100, [(10, 20), (15, 30), (50, 60)]) == 30.0
    assert pauses.paused_seconds(12, 18, [(10, 20)]) == 6.0
    assert pauses.paused_seconds(0, 5, [(10, 20)]) == 0.0


def test_clean_samples_reports_dropped():
    samples = [(0, 5, "a"), (8, 12, "b"), (21, 25, "c"), (19, 22, "d")]
    kept, dropped = pauses.clean_samples(samples, [(10.0, 20.0)])
    assert kept == ["a", "c"] and dropped == 2
