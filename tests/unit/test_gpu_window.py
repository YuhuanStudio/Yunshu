import json

from yunshu_engine import gpu_window


def test_tick_is_inert_without_the_interposer(monkeypatch):
    monkeypatch.delenv("DYLD_INSERT_LIBRARIES", raising=False)
    gpu_window._STATE.update({"lib": False, "n": 0})
    for _ in range(100):
        gpu_window.tick()
    assert gpu_window._STATE["lib"] is None and gpu_window._STATE["n"] == 0


def test_gaps_split_across_segments_named_by_the_opening_mark():
    marks = [
        (0.0, "tick"),
        (1.0, "readback"),
        (2.0, "commit_sent"),
        (4.0, "verify_sent"),
    ]
    gaps = [(-1.0, 0.5), (1.5, 2.5), (3.0, 3.25), (5.0, 6.0)]
    out = gpu_window.charge_gaps(gaps, marks)
    assert out == {
        "tick": 500.0,  # clipped at the first mark
        "readback": 500.0,
        "commit_sent": 750.0,
        "verify_sent": 1000.0,  # after the last mark
    }


def test_marks_are_ignored_outside_the_window():
    gpu_window._STATE["marks"] = None
    gpu_window.mark("readback")
    assert gpu_window._STATE["marks"] is None


def test_segment_times_sum_each_marks_span():
    marks = [(0.0, "a"), (0.001, "b"), (0.003, "a"), (0.004, "b")]
    out = gpu_window.segment_times(marks)
    assert round(out["a"], 6) == 2.0 and round(out["b"], 6) == 2.0


class _FakeInterposer:
    """Stands in for libmetalrt_trace: counts begins/ends and records the round each begin came at."""

    def __init__(self):
        self.begin_rounds = []
        self.ends = 0

        def metalrt_gaps(big, size, thr):
            big.value = b"[]"

        self.metalrt_gaps = metalrt_gaps

    def metalrt_begin(self):
        self.begin_rounds.append(gpu_window._STATE["n"])

    def metalrt_end(self, buf, size):
        self.ends += 1
        buf.value = json.dumps(
            {
                "busy_union_ms": 40.0 * (gpu_window.END - gpu_window.START),
                "dispatches": 100 * (gpu_window.END - gpu_window.START),
                "commit_cpu_ms": 1.0 * (gpu_window.END - gpu_window.START),
            }
        ).encode()


def _fresh_window_state(monkeypatch, lib):
    for key, value in {
        "lib": lib,
        "n": 0,
        "marks": None,
        "start": 0,
        "last": 0.0,
        "done": False,
        "t0": 0.0,
    }.items():
        monkeypatch.setitem(gpu_window._STATE, key, value)


def test_window_with_a_prefill_inside_is_dropped_and_steady_window_reported(
    monkeypatch, caplog
):
    import logging

    caplog.set_level(logging.INFO, logger="yunshu_engine.gpu_window")
    lib = _FakeInterposer()
    _fresh_window_state(monkeypatch, lib)
    for _ in range(
        gpu_window.START + 10
    ):  # the window opens at round START, 10 rounds in
        gpu_window.tick(capture_kernels=False)
    gpu_window.boundary("prefill")  # a 32K prefill starts inside the open window
    assert lib.ends == 1 and gpu_window._STATE["marks"] is None
    for _ in range(4 * gpu_window.ROUNDS):
        gpu_window.tick(capture_kernels=False)
    # the next window opens only after the prefill, and it reports once
    assert lib.begin_rounds[0] == gpu_window.START
    assert lib.begin_rounds[1] > gpu_window.START + 10
    messages = [r.getMessage() for r in caplog.records]
    assert sum("dropped (boundary prefill)" in m for m in messages) == 1
    assert sum(m.startswith("GPU window (") for m in messages) == 1
    assert sum(m.startswith("Host time by segment") for m in messages) == 1


def test_boundary_outside_a_window_is_a_no_op(monkeypatch):
    lib = _FakeInterposer()
    _fresh_window_state(monkeypatch, lib)
    gpu_window.boundary("prefill")
    assert lib.ends == 0 and lib.begin_rounds == []


def test_host_gap_inside_a_window_is_not_charged_to_decode(monkeypatch, caplog):
    """Base interfaces only (tick/mark): a 1.5 s host stall after mark("dflash.return") must not show
    up as decode time in the printed per-segment table."""
    import logging
    import time

    caplog.set_level(logging.INFO, logger="yunshu_engine.gpu_window")
    lib = _FakeInterposer()
    _fresh_window_state(monkeypatch, lib)
    for i in range(gpu_window.START + 2 * (gpu_window.END - gpu_window.START) + 5):
        gpu_window.tick(capture_kernels=False)
        gpu_window.mark("dflash.return")
        if i == gpu_window.START + 10:
            time.sleep(1.5)  # stands in for a prefill inside the open window
    tables = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("Host time")
    ]
    for line in tables:
        per_round = eval(line.split(": ", 1)[1])  # printed dict of ms/round
        assert per_round.get("dflash.return", 0.0) < 5.0, line
    assert tables, "the steady window after the stall must still report"


class _FakeClock:
    """Replaces gpu_window's time module: the test advances the clock between rounds."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def perf_counter(self):
        return self.t


def _run_rounds(monkeypatch, caplog, round_s, stall_at=None, stall_s=0.0, rounds=None):
    import logging

    caplog.set_level(logging.INFO, logger="yunshu_engine.gpu_window")
    clock = _FakeClock()
    monkeypatch.setattr(gpu_window, "time", clock)
    lib = _FakeInterposer()
    _fresh_window_state(monkeypatch, lib)
    total = rounds or gpu_window.START + 2 * gpu_window.ROUNDS + 5
    for i in range(total):
        clock.t += stall_s if i == stall_at else round_s
        gpu_window.tick(capture_kernels=False)
        gpu_window.mark("dflash.return")
    return [r.getMessage() for r in caplog.records]


def test_slow_steady_rounds_are_reported_not_dropped(monkeypatch, caplog):
    """A 27B-class round of 400 ms is steady decode: every round gap is 400 ms, so no stall."""
    messages = _run_rounds(monkeypatch, caplog, round_s=0.4)
    assert not [m for m in messages if "dropped" in m]
    assert sum(m.startswith("GPU window (40 rounds)") for m in messages) == 1


def test_stall_far_above_the_median_round_drops_the_window_with_reason(
    monkeypatch, caplog
):
    messages = _run_rounds(
        monkeypatch, caplog, round_s=0.045, stall_at=gpu_window.START + 10, stall_s=1.5
    )
    dropped = [m for m in messages if "GPU window dropped" in m]
    assert len(dropped) == 1 and "host gap of 1500 ms" in dropped[0]
    assert sum(m.startswith("GPU window (40 rounds)") for m in messages) == 1


def test_boundary_drop_prints_its_reason(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="yunshu_engine.gpu_window")
    lib = _FakeInterposer()
    _fresh_window_state(monkeypatch, lib)
    for _ in range(gpu_window.START + 3):
        gpu_window.tick(capture_kernels=False)
    gpu_window.boundary("prefill")
    assert any(
        "GPU window dropped (boundary prefill)" in r.getMessage()
        for r in caplog.records
    )
