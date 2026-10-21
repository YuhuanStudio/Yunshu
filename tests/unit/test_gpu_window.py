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
