from yunshu_engine import gpu_window


def test_tick_is_inert_without_the_interposer(monkeypatch):
    monkeypatch.delenv("DYLD_INSERT_LIBRARIES", raising=False)
    gpu_window._STATE.update({"lib": False, "n": 0})
    for _ in range(100):
        gpu_window.tick()
    assert gpu_window._STATE["lib"] is None and gpu_window._STATE["n"] == 0
