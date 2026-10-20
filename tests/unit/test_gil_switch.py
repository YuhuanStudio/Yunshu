import sys

from yunshu_engine import gil_switch, settings


def test_apply_sets_and_zero_keeps(monkeypatch):
    old = sys.getswitchinterval()
    try:
        settings.set_override("YUNSHU_GIL_SWITCH_MS", "0.5")
        assert abs(gil_switch.apply() - 0.0005) < 1e-9
        sys.setswitchinterval(0.005)
        settings.set_override("YUNSHU_GIL_SWITCH_MS", "0")
        assert abs(gil_switch.apply() - 0.005) < 1e-9
    finally:
        settings.clear_overrides()
        sys.setswitchinterval(old)


def test_zero_after_nonzero_restores_the_original_interval():
    old = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.005)
        gil_switch._original = None
        settings.set_override("YUNSHU_GIL_SWITCH_MS", "0.5")
        assert abs(gil_switch.apply() - 0.0005) < 1e-9
        settings.set_override("YUNSHU_GIL_SWITCH_MS", "0")
        assert abs(gil_switch.apply() - 0.005) < 1e-9  # not left at 0.5 ms
    finally:
        settings.clear_overrides()
        gil_switch._original = None
        sys.setswitchinterval(old)


def test_batched_engine_start_applies_the_setting():
    import inspect

    from yunshu_engine.batched_engine import BatchedEngine

    assert "gil_switch.apply()" in inspect.getsource(BatchedEngine.start)
