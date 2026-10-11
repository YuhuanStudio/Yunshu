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
