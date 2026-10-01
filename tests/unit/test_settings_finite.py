import pytest

from yunshu_engine import settings


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "Infinity"])
def test_float_rejects_non_finite(monkeypatch, bad):
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", bad)
    with pytest.raises(settings.SettingError, match="YUNSHU_DRAIN_TIMEOUT"):
        settings.get("YUNSHU_DRAIN_TIMEOUT")


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_gb_rejects_non_finite(monkeypatch, bad):
    monkeypatch.setenv("YUNSHU_MAX_MEMORY_GB", bad)
    with pytest.raises(settings.SettingError, match="YUNSHU_MAX_MEMORY_GB"):
        settings.get("YUNSHU_MAX_MEMORY_GB")


def test_valid_and_disabled(monkeypatch):
    monkeypatch.setenv("YUNSHU_DRAIN_TIMEOUT", "5")
    assert settings.get("YUNSHU_DRAIN_TIMEOUT") == 5.0
    monkeypatch.setenv("YUNSHU_MAX_MEMORY_GB", "disabled")
    assert settings.get("YUNSHU_MAX_MEMORY_GB") == 0.0
