"""YUNSHU_GENERIC_LANE and YUNSHU_QWEN4_PREFILL_STEP: defaults keep the shipped arithmetic, overrides flip them."""

import pytest

from yunshu_engine import settings


@pytest.fixture(autouse=True)
def _clean():
    settings.clear_overrides()
    yield
    settings.clear_overrides()


def test_generic_lane_is_off_by_default_and_flips():
    assert settings.get_bool("YUNSHU_GENERIC_LANE") is False
    settings.set_override("YUNSHU_GENERIC_LANE", True)
    assert settings.get_bool("YUNSHU_GENERIC_LANE") is True


def test_qwen4_prefill_step_default_and_override():
    assert int(settings.get("YUNSHU_QWEN4_PREFILL_STEP")) == 2048
    settings.set_override("YUNSHU_QWEN4_PREFILL_STEP", 8192)
    assert int(settings.get("YUNSHU_QWEN4_PREFILL_STEP")) == 8192
