"""YUNSHU_QWEN4_PREFILL_STEP: default keeps the shipped arithmetic, overrides flip it."""

import pytest

from yunshu_engine import settings


@pytest.fixture(autouse=True)
def _clean():
    settings.clear_overrides()
    yield
    settings.clear_overrides()


def test_qwen4_prefill_step_default_and_override():
    assert int(settings.get("YUNSHU_QWEN4_PREFILL_STEP")) == 2048
    settings.set_override("YUNSHU_QWEN4_PREFILL_STEP", 8192)
    assert int(settings.get("YUNSHU_QWEN4_PREFILL_STEP")) == 8192
