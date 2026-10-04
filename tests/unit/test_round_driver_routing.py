"""Round driver routing: a lone text request keeps the speculative lane."""

from types import SimpleNamespace

import pytest

from yunshu_engine import settings
from yunshu_engine.vlm_batch_runner import VLMBatchRunner


def _runner(driver):
    r = object.__new__(VLMBatchRunner)
    r.driver = driver
    return r


@pytest.mark.parametrize(
    ("min_conc", "alone", "pkw", "has_driver", "expect"),
    [
        (2, True, None, True, False),  # lone text request: lane
        (2, False, None, True, True),  # concurrent text: driver
        (1, True, None, True, True),  # legacy: always driver
        (2, False, {"pixel_values": 1}, True, False),  # image prompt: upstream
        (2, False, None, False, False),  # no driver built
    ],
)
def test_driver_takes(monkeypatch, min_conc, alone, pkw, has_driver, expect):
    monkeypatch.setenv("YUNSHU_ROUND_DRIVER_MIN_CONCURRENCY", str(min_conc))
    settings.reload() if hasattr(settings, "reload") else None
    r = _runner(object() if has_driver else None)
    assert r._driver_takes(SimpleNamespace(prompt_kwargs=pkw), alone) is expect
