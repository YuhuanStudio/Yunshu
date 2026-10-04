"""Round driver routing: a lone text request keeps the speculative lane."""

from types import SimpleNamespace

import pytest

from yunshu_engine.vlm_batch_runner import VLMBatchRunner


@pytest.mark.parametrize(
    ("alone", "pkw", "has_driver", "expect"),
    [
        (True, None, True, False),  # lone text request: lane
        (False, None, True, True),  # concurrent text: driver
        (False, {"pixel_values": 1}, True, False),  # image prompt: upstream
        (False, None, False, False),  # no driver built
    ],
)
def test_driver_takes(alone, pkw, has_driver, expect):
    r = object.__new__(VLMBatchRunner)
    r.driver = object() if has_driver else None
    assert r._driver_takes(SimpleNamespace(prompt_kwargs=pkw), alone) is expect
