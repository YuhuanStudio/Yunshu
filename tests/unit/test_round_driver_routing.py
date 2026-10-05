"""Round driver routing: a lone text request keeps the speculative lane."""

from types import SimpleNamespace

import pytest

from yunshu_engine.vlm_batch_runner import VLMBatchRunner


@pytest.mark.parametrize(
    ("alone", "pkw", "has_driver", "uncached", "expect"),
    [
        (True, None, True, 100, False),  # lone text request: lane
        (False, None, True, 100, True),  # concurrent text: driver
        (False, None, True, 4096, True),  # at the limit
        (False, None, True, 4097, False),  # long cold prompt: upstream prefill
        (False, {"pixel_values": 1}, True, 100, False),  # image prompt: upstream
        (False, None, False, 100, False),  # no driver built
    ],
)
def test_driver_takes(alone, pkw, has_driver, uncached, expect):
    r = object.__new__(VLMBatchRunner)
    r.driver = object() if has_driver else None
    r._work = lambda job: SimpleNamespace(uncached_tokens=uncached)
    assert r._driver_takes(SimpleNamespace(prompt_kwargs=pkw), alone) is expect
