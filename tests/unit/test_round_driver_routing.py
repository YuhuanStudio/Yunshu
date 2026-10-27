"""Round driver routing: the driver is the path for every text request."""

from types import SimpleNamespace

import pytest

from yunshu_engine.vlm_batch_runner import VLMBatchRunner


@pytest.mark.parametrize(
    ("alone", "pkw", "has_driver", "expect"),
    [
        (True, None, True, True),  # a lone request: the driver (no lane)
        (False, None, True, True),  # concurrent text: the driver
        (False, {"pixel_values": 1}, True, False),  # media prompt: stock runner
        (False, None, False, False),  # no driver built (other family, int8 KV, MoE)
    ],
)
def test_driver_takes(alone, pkw, has_driver, expect):
    r = object.__new__(VLMBatchRunner)
    r.driver = SimpleNamespace(can_draft=True) if has_driver else None
    assert r._driver_takes(SimpleNamespace(prompt_kwargs=pkw), alone) is expect


@pytest.mark.parametrize("can_draft", [True, False])
def test_processor_rows_decode_in_the_driver_too(can_draft):
    r = object.__new__(VLMBatchRunner)
    r.driver = SimpleNamespace(can_draft=can_draft)
    job = SimpleNamespace(prompt_kwargs=None, processors=[object()], guide=object())
    assert r._driver_takes(job, True) is True
