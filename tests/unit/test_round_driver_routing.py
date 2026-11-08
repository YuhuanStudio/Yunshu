"""Round driver routing: the driver is the one path of every request of every model, media prompts included."""

from types import SimpleNamespace

import pytest

from yunshu_engine import vlm_batch_runner as vbr
from yunshu_engine.vlm_batch_runner import VLMBatchRunner


def _runner(driver):
    r = object.__new__(VLMBatchRunner)
    r.driver = driver
    r.apc_manager = None
    r._decode_follows = lambda job: False
    r._admitted = []
    r._admit_driver = lambda job, use_apc: r._admitted.append(job)
    return r


def _job(pkw=None, guide=None, processors=()):
    return SimpleNamespace(
        ids=[1, 2],
        guide=guide,
        priority=0,
        cache_plan=None,
        prompt_kwargs=pkw,
        processors=list(processors),
        logprobs=False,
        use_draft=True,
        allow_draft=True,
        stats=vbr.RunStats(),
    )


@pytest.mark.parametrize("alone", [True, False])
@pytest.mark.parametrize("pkw", [None, {"inputs_embeds": 1}, {"pixel_values": 1}])
def test_every_request_is_admitted_to_the_driver(alone, pkw):
    r = _runner(SimpleNamespace(can_draft=True))
    job = _job(pkw)
    r._admit(job, alone)
    assert r._admitted == [job]


@pytest.mark.parametrize("can_draft", [True, False])
def test_processor_rows_decode_in_the_driver_too(can_draft):
    r = _runner(SimpleNamespace(can_draft=can_draft))
    job = _job(processors=[object()], guide=object())
    r._admit(job, True)
    assert r._admitted == [job]


def test_a_runner_without_a_driver_refuses_the_request():
    r = _runner(None)
    with pytest.raises(RuntimeError, match="no round driver"):
        r._admit(_job(), True)
    assert r._admitted == []
