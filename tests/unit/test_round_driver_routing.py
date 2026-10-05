"""Round driver routing: a lone text request keeps the speculative lane."""

from types import SimpleNamespace

import pytest

from yunshu_engine.vlm_batch_runner import VLMBatchRunner


@pytest.mark.parametrize(
    ("alone", "pkw", "has_driver", "uncached", "expect"),
    [
        (True, None, True, 100, False),  # lone text request: lane
        (False, None, True, 100, True),  # concurrent text: driver
        (False, None, True, 12288, True),  # at the limit
        (False, None, True, 12289, False),  # long cold prompt: upstream prefill
        (False, {"pixel_values": 1}, True, 100, False),  # image prompt: upstream
        (False, None, False, 100, False),  # no driver built
    ],
)
def test_driver_takes(alone, pkw, has_driver, uncached, expect):
    r = object.__new__(VLMBatchRunner)
    r.driver = object() if has_driver else None
    r._driver_uncached = lambda job: uncached
    assert r._driver_takes(SimpleNamespace(prompt_kwargs=pkw), alone) is expect


def test_driver_uncached_counts_only_the_drivers_own_cache_entries():
    """A prefix cached by the upstream path is cold for the driver (its APC keys
    differ): routing a warm 32K session there prefilled 34K tokens (TTFT 64 s vs
    5 s)."""
    import threading

    ids = list(range(40000))
    drv_salt, up_salt = 111, 222
    entries = {
        "up": SimpleNamespace(token_ids=tuple(ids[:32768]), extra_hash=up_salt),
        "drv": SimpleNamespace(token_ids=tuple(ids[:8192]), extra_hash=drv_salt),
    }
    r = object.__new__(VLMBatchRunner)
    r.driver = SimpleNamespace(
        apc=object(),
        will_draft=lambda **kw: True,
        apc_salt=lambda extra, drafting: drv_salt,
    )
    r.apc_manager = SimpleNamespace(lock=threading.Lock(), _exact_cache=entries)
    r.apc_semantic_hash = 0
    job = SimpleNamespace(
        ids=ids,
        priority=0,
        sampling=None,
        processors=[],
        logprobs=False,
        allow_draft=True,
    )
    assert r._driver_uncached(job) == 40000 - 8192
    del entries["drv"]
    assert r._driver_uncached(job) == 40000  # the upstream entry is not the driver's
