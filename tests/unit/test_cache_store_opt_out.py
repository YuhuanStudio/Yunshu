"""X-Yunshu-Cache-Store: off (omlx c4bb311e): the request reuses a cached prefix but stores nothing."""

import threading
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import KVCache  # noqa: E402

from tests.unit.test_apc_manager import _coordinator, _mgr, _prompt  # noqa: E402
from yunshu_engine.kv_prefix_cache import KVPrefixCache, store_opt_out  # noqa: E402
from yunshu_engine.request_tracker import (  # noqa: E402
    RequestTracker,
    current_request_info,
)
from yunshu_gateway.x_yunshu import RequestInfo, cache_store_off  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


@pytest.mark.parametrize("raw", [b"off", b"OFF", b" Off ", "off"])
def test_header_off_values(raw):
    assert cache_store_off(raw) is True


@pytest.mark.parametrize("raw", [None, b"", b"on", b"false", b"0", b"no"])
def test_other_values_keep_storing(raw):
    assert cache_store_off(raw) is False


def test_tracker_puts_the_opt_out_on_the_cancel_event():
    info = RequestInfo("r1", "POST", "/v1/chat/completions")
    info.cache_store = False
    token = current_request_info.set(info)
    try:
        gen = RequestTracker().register("e1")
    finally:
        current_request_info.reset(token)
    assert gen.cancel_event.cache_store is False
    # default request: the attribute is absent and the engines read True
    gen2 = RequestTracker().register("e2")
    assert getattr(gen2.cancel_event, "cache_store", True) is True


def _kv(n):
    c = KVCache()
    c.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    return [c]


def test_text_prefix_cache_view_looks_up_but_does_not_store():
    cache = KVPrefixCache(min_prefix_length=4)
    stored = mx.array(list(range(40)))
    cache.add(stored, _kv(40))
    off = threading.Event()
    off.cache_store = False  # type: ignore[attr-defined]
    view = store_opt_out(cache, off)
    assert view is not cache
    # lookup is the cache's own: the stored prefix is reused
    hit, matched, rest = view.get(mx.array(list(range(40)) + [99, 98]))
    assert hit is not None and matched > 0 and rest >= 0
    # the opted-out request finishes: nothing is added
    before = cache.size
    view.add(mx.array(list(range(100, 140))), _kv(40))
    assert cache.size == before
    # a default request still stores
    assert store_opt_out(cache, threading.Event()) is cache
    store_opt_out(cache, threading.Event()).add(
        mx.array(list(range(100, 140))), _kv(40)
    )
    assert cache.size == before + 1
    assert store_opt_out(None, off) is None


def test_apc_store_is_refused_but_plan_and_lookup_are_unchanged():
    m = _mgr()
    c = _coordinator(m)
    c.defer_checkpoint_stores = False
    ids = _prompt(list(range(100, 3000)), range(5000, 8000))
    plan_normal = c.checkpoint_lengths(ids, set(), begin=False)

    c.set_store_off(ids)
    # the prefill split points (numerics) are exactly an ordinary request's
    assert c.checkpoint_lengths(ids, set(), begin=False) == plan_normal
    assert (
        c.request(ids) is None
    )  # no policy installed: lookup takes the canonical path
    c.checkpoint_lengths(ids, set())  # begin: sets _current_ids like the runner does
    assert c.store_checkpoint(list(ids[: plan_normal[0]]), _kv(8)) is False
    assert not getattr(m, "_exact_cache", None)

    c.release_store_off(ids)
    assert tuple(ids) not in c.__dict__.get("_store_off", {})


def test_identical_prompts_release_independently():
    c = _coordinator(_mgr())
    c.set_store_off([1, 2, 3])
    c.set_store_off([1, 2, 3])
    c.release_store_off([1, 2, 3])
    assert (1, 2, 3) in c.__dict__["_store_off"]
    c.release_store_off([1, 2, 3])
    assert (1, 2, 3) not in c.__dict__["_store_off"]
    c.release_store_off([1, 2, 3])  # idempotent


def test_runner_job_reads_the_event_flag():
    from yunshu_engine import vlm_batch_runner as vbr

    src = Path(vbr.__file__).read_text()
    assert 'cache_store=bool(getattr(cancel_event, "cache_store", True))' in src
