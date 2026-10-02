"""R27: a model handed out by get_engine() must not be unloaded before its
request registers as active (and an explicit lease holds it until release)."""

from __future__ import annotations

import asyncio

import pytest

from yunshu_engine.model_manager import (
    ModelEntry,
    ModelManager,
    ModelType,
    _lease_scope_var,
    lease_scope,
)


class _IdleEngine:
    stopped = False

    def has_active_requests(self):
        return False

    async def stop(self):
        self.stopped = True


def _mgr():
    mgr = ModelManager()
    eng = _IdleEngine()
    mgr._entries["m"] = ModelEntry(
        model_id="m",
        model_path="/x",
        model_type=ModelType.LLM,
        engine=eng,
        is_loaded=True,
        estimated_bytes=0,
    )
    return mgr, eng


@pytest.mark.asyncio
async def test_unload_refused_between_get_engine_and_request_end():
    mgr, eng = _mgr()
    with lease_scope() as scope:
        got = await mgr.get_engine("m")  # request now preprocessing / streaming
        assert got is eng
        async with mgr._lock:
            assert await mgr._unload_model_locked("m") is False
        assert mgr._find_lru_victim() is None
    assert scope.closed and mgr._entries["m"].leases == 0
    async with mgr._lock:
        assert await mgr._unload_model_locked("m") is True


@pytest.mark.asyncio
async def test_scope_released_on_error_and_cancel():
    mgr, _ = _mgr()
    with pytest.raises(RuntimeError):
        with lease_scope():
            await mgr.get_engine("m")
            raise RuntimeError("boom")
    assert mgr._entries["m"].leases == 0

    async def work():
        with lease_scope():
            await mgr.get_engine("m")
            await asyncio.sleep(60)

    t = asyncio.create_task(work())
    await asyncio.sleep(0.05)
    assert mgr._entries["m"].leases == 1
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert mgr._entries["m"].leases == 0


@pytest.mark.asyncio
async def test_late_task_after_scope_closed_does_not_leak():
    mgr, _ = _mgr()
    with lease_scope() as scope:
        pass
    token = _lease_scope_var.set(scope)  # a task that outlived its request
    try:
        await mgr.get_engine("m")
    finally:
        _lease_scope_var.reset(token)
    assert mgr._entries["m"].leases == 0


@pytest.mark.asyncio
async def test_multi_model_scope_holds_each_and_ttl_skips():
    mgr, eng = _mgr()
    eng2 = _IdleEngine()
    mgr._entries["n"] = ModelEntry(
        model_id="n",
        model_path="/y",
        model_type=ModelType.LLM,
        engine=eng2,
        is_loaded=True,
        estimated_bytes=0,
    )
    with lease_scope():
        await mgr.get_engine("m")
        async with mgr._lock:
            assert await mgr._unload_model_locked("m") is False
            assert await mgr._unload_model_locked("n") is True  # not leased
    assert eng2.stopped and not eng.stopped
    assert mgr._entries["m"].leases == 0


@pytest.mark.asyncio
async def test_concurrent_scopes_are_independent():
    mgr, _ = _mgr()
    gate = asyncio.Event()

    async def req():
        with lease_scope():
            await mgr.get_engine("m")
            await gate.wait()

    ts = [asyncio.create_task(req()) for _ in range(3)]
    await asyncio.sleep(0.05)
    assert mgr._entries["m"].leases == 3
    gate.set()
    await asyncio.gather(*ts)
    assert mgr._entries["m"].leases == 0


@pytest.mark.asyncio
async def test_middleware_releases_after_streaming_and_on_exception():
    from yunshu_gateway.middleware.model_lease import ModelLeaseMiddleware

    mgr, _ = _mgr()
    seen = []

    async def app(scope, receive, send):
        await mgr.get_engine("m")
        seen.append(mgr._entries["m"].leases)
        if scope.get("boom"):
            raise ValueError("x")

    mw = ModelLeaseMiddleware(app)
    for typ in ("http", "websocket"):
        await mw({"type": typ}, None, None)
        assert mgr._entries["m"].leases == 0
    with pytest.raises(ValueError):
        await mw({"type": "http", "boom": True}, None, None)
    assert mgr._entries["m"].leases == 0 and seen == [1, 1, 1]


@pytest.mark.asyncio
async def test_lease_blocks_unload_until_release():
    mgr, eng = _mgr()
    async with mgr.lease("m") as e:
        assert e is eng
        async with mgr._lock:
            assert await mgr._unload_model_locked("m") is False
        assert mgr._find_lru_victim() is None
    async with mgr._lock:
        assert await mgr._unload_model_locked("m") is True


@pytest.mark.asyncio
async def test_lease_released_on_error():
    mgr, _ = _mgr()
    with pytest.raises(ValueError):
        async with mgr.lease("m"):
            raise ValueError("boom")
    assert mgr._entries["m"].leases == 0
