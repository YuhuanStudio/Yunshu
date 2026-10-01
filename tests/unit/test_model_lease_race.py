"""R27: a model handed out by get_engine() must not be unloaded before its
request registers as active (and an explicit lease holds it until release)."""

from __future__ import annotations

import pytest

from yunshu_engine.model_manager import ModelEntry, ModelManager, ModelType


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
async def test_unload_refused_between_get_engine_and_request_start():
    mgr, eng = _mgr()
    got = await mgr.get_engine("m")  # caller now awaiting preprocessing
    assert got is eng
    async with mgr._lock:
        assert await mgr._unload_model_locked("m") is False
    assert eng.stopped is False


@pytest.mark.asyncio
async def test_lease_blocks_unload_until_release():
    mgr, eng = _mgr()
    mgr._entries["m"].handoff_until = 0.0
    async with mgr.lease("m") as e:
        assert e is eng
        async with mgr._lock:
            assert await mgr._unload_model_locked("m") is False
        assert mgr._find_lru_victim() is None
    mgr._entries["m"].handoff_until = 0.0
    async with mgr._lock:
        assert await mgr._unload_model_locked("m") is True


@pytest.mark.asyncio
async def test_lease_released_on_error():
    mgr, _ = _mgr()
    with pytest.raises(ValueError):
        async with mgr.lease("m"):
            raise ValueError("boom")
    assert mgr._entries["m"].leases == 0
