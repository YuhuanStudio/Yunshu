"""_unload_model_locked must NOT tear down a model with active
requests (the single choke point that protects against the enforcer's TOCTOU
lock-drop and check_ttl's stale last_access). force=True (shutdown) overrides.
"""

from __future__ import annotations

import pytest

from yunshu_engine.model_manager import ModelEntry, ModelManager, ModelType


class _BusyEngine:
    def __init__(self, busy):
        self._busy = busy
        self.stopped = False

    def has_active_requests(self):
        return self._busy

    async def stop(self):
        self.stopped = True


def _mgr_with_entry(busy):
    mgr = ModelManager()
    eng = _BusyEngine(busy)
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
async def test_unload_skips_when_active_requests():
    mgr, eng = _mgr_with_entry(busy=True)
    async with mgr._lock:
        unloaded = await mgr._unload_model_locked("m")
    assert unloaded is False
    assert mgr._entries["m"].is_loaded is True  # not torn down
    assert eng.stopped is False


@pytest.mark.asyncio
async def test_unload_force_overrides_active_requests():
    mgr, eng = _mgr_with_entry(busy=True)
    async with mgr._lock:
        unloaded = await mgr._unload_model_locked("m", force=True)
    assert unloaded is True
    assert mgr._entries["m"].is_loaded is False
    assert eng.stopped is True


class _NoActivityEngine:
    """Mirrors TTS/ASR/Image/STS/Video/OCR engines, which don't define
    has_active_requests()."""

    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_unload_fail_safe_when_activity_unknown():
    """an engine WITHOUT has_active_requests must NOT be torn down by a
    non-forced unload — the old try/except swallowed the AttributeError and
    proceeded, crashing in-flight image/audio generations."""
    mgr = ModelManager()
    eng = _NoActivityEngine()
    mgr._entries["m"] = ModelEntry(
        model_id="m",
        model_path="/x",
        model_type=ModelType.IMAGE_GEN,
        engine=eng,
        is_loaded=True,
        estimated_bytes=0,
    )
    async with mgr._lock:
        unloaded = await mgr._unload_model_locked("m")
    assert unloaded is False
    assert mgr._entries["m"].is_loaded is True
    assert eng.stopped is False
    # force still tears it down (admin / shutdown)
    async with mgr._lock:
        forced = await mgr._unload_model_locked("m", force=True)
    assert forced is True and eng.stopped is True


# ── the fail-safe created a livelock — _find_lru_victim nominated
# non-LLM engines (no has_active_requests) that _unload_model_locked then REFUSED,
# and the eviction loops spun forever holding _lock. ──


def _loaded_entry(mgr, mid, engine, mtype, nbytes, last_access):
    mgr._entries[mid] = ModelEntry(
        model_id=mid,
        model_path="/x",
        model_type=mtype,
        engine=engine,
        is_loaded=True,
        estimated_bytes=nbytes,
    )
    mgr._entries[mid].last_access = last_access


def test_find_lru_victim_excludes_unprovable_engines():
    """A non-LLM engine (no has_active_requests) must NOT be nominated as an LRU
    victim — the unloader will refuse it, so nominating it would livelock."""
    mgr = ModelManager()
    # Only a non-LLM (image) model is loaded, idle, non-pinned.
    _loaded_entry(
        mgr, "img", _NoActivityEngine(), ModelType.IMAGE_GEN, 1000, last_access=1.0
    )
    assert mgr._find_lru_victim() is None  # not nominated (can't prove idle)

    # An idle LLM IS a valid victim and should be picked over the un-provable image.
    _loaded_entry(
        mgr, "llm", _BusyEngine(busy=False), ModelType.LLM, 1000, last_access=2.0
    )
    v = mgr._find_lru_victim()
    assert v is not None and v.model_id == "llm"


@pytest.mark.asyncio
async def test_ensure_memory_no_livelock_with_only_nonllm_loaded():
    """_ensure_memory_available must not spin forever when the only loaded model is
    a non-evictable non-LLM engine — it must raise MemoryError, not hang."""
    mgr = ModelManager()
    mgr.max_memory_bytes = 1000
    _loaded_entry(
        mgr, "img", _NoActivityEngine(), ModelType.IMAGE_GEN, 900, last_access=1.0
    )
    mgr._current_memory_bytes = 900
    async with mgr._lock:
        with pytest.raises(MemoryError):
            # Need 500 more but only an un-evictable image model exists → must raise,
            # not livelock. (Bounded by pytest's own runtime; a hang fails the suite.)
            await mgr._ensure_memory_available(500)
    # The image model was NOT torn down.
    assert mgr._entries["img"].is_loaded is True


@pytest.mark.asyncio
async def test_evict_lru_returns_none_when_only_nonllm():
    mgr = ModelManager()
    _loaded_entry(
        mgr, "img", _NoActivityEngine(), ModelType.IMAGE_GEN, 100, last_access=1.0
    )
    assert await mgr._evict_lru_model() is None
    assert mgr._entries["img"].is_loaded is True
