"""L2 sleep must keep the ModelManager consistent.

The old L2 teardown called entry.engine.stop() directly on every manager entry, leaving
each entry is_loaded=True pointing at a stopped engine and the manager's
_current_memory_bytes never decremented — accounting drift that later mis-triggers/blocks
eviction, and in multi-model mode get_engine() would route a request to the stale (gutted)
entry. Route through manager.unload_model(model_id, force=True) so is_loaded and the memory
accounting stay consistent.
"""

from __future__ import annotations

import inspect


def test_l2_sleep_routes_through_manager_unload():
    from yunshu_gateway.routers import sleep

    src = inspect.getsource(sleep)
    # the L2 block uses the manager's accounting-aware unload, force=True
    assert "manager.unload_model(entry.model_id, force=True)" in src
    # and no longer tears engines down directly behind the manager's back (check code
    # lines only — a comment may still quote the old form for explanation)
    code = [ln.split("#", 1)[0] for ln in src.splitlines()]
    assert not any("entry.engine.stop()" in ln for ln in code)


def test_unload_model_force_decrements_accounting_path_exists():
    # sanity: unload_model accepts force and the locked impl decrements _current_memory_bytes
    from yunshu_engine.model_manager import ModelManager

    sig = inspect.signature(ModelManager.unload_model)
    assert "force" in sig.parameters
    locked = inspect.getsource(ModelManager._unload_model_locked)
    assert "_current_memory_bytes" in locked
    assert "if not force" in locked  # force bypasses the active-request guard
