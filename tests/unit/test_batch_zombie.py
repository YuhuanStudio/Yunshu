"""batch jobs can't get stuck at in_progress → unbounded store-growth leak.

The cancel handler set the entry's status AFTER awaiting asyncio.gather() — but the gather is
itself a cancellation point, so a re-delivered CancelledError (normal on client disconnect)
propagated out before the status write, leaving the entry stuck at "in_progress" FOREVER.
_cleanup_batch_store refuses to ever evict an in_progress entry, and once 1000 zombies
accumulate it gives up entirely (`if not _evictable: break`) → the store grows unbounded.

Fix: (1) _mark_batch_terminal() sets the terminal status BEFORE the gather (so a cancellation
during gather can't lose it); (2) _cleanup_batch_store evicts in_progress ZOMBIES older than
2*TTL as a belt-and-suspenders.
"""
from __future__ import annotations

import time

import yunshu_gateway.routers.batch_inference as bi


def _reset():
    bi._batch_store.clear()


def test_mark_terminal_forces_in_progress_and_preserves_terminal():
    _reset()
    bi._batch_store["b"] = {"id": "b", "status": "in_progress", "started_at": time.time()}
    bi._mark_batch_terminal("b", "cancelled")
    assert bi._batch_store["b"]["status"] == "cancelled"
    assert "finished_at" in bi._batch_store["b"]
    # an already-terminal entry is NOT clobbered
    bi._batch_store["c"] = {"id": "c", "status": "completed", "started_at": time.time()}
    bi._mark_batch_terminal("c", "error")
    assert bi._batch_store["c"]["status"] == "completed"
    # a missing entry is a no-op (no KeyError)
    bi._mark_batch_terminal("nope", "error")
    _reset()


def test_cleanup_evicts_in_progress_zombies_but_keeps_live():
    _reset()
    now = time.time()
    bi._batch_store["zombie"] = {"id": "zombie", "status": "in_progress",
                                 "started_at": now - bi._BATCH_STORE_TTL * 2 - 100}
    bi._batch_store["live"] = {"id": "live", "status": "in_progress", "started_at": now}
    bi._cleanup_batch_store()
    assert "zombie" not in bi._batch_store   # old in_progress zombie evicted
    assert "live" in bi._batch_store          # fresh in_progress kept
    _reset()


def test_cancel_handler_marks_before_gather_in_source():
    import inspect
    src = inspect.getsource(bi.create_batch)
    # the cancel handler calls _mark_batch_terminal BEFORE awaiting the gather
    ci = src.index("except asyncio.CancelledError:")
    mark = src.index("_mark_batch_terminal(batch_id, \"cancelled\")", ci)
    gather = src.index("await asyncio.gather", ci)
    assert mark < gather, "status must be set BEFORE the gather (the cancellation point)"
