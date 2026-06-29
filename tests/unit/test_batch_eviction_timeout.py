"""batch_inference hardening.

H1 (HIGH): _cleanup_batch_store (run at every create_batch) evicted entries by TTL/size
WITHOUT checking status — a long-running batch could have its own store entry evicted by
a concurrently-created batch, then its unguarded _batch_store[id][...] writes raised
KeyError (lost items / 500 after generation already succeeded). Fix: never evict an
in_progress batch; guard the writes.

M1 (MED): asyncio.wait() RETURNS (done, pending) on timeout (never raises TimeoutError),
so the timeout->"expired" status was unreachable — a window-expired batch reported
completed/failed, never "expired". Fix: derive expired from the pending count.

M2 (MED): the batch /v1/completions executor passed token-id prompts straight to the
engine -> str([785,...]) garbage. Fix: decode token-id forms via _normalize_prompts.
"""

from __future__ import annotations

import inspect
import time

from yunshu_gateway.routers import (
    batch_inference as B,  # noqa: N812  # intentional short module alias
)


def test_inprogress_batch_not_evicted_by_ttl():
    B._batch_store.clear()
    old = time.time() - (B._BATCH_STORE_TTL + 100)
    B._batch_store["running"] = {
        "id": "running",
        "status": "in_progress",
        "started_at": old,
    }
    B._batch_store["done"] = {"id": "done", "status": "completed", "started_at": old}
    B._cleanup_batch_store()
    assert "running" in B._batch_store, "in-progress batch must survive TTL eviction"
    assert "done" not in B._batch_store, "finished+expired batch should be evicted"
    B._batch_store.clear()


def test_size_eviction_skips_inprogress():
    B._batch_store.clear()
    orig = B._BATCH_STORE_MAX_SIZE
    try:
        B._BATCH_STORE_MAX_SIZE = 1
        now = time.time()
        # two in-progress + one finished; size cap is 1 → only the finished one is evictable
        B._batch_store["r1"] = {
            "id": "r1",
            "status": "in_progress",
            "started_at": now - 5,
        }
        B._batch_store["r2"] = {
            "id": "r2",
            "status": "in_progress",
            "started_at": now - 4,
        }
        B._batch_store["d1"] = {
            "id": "d1",
            "status": "completed",
            "started_at": now - 3,
        }
        B._cleanup_batch_store()
        assert "r1" in B._batch_store and "r2" in B._batch_store
        assert "d1" not in B._batch_store  # the only evictable one went first
    finally:
        B._BATCH_STORE_MAX_SIZE = orig
        B._batch_store.clear()


def test_expired_status_derived_from_timed_out():
    src = inspect.getsource(B.create_batch)
    # the timed_out>0 -> expired override is present, after the (dead) mapping
    assert "if timed_out > 0:" in src
    assert '_oai_status = "expired"' in src


def test_progress_write_is_guarded():
    src = inspect.getsource(B.create_batch)
    assert (
        "_batch_store.get(batch_id)" in src
    )  # both progress + final writes use .get()


def test_completion_executor_decodes_token_ids():
    src = inspect.getsource(B._execute_completion)
    assert "_normalize_prompts" in src
    assert "all(isinstance(p, str) for p in prompt)" in src
