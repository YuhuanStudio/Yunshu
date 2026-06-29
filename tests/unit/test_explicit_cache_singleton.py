"""(LOW): get_store() lazy-init raced on cold start.

The unguarded `if _STORE is None: _STORE = ExplicitContextCache()` let two concurrent
first requests each build their own store; the last assignment won, so a create() the
loser store already serviced was silently lost. Fixed with double-checked locking.
"""

from __future__ import annotations

import inspect
import threading

from yunshu_gateway import explicit_cache


def test_concurrent_get_store_returns_one_singleton():
    explicit_cache._STORE = None  # simulate cold start
    barrier = threading.Barrier(16)
    results = []
    lock = threading.Lock()

    def _worker():
        barrier.wait()  # maximize the race window
        s = explicit_cache.get_store()
        with lock:
            results.append(s)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 16
    # every concurrent caller observed the SAME instance — no lost loser-store.
    assert all(s is results[0] for s in results)
    assert all(s is explicit_cache.get_store() for s in results)


def test_get_store_uses_double_checked_locking():
    src = inspect.getsource(explicit_cache.get_store)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_STORE_LOCK" in code
    # the None-check appears twice (outside + inside the lock)
    assert code.count("_STORE is None") == 2
