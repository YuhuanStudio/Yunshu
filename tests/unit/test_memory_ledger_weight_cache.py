"""The weight cache must not hand one engine's bytes to another object."""

from __future__ import annotations

import gc
import weakref

from yunshu_gateway import memory_ledger


class _Arr:
    def __init__(self, n):
        self.nbytes = n


class _Model:
    def __init__(self, n):
        self._n = n

    def parameters(self):
        return {"w": _Arr(self._n)}


class _Engine:
    def __init__(self, n):
        self._model = _Model(n)


def test_reused_address_does_not_alias_cached_weights():
    memory_ledger._weight_cache.clear()
    small, big = _Engine(10), _Engine(1000)

    # Simulate the address reuse: a dead engine's entry sits under big's id.
    class _Gone:
        pass

    gone = _Gone()
    dead = weakref.ref(gone)
    del gone
    gc.collect()
    memory_ledger._weight_cache[id(big)] = (dead, 10)
    assert memory_ledger.measured_weight_bytes(big) == 1000
    assert memory_ledger.measured_weight_bytes(small) == 10


def test_cache_hit_and_cleanup_on_engine_death():
    memory_ledger._weight_cache.clear()
    eng = _Engine(64)
    assert memory_ledger.measured_weight_bytes(eng) == 64
    assert id(eng) in memory_ledger._weight_cache
    key = id(eng)
    del eng
    gc.collect()
    assert key not in memory_ledger._weight_cache
