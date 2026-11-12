"""After the idle store, a finished reply stays standing for its follow-up, which appends to it in place
(no restore copy: 40 ms of a 33K follow-up's TTFT, 2026-11-12), at the memory of before:

- an exact cut-out (the slot batch's): the APC keeps it as it is (owned), and the one copy the store made
  before becomes the standing copy, with STANDING_ROOM spare keys;
- caches with room already (a DFlash row decoded in place): copied into the APC as before, and they stand.

A reply the pool would not keep, an unstored reply, an unclaimed standing cache and the idle trim all
return the buffers as before. The cut-out itself never carries room (+7 GiB at 96K, fu-mem-attrib)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine import vlm_batch_runner as vbr  # noqa: E402
from yunshu_engine.vlm_batch_runner import VLMBatchRunner, _Job  # noqa: E402


def _kv(n, spare=0):
    from mlx_vlm.models.cache import KVCache

    c = KVCache()
    c.keys = mx.arange(2 * (n + spare) * 4, dtype=mx.float32).reshape(
        1, 2, n + spare, 4
    )
    c.values = c.keys + 1
    c.offset = n
    return c


def _gdn():
    from mlx_vlm.models.cache import ArraysCache

    c = ArraysCache(2)
    c.cache = [mx.ones((1, 3, 8)), mx.ones((1, 2, 4, 4))]
    return c


class _Coord:
    def __init__(self):
        self.manager = SimpleNamespace()
        self.standing = []
        self.stores = []

    def offer_standing(self, tokens, rows, *, extra_hash=0):
        self.standing.append((list(tokens), rows, extra_hash))
        self.manager._standing = (tuple(tokens), rows, extra_hash)

    def take_standing(self):
        return self.manager.__dict__.pop("_standing", None)


def _job(prompt, generated, salt=7):
    j = _Job.__new__(_Job)
    j.ids, j.generated, j.apc_salt = list(prompt), list(generated), salt
    return j


def _runner():
    import threading

    r = VLMBatchRunner.__new__(VLMBatchRunner)
    r._lock, r._pending, r._executor = threading.Lock(), [], None
    return r


def _setup(monkeypatch, *, accepts=True, stored=True):
    from yunshu_engine.kernels import buffer_cache

    offered = []
    monkeypatch.setattr(buffer_cache, "offer_restore_buffers", offered.append)
    monkeypatch.setattr(buffer_cache, "pool_accepts", lambda n: accepts)
    coord = _Coord()

    def store(g, j, caches=None, window_tail=None, salt=None, owned=False):
        coord.stores.append(owned)
        return stored

    monkeypatch.setattr(VLMBatchRunner, "_store_decode_cache", staticmethod(store))
    group = SimpleNamespace(gen=SimpleNamespace(apc=coord), jobs={}, driver_salt=5)
    return offered, coord, group


def _flush(group, caches, generated=(10, 11, 12)):
    r = _runner()
    r._decode_pending = (group, _job([1, 2, 3, 4], list(generated)), caches, None)
    r._flush_decode_cache()
    return r


def test_an_exact_cut_out_is_stored_as_is_and_a_copy_with_room_stands(monkeypatch):
    offered, coord, group = _setup(monkeypatch)
    caches = [_kv(6), _gdn()]
    _flush(group, caches)
    assert coord.stores == [True]  # owned: the APC keeps the cut-out, no second copy
    [(tokens, rows, salt)] = coord.standing
    assert tokens == [1, 2, 3, 4, 10, 11] and salt == 5
    kv, gdn = rows
    assert kv is not caches[0] and kv.offset == 6
    assert kv.keys.shape[-2] == 6 + vbr.STANDING_ROOM
    assert mx.array_equal(kv.keys[..., :6, :], caches[0].keys).item()
    assert not mx.any(kv.keys[..., 6:, :]).item()
    # the follow-up's prefill replaces its cache's recurrent arrays: never the stored ones
    assert gdn is not caches[1] and gdn.cache is not caches[1].cache
    assert (
        offered == []
    )  # the stored buffers are the APC's: never pooled for overwriting


def test_caches_with_room_are_copied_into_the_apc_and_stand(monkeypatch):
    offered, coord, group = _setup(monkeypatch)
    caches = [_kv(6, spare=vbr.STANDING_ROOM), _gdn()]
    _flush(group, caches)
    assert coord.stores == [False]
    kv = coord.standing[0][1][0]
    # the reply's own buffers stand (their room kept), not a copy
    assert kv.offset == 6 and kv.keys.shape[-2] == 6 + vbr.STANDING_ROOM
    assert offered == []


def test_a_reply_the_pool_would_not_keep_is_not_held(monkeypatch):
    offered, coord, group = _setup(monkeypatch, accepts=False)
    exact = [_kv(6)]
    _flush(group, exact)
    assert coord.standing == [] and offered == []  # stored as is: the APC's
    roomy = [_kv(6, spare=vbr.STANDING_ROOM)]
    _flush(group, roomy)
    assert coord.standing == [] and offered == [roomy]  # copied: pooled as before


def test_an_unstored_reply_is_not_held(monkeypatch):
    offered, coord, group = _setup(monkeypatch, stored=False)
    caches = [_kv(6)]
    _flush(group, caches)
    assert coord.standing == [] and offered == [caches]


def test_an_unclaimed_standing_cache_goes_to_the_pool_at_the_next_reply(monkeypatch):
    from yunshu_engine import dflash_apc_window

    monkeypatch.setattr(dflash_apc_window, "take_decode_tail", lambda: None)
    offered, coord, group = _setup(monkeypatch)
    r = _flush(group, [_kv(6)])
    rows = coord.standing[0][1]
    r._defer_decode_cache(group, _job([9], [8]), caches=["next"])
    assert offered == [rows]  # the earlier reply's standing copy, pooled as before


class _Owner:
    def __init__(self):
        self._standing = ("tokens", ["rows"], 0)


def test_the_idle_trim_releases_a_standing_cache():
    from yunshu_engine.kernels import buffer_cache

    owner = _Owner()
    buffer_cache.hold_standing(owner, "_standing")
    buffer_cache.clear_restore_buffers()
    assert "_standing" not in owner.__dict__
