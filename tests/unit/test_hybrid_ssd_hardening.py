"""HybridSnapshotStore: size cap shared with the block store (R05) and corrupt /
foreign files invalidate and fall back cold (R07)."""

from __future__ import annotations

import os
import threading

import mlx.core as mx
import pytest

from yunshu_engine.hybrid_ssd_snapshot import HybridSnapshotStore


def _cache(n=8, fill=1.0):
    from mlx_lm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.keys = mx.ones((1, 2, n, 4)) * fill
    kv.values = mx.ones((1, 2, n, 4)) * fill
    kv.offset = n
    ac = ArraysCache(2)
    ac.cache = [mx.ones((1, 3, 4)) * fill, mx.ones((1, 2, 2, 2)) * fill]
    return [kv, ac]


def _key(i):
    return i.to_bytes(16, "big")


def _file_size(store, key):
    return store._path(key.hex()).stat().st_size


# ---- R05: size cap -------------------------------------------------------------
def test_max_bytes_is_honored_with_lru(tmp_path):
    probe = HybridSnapshotStore(str(tmp_path / "probe"))
    probe.save(_key(0), _cache(), 8)
    one = _file_size(probe, _key(0))
    store = HybridSnapshotStore(str(tmp_path / "s"), max_bytes=int(one * 3.5))
    for i in range(1, 7):
        store.save(_key(i), _cache(fill=float(i)), 8)
        if i == 3:
            assert store.load(_key(1))[0] is not None  # touch 1: now most recent
    assert store.total_bytes <= int(one * 3.5)
    on_disk = sum(f.stat().st_size for f in (tmp_path / "s").rglob("*.safetensors"))
    assert on_disk == store.total_bytes
    assert store.evictions >= 3
    assert store.has(_key(6))  # the newest survives
    assert not store.has(_key(2))  # least recently used went first


def test_shared_budget_callable_is_followed(tmp_path):
    room = {"bytes": 10**9}
    store = HybridSnapshotStore(str(tmp_path), budget=lambda: room["bytes"])
    store.save(_key(1), _cache(), 8)
    store.save(_key(2), _cache(), 8)
    assert store.has(_key(1)) and store.has(_key(2))
    room["bytes"] = _file_size(store, _key(1)) + 10  # the block store grew
    store.save(_key(3), _cache(), 8)
    assert store.total_bytes <= room["bytes"]
    assert store.has(_key(3))


def test_rescan_after_restart_enforces_the_cap(tmp_path):
    big = HybridSnapshotStore(str(tmp_path))
    for i in range(5):
        big.save(_key(i), _cache(), 8)
    one = _file_size(big, _key(0))
    small = HybridSnapshotStore(str(tmp_path), max_bytes=2 * one + 1)
    assert small.total_bytes <= 2 * one + 1


def test_block_store_and_hybrid_share_one_budget(tmp_path):
    from yunshu_engine.kv_prefix_cache import KVPrefixCache

    cache = KVPrefixCache.__new__(KVPrefixCache)
    cache._lock = threading.RLock()
    KVPrefixCache.enable_ssd_cache(
        cache, cache_dir=str(tmp_path), max_size_bytes=10**6, model_name="m"
    )
    assert cache._hybrid_ssd is not None
    hybrid = cache._hybrid_ssd
    hybrid.save(_key(1), _cache(), 8)
    assert cache._ssd_cache._external_bytes() == hybrid.total_bytes > 0
    assert hybrid._limit() <= 10**6 - cache._ssd_cache.block_bytes()


# ---- R07: corrupt / foreign files ----------------------------------------------
def _saved(tmp_path, **kw):
    store = HybridSnapshotStore(str(tmp_path), **kw)
    store.save(_key(1), _cache(), 8)
    assert store.load(_key(1))[0] is not None
    return store, store._path(_key(1).hex())


def _assert_cold_and_gone(store, path):
    assert store.load(_key(1)) == (None, 0)
    assert not path.exists()
    assert not store.has(_key(1))
    assert store.load(_key(1)) == (None, 0)  # never retried


def test_truncated_file(tmp_path):
    store, path = _saved(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    _assert_cold_and_gone(store, path)
    assert store.invalidated == 1


def test_garbage_and_empty_file(tmp_path):
    store, path = _saved(tmp_path)
    path.write_bytes(os.urandom(256))
    _assert_cold_and_gone(store, path)
    store, path = _saved(tmp_path)
    path.write_bytes(b"")
    _assert_cold_and_gone(store, path)


def test_header_length_lies(tmp_path):
    store, path = _saved(tmp_path)
    data = bytearray(path.read_bytes())
    data[:8] = (2**40).to_bytes(8, "little")
    path.write_bytes(bytes(data))
    _assert_cold_and_gone(store, path)


def _rewrite(path, mutate):
    arrays, meta = mx.load(str(path), return_metadata=True)
    arrays, meta = mutate(dict(arrays), dict(meta))
    mx.save_safetensors(str(path), arrays, meta)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a, m: (a, {**m, "schema": "1"}),  # older layout
        lambda a, m: (a, {**m, "fp": "another-checkpoint"}),
        lambda a, m: (a, {**m, "n": "9"}),  # more layers than tensors
        lambda a, m: (a, {**m, "tok": "0"}),
        lambda a, m: (a, {**m, "l0": "mystery"}),
        lambda a, m: (a, {**m, "l0o": "99"}),  # offset beyond the keys
        lambda a, m: (a, {**m, "l0o": "0"}),
        lambda a, m: (a, {**m, "l1m": "0,5"}),  # state index out of range
        lambda a, m: ({k: v for k, v in a.items() if k != "l0_v"}, m),
        lambda a, m: ({**a, "l0_k": mx.ones((2, 4))}, m),  # wrong rank
        lambda a, m: ({**a, "l0_v": a["l0_v"].astype(mx.float16)}, m),  # dtype split
        lambda a, m: ({**a, "l0_v": mx.ones((1, 2, 7, 4))}, m),  # shape mismatch
        lambda a, m: ({**a, "l1_a0": mx.zeros((0,))}, m),
    ],
)
def test_structurally_invalid_snapshots_go_cold(tmp_path, mutate):
    store, path = _saved(tmp_path, fingerprint="rev-A")
    _rewrite(path, mutate)
    _assert_cold_and_gone(store, path)


def test_int8_scale_record_must_be_sane(tmp_path):
    store, path = _saved(tmp_path, precision="int8")
    _rewrite(path, lambda a, m: (a, {**m, "l0_ks": "nan"}))
    _assert_cold_and_gone(store, path)


def test_foreign_files_are_dropped_on_startup(tmp_path):
    _store, path = _saved(tmp_path, fingerprint="rev-A")
    other = HybridSnapshotStore(str(tmp_path), fingerprint="rev-B")
    assert not other.has(_key(1))
    assert not path.exists()  # another revision's snapshot is deleted, not served
    legacy = tmp_path / "hybrid_snapshots" / "ab"
    legacy.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(legacy / ("ab" * 16 + ".safetensors")), {"x": mx.ones(2)})
    again = HybridSnapshotStore(str(tmp_path), fingerprint="rev-B")
    assert again.candidate_token_counts() == []  # no schema: invalidated


def test_roundtrip_still_exact_and_atomic(tmp_path):
    store, path = _saved(tmp_path, fingerprint="rev-A")
    out, tok = store.load(_key(1))
    assert tok == 8 and out[0].offset == 8
    assert mx.array_equal(out[1].cache[0], mx.ones((1, 3, 4))).item()
    assert not [f for f in path.parent.iterdir() if ".tmp." in f.name]
    assert HybridSnapshotStore(str(tmp_path), fingerprint="rev-A").has(_key(1))
