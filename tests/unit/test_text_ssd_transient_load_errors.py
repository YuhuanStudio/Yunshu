"""A transient load failure (out of memory, interrupted or exhausted system call, MLX allocation failure)
keeps the SSD checkpoint file and the lookup goes cold; only corruption deletes (port of the APC lane's
ae33517 to the text path's SSDKVCache and HybridSnapshotStore)."""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from yunshu_engine.hybrid_ssd_snapshot import HybridSnapshotStore
from yunshu_engine.ssd_kv_cache import SSDKVCache
from yunshu_engine.transient_errors import is_transient_load_error

TRANSIENT = [
    MemoryError(),
    OSError(12, "Cannot allocate memory"),
    OSError(4, "Interrupted system call"),
    RuntimeError("[metal::malloc] Unable to allocate 4096 bytes"),
]


def test_classifier():
    for e in TRANSIENT:
        assert is_transient_load_error(e)
    for e in (ValueError("bad header"), OSError(2, "No such file"), KeyError("x")):
        assert not is_transient_load_error(e)


@pytest.mark.parametrize("exc", TRANSIENT)
def test_ssd_block_survives_a_transient_load_error(tmp_path, monkeypatch, exc):
    cache = SSDKVCache(cache_dir=str(tmp_path), max_size_bytes=10 * 1024**2)
    key = b"\x07" * 16
    cache.save_block(key, [mx.ones((1, 4, 8))], token_count=16)
    cache._process_pending_writes()
    path = cache._block_path(key)
    cache._hot_cache.clear()

    def boom(*a, **k):
        raise exc

    with monkeypatch.context() as mp:
        mp.setattr(mx, "load", boom)
        assert cache.load_block(key) is None  # goes cold this time
    assert os.path.exists(path) and cache.has_block(key)
    assert cache.load_block(key) is not None  # and serves once the pressure is gone
    cache.close()


def _kv(n=8):
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    c.keys = mx.ones((1, 2, n, 4))
    c.values = mx.ones((1, 2, n, 4))
    c.offset = n
    return [c]


@pytest.mark.parametrize("exc", TRANSIENT)
def test_hybrid_snapshot_survives_a_transient_load_error(tmp_path, monkeypatch, exc):
    store = HybridSnapshotStore(str(tmp_path))
    key = b"deadbeefcafef00d"
    store.save(key, _kv(), token_count=8)
    path = store._path(key.hex())

    def boom(*a, **k):
        raise exc

    with monkeypatch.context() as mp:
        mp.setattr(mx, "load", boom)
        assert store.load(key) == (None, 0)
    assert path.exists() and store.has(key)
    cache_list, tok = store.load(key)
    assert cache_list is not None and tok == 8


def test_hybrid_snapshot_corruption_still_deletes(tmp_path):
    store = HybridSnapshotStore(str(tmp_path))
    key = b"deadbeefcafef00d"
    store.save(key, _kv(), token_count=8)
    path = store._path(key.hex())
    path.write_bytes(b"not a safetensors file")
    assert store.load(key) == (None, 0)
    assert not path.exists()
