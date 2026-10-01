"""One disk budget per SSD-cache root: a global cap across namespaces, stale-namespace
pruning, a free-space reserve, and write errors (ENOSPC) that drop the checkpoint, warn once
and pause instead of crashing or leaving temp files. APC tier and text tier."""

from __future__ import annotations

import errno
import logging
import os
import time
from collections import namedtuple
from pathlib import Path

import mlx.core as mx
import pytest

from yunshu_kv import cache_gc, disk_budget
from yunshu_kv.disk_budget import GIB, DiskBudget

BLOCK = 16
Usage = namedtuple("Usage", "total used free")


def _budget(root, cap=0, **kw):
    kw.setdefault("reserve_pct", 0.0)
    kw.setdefault("reserve_min_bytes", 0)
    kw.setdefault("recheck_s", 0.0)
    return DiskBudget(root, cap_bytes=cap, **kw)


def _free(monkeypatch, free_gib, total_gib=1000):
    monkeypatch.setattr(
        disk_budget,
        "disk_usage",
        lambda p: Usage(total_gib * GIB, (total_gib - free_gib) * GIB, free_gib * GIB),
    )


def _cache(n=32):
    from mlx_vlm.models.cache import KVCache

    kv = KVCache()
    kv.keys = mx.ones((1, 2, n, 4), dtype=mx.bfloat16)
    kv.values = mx.ones((1, 2, n, 4), dtype=mx.bfloat16) * 2
    kv.offset = n
    return [kv]


def _apc_store(root, ns, budget):
    from yunshu_engine.apc_manager import SpillDiskStore

    store = SpillDiskStore(root, namespace=ns, num_workers=1)
    store.attach_budget(budget)
    return store


def _apc_put(store, seed):
    from mlx_vlm.apc import _sequence_hash

    tokens = tuple(range(seed * 100, seed * 100 + 32))
    key = _sequence_hash(tokens, 0, BLOCK)
    ok = store.write_now(key, tokens, 0, _cache(), True)
    store.flush()
    return key, ok


def _files(root):
    return [p for p in Path(root).rglob("*.safetensors")]


def _leftovers(root):
    return [
        p for p in Path(root).rglob("*") if p.is_file() and disk_budget.is_tmp(p.name)
    ]


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


# ── global cap across namespaces ───────────────────────────────────────────────
def test_apc_global_cap_evicts_lru_across_namespaces(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    probe = _budget(tmp_path)
    s0 = _apc_store(tmp_path, "probe", probe)
    k, _ = _apc_put(s0, 9)
    one = s0._exact_index[k].stat().st_size
    s0.close()
    for p in (tmp_path / "probe").iterdir():
        p.unlink()
    (tmp_path / "probe").rmdir()

    cap = int(one * 4.5)
    budget = _budget(tmp_path, cap=cap)
    stores = [_apc_store(tmp_path, f"ns{i}", budget) for i in range(3)]
    keys = []
    for round_ in range(3):
        for i, st in enumerate(stores):
            keys.append(_apc_put(st, round_ * 3 + i + 1)[0])
    total = sum(p.stat().st_size for p in _files(tmp_path))
    assert total <= cap, f"{total} > global cap {cap}"
    assert len(_files(tmp_path)) == 4
    # the survivors are the most recent writes, not "the first namespace's"
    present = {p.stem for p in _files(tmp_path)}
    newest = {stores[0]._exact_id_for(k) for k in keys[-4:]}
    assert present == newest
    for st in stores:
        st.close()


def test_apc_cap_is_min_of_cap_and_free_space(tmp_path, monkeypatch):
    _free(monkeypatch, 30)  # 30 GiB free, 10 GiB reserve
    b = _budget(tmp_path, cap=64 * GIB, reserve_min_bytes=10 * GIB)
    assert b.effective_cap() == 20 * GIB  # used 0 + free 30 - reserve 10
    _free(monkeypatch, 12)  # space shrinks later: re-checked, not fixed at startup
    assert b.effective_cap() == 2 * GIB


# ── stale namespaces ───────────────────────────────────────────────────────────
def test_stale_namespace_is_pruned_fresh_and_own_are_kept(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    budget = _budget(tmp_path, stale_days=7)
    old = _apc_store(tmp_path, "old", budget)
    _apc_put(old, 1)
    old.close()
    new = _apc_store(tmp_path, "new", budget)
    _apc_put(new, 2)
    for p in (tmp_path / "old").rglob("*"):
        if p.is_file():
            _age(p, 10)
    _age(tmp_path / "old", 10)
    removed = budget.prune_stale()
    assert [r[0] for r in removed] == ["old"]
    assert not (tmp_path / "old").exists()
    assert len(_files(tmp_path / "new")) == 1
    new.close()


def test_namespace_whose_checkpoint_is_gone_is_pruned(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    ns = tmp_path / "dead"
    ns.mkdir()
    (
        ns / "exact_" + "0" * 0
        if False
        else ns / ("exact_" + "a" * 32 + ".safetensors")
    ).write_bytes(b"x" * 10)
    disk_budget.write_marker(ns, str(tmp_path / "no-such-model"), "abc")
    live = tmp_path / "model"
    live.mkdir()
    (live / "config.json").write_text("{}")
    from yunshu_kv.fingerprint import checkpoint_fingerprint

    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / ("exact_" + "b" * 32 + ".safetensors")).write_bytes(b"x" * 10)
    disk_budget.write_marker(
        keep, str(live), checkpoint_fingerprint(live, digest_size=8)
    )
    assert disk_budget.namespace_orphaned(ns) == "checkpoint gone"
    assert disk_budget.namespace_orphaned(keep) is None
    (live / "config.json").write_text('{"x": 1}')  # revised in place
    assert disk_budget.namespace_orphaned(keep) == "checkpoint changed"
    removed = _budget(tmp_path, stale_days=0).prune_stale()
    assert {r[0] for r in removed} == {"dead", "keep"}


def test_gc_reports_and_removes_stale_namespaces(tmp_path):
    ns = tmp_path / "oldns"
    ns.mkdir()
    f = ns / ("shard_" + "c" * 32 + ".safetensors")
    # a valid empty safetensors file: header "{}"
    f.write_bytes((2).to_bytes(8, "little") + b"{}")
    _age(f, 30)
    _age(ns, 30)
    rep = cache_gc.scan(tmp_path, stale_days=7)
    assert rep.stale_namespaces == ["oldns"]
    assert rep.by_reason() == {"stale-namespace": 1}
    assert f.exists()  # dry run
    cache_gc.scan(tmp_path, stale_days=7, apply=True)
    assert not ns.exists()


# ── free-space reserve ─────────────────────────────────────────────────────────
def test_reserve_blocks_writes_then_resumes(tmp_path, monkeypatch):
    budget = _budget(tmp_path, reserve_pct=10, reserve_min_bytes=20 * GIB)
    store = _apc_store(tmp_path, "ns", budget)
    _free(monkeypatch, 15)  # < max(10% of 1000 GiB = 100, 20)
    key, ok = _apc_put(store, 1)
    assert not ok and not _files(tmp_path) and not _leftovers(tmp_path)
    assert budget.dropped_writes == 1
    _free(monkeypatch, 500)
    key, ok = _apc_put(store, 1)
    assert ok and len(_files(tmp_path)) == 1
    store.close()


def test_reserve_is_max_of_percent_and_floor(tmp_path, monkeypatch):
    _free(monkeypatch, 500, total_gib=1000)
    assert disk_budget.reserve_bytes(tmp_path, 10, 20 * GIB) == 100 * GIB
    _free(monkeypatch, 50, total_gib=100)
    assert disk_budget.reserve_bytes(tmp_path, 10, 20 * GIB) == 20 * GIB


# ── write errors ───────────────────────────────────────────────────────────────
def _enospc(monkeypatch):
    real = mx.save_safetensors
    state = {"fail": True}

    def save(path, *a, **k):
        if state["fail"]:
            Path(path).write_bytes(b"partial")  # what a full disk leaves behind
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, *a, **k)

    monkeypatch.setattr(mx, "save_safetensors", save)
    return state


def test_apc_enospc_drops_checkpoint_warns_once_cleans_and_resumes(
    tmp_path, monkeypatch, caplog
):
    _free(monkeypatch, 900)
    budget = _budget(tmp_path)
    store = _apc_store(tmp_path, "ns", budget)
    state = _enospc(monkeypatch)
    with caplog.at_level(logging.WARNING):
        for i in range(1, 4):
            _, ok = _apc_put(store, i)
            assert ok is False
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert all(r.exc_info is None for r in warnings)
    assert "No space left" in warnings[0].getMessage()
    assert not _files(tmp_path) and not _leftovers(tmp_path)
    assert budget.paused
    state["fail"] = False  # space is back
    key, ok = _apc_put(store, 4)
    assert ok and not budget.paused
    assert store.load_exact_cache(key, prefix_len=32) is not None
    store.close()


def test_paused_budget_skips_writes_until_recheck(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    budget = _budget(tmp_path, recheck_s=3600)
    budget.record_failure(OSError(errno.ENOSPC, "No space left on device"))
    assert budget.allow_write(1) is False
    budget._paused_at -= 7200
    assert budget.allow_write(1) is True


def test_apc_other_write_error_does_not_raise(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    budget = _budget(tmp_path)
    store = _apc_store(tmp_path, "ns", budget)

    def boom(*a, **k):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(mx, "save_safetensors", boom)
    _, ok = _apc_put(store, 1)
    assert ok is False and not _leftovers(tmp_path)
    store.close()


# ── text SSD cache ─────────────────────────────────────────────────────────────
def _text(root, ns, budget):
    from yunshu_engine.ssd_kv_cache import SSDKVCache

    c = SSDKVCache(cache_dir=str(root / ns), max_size_bytes=1 << 40, budget=budget)
    c._budget_ns = ns
    budget.register_owner(ns, c.evict_path)
    return c


def _block(i):
    return bytes([i]) * 16, [mx.zeros((1, 4, 64)), mx.ones((1, 4, 64))]


def test_text_cache_global_cap_across_namespaces(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    budget = _budget(tmp_path)
    caches = [_text(tmp_path, f"m{i}", budget) for i in range(2)]
    h, d = _block(1)
    caches[0].save_block(h, d, token_count=64)
    one = _files(tmp_path)[0].stat().st_size
    budget.cap_bytes = int(one * 3.5)
    for i in range(2, 9):
        h, d = _block(i)
        caches[i % 2].save_block(h, d, token_count=64)
    total = sum(p.stat().st_size for p in _files(tmp_path))
    assert total <= budget.cap_bytes
    assert len(_files(tmp_path)) == 3
    for c in caches:
        c.close()


def test_text_cache_stale_namespace_pruned(tmp_path, monkeypatch):
    _free(monkeypatch, 900)
    budget = _budget(tmp_path, stale_days=7)
    c = _text(tmp_path, "oldmodel", budget)
    h, d = _block(1)
    c.save_block(h, d, token_count=64)
    c.close()
    for p in (tmp_path / "oldmodel").rglob("*"):
        if p.is_file():
            _age(p, 9)
    _age(tmp_path / "oldmodel", 9)
    assert [r[0] for r in budget.prune_stale()] == ["oldmodel"]
    assert not (tmp_path / "oldmodel").exists()


def test_text_cache_reserve_blocks_write(tmp_path, monkeypatch):
    budget = _budget(tmp_path, reserve_pct=10, reserve_min_bytes=20 * GIB)
    c = _text(tmp_path, "m", budget)
    _free(monkeypatch, 5)
    h, d = _block(1)
    c.save_block(h, d, token_count=64)
    assert not _files(tmp_path) and not c.has_block(h) or c.load_block(h) is not None
    assert not _files(tmp_path)
    _free(monkeypatch, 800)
    h2, d2 = _block(2)
    c.save_block(h2, d2, token_count=64)
    assert len(_files(tmp_path)) == 1
    c.close()


def test_text_cache_enospc_no_traceback_no_tmp_and_resumes(
    tmp_path, monkeypatch, caplog
):
    from yunshu_engine import ssd_kv_cache

    _free(monkeypatch, 900)
    budget = _budget(tmp_path)
    c = _text(tmp_path, "m", budget)
    real = ssd_kv_cache._write_safetensors
    state = {"fail": True}

    def write(path, tensors, metadata=None):
        if state["fail"]:
            # a full disk: the temp file is created, then the write raises
            Path(f"{path}.dead.tmp").write_bytes(b"x")
            try:
                raise OSError(errno.ENOSPC, "No space left on device")
            finally:
                os.unlink(f"{path}.dead.tmp")
        return real(path, tensors, metadata)

    monkeypatch.setattr(ssd_kv_cache, "_write_safetensors", write)
    with caplog.at_level(logging.WARNING):
        for i in range(1, 4):
            h, d = _block(i)
            c.save_block(h, d, token_count=64)  # must not raise
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is None
    assert not _files(tmp_path) and not _leftovers(tmp_path)
    state["fail"] = False
    h, d = _block(9)
    c.save_block(h, d, token_count=64)
    assert len(_files(tmp_path)) == 1 and not budget.paused
    c.close()


def test_hybrid_store_enospc_and_reserve(tmp_path, monkeypatch, caplog):
    from yunshu_engine.hybrid_ssd_snapshot import HybridSnapshotStore

    _free(monkeypatch, 900)
    budget = _budget(tmp_path)
    store = HybridSnapshotStore(str(tmp_path / "m"), disk_budget=budget)
    store.budget_ns = "m"
    budget.register_owner("m", store.evict_path)
    from mlx_lm.models.cache import KVCache

    def kv():
        c = KVCache()
        c.keys = mx.ones((1, 2, 8, 4))
        c.values = mx.ones((1, 2, 8, 4))
        c.offset = 8
        return [c]

    real = mx.save_safetensors

    def save(path, *a, **k):
        Path(path).write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(mx, "save_safetensors", save)
    with caplog.at_level(logging.WARNING):
        store.save(b"k" * 16, kv(), token_count=8)
        store.save(b"j" * 16, kv(), token_count=8)
    assert not _files(tmp_path) and not _leftovers(tmp_path)
    assert not store.has(b"k" * 16) or not _files(tmp_path)
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    monkeypatch.setattr(mx, "save_safetensors", real)
    store.save(b"i" * 16, kv(), token_count=8)
    assert len(_files(tmp_path)) == 1
    _free(monkeypatch, 0)  # nothing left: reserve blocks the next one
    store.save(b"h" * 16, kv(), token_count=8)
    assert len(_files(tmp_path)) == 1


@pytest.mark.parametrize("n", [0])
def test_budget_status_lists_namespaces(tmp_path, n):
    for ns in ("nsa", "nsb"):
        (tmp_path / ns).mkdir()
        (tmp_path / ns / ("exact_" + "d" * 32 + ".safetensors")).write_bytes(b"x" * 100)
    st = _budget(tmp_path).status()
    assert st.namespaces == {"nsa": 100, "nsb": 100} and st.used == 200
