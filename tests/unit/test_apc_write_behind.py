"""APC write-behind: a stored checkpoint reaches the SSD without close() (crash-durable)."""

import threading

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from yunshu_engine.apc_manager import SpillDiskStore, YunshuAPCManager  # noqa: E402


def _cache(n: int):
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 4, 8)), mx.ones((1, 2, 3)) * n]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    mx.eval(rec.cache, kv.keys, kv.values)
    return [rec, kv]


def _disk_mgr(tmp_path, **kw):
    disk = SpillDiskStore(tmp_path, namespace="t", num_workers=1, max_bytes=1 << 30)
    return YunshuAPCManager(
        num_blocks=8,
        block_size=16,
        disk=disk,
        overrides={"memory_max_gb": 1},
        head_marker=(900, 901),
        **kw,
    )


def _files(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("*.safetensors"))


def _ids(base, n=96):
    return list(range(base, base + n))


def test_stored_checkpoint_reaches_disk_without_close(tmp_path):
    m = _disk_mgr(tmp_path, persist_on_store=True)
    m.begin_request()
    assert m.store_exact_cache(_ids(2000), _cache(96))
    m.disk.flush()  # waits for the writer thread only; no close(), no eviction
    assert len(_files(tmp_path)) == 1


def test_default_keeps_evict_only_behaviour(tmp_path):
    m = _disk_mgr(tmp_path)
    m.begin_request()
    m.store_exact_cache(_ids(2000), _cache(96))
    m.disk.flush()
    assert _files(tmp_path) == []


def test_write_is_scheduled_not_inline(tmp_path):
    m = _disk_mgr(tmp_path, persist_on_store=True)
    gate, started = threading.Event(), threading.Event()
    real = m.disk._write_payload

    def slow(shard_id, hashes, payload):
        started.set()
        gate.wait(10)
        return real(shard_id, hashes, payload)

    m.disk._write_payload = slow
    m.begin_request()
    assert m.store_exact_cache(
        _ids(2000), _cache(96)
    )  # returns while the writer is blocked
    assert started.wait(5)
    assert _files(tmp_path) == []
    gate.set()
    m.disk.flush()
    assert len(_files(tmp_path)) == 1


def test_restart_finds_the_entry_after_a_kill(tmp_path):
    ids = _ids(2000)
    m = _disk_mgr(tmp_path, persist_on_store=True)
    m.begin_request()
    m.store_exact_cache(ids, _cache(96))
    m.disk.flush()
    # SIGKILL: the first manager is never closed
    fresh = _disk_mgr(tmp_path)
    cache, n = fresh.lookup_exact_cache(ids + [9, 9])
    assert n == 96 and cache is not None
    assert fresh.take_lookup(98, 96).tier == "ssd"
    fresh.close()


def test_half_written_file_is_ignored_on_restart(tmp_path):
    ids = _ids(2000)
    m = _disk_mgr(tmp_path, persist_on_store=True)
    m.begin_request()
    m.store_exact_cache(ids, _cache(96))
    m.disk.flush()
    (real,) = list(tmp_path.rglob("*.safetensors"))
    # a crash mid-write leaves a temp file that was never renamed; make it a truncated copy
    stray = real.with_name("exact-deadbeef.1-2.safetensors")
    stray.write_bytes(real.read_bytes()[:40])
    other = _disk_mgr(tmp_path)
    cache, n = other.lookup_exact_cache(_ids(9000) + [1])
    assert n == 0 and cache is None
    cache, n = other.lookup_exact_cache(ids + [9, 9])
    assert n == 96  # the complete entry still loads
    other.close()


def test_close_does_not_write_a_persisted_entry_twice(tmp_path):
    m = _disk_mgr(tmp_path, persist_on_store=True)
    m.begin_request()
    m.store_exact_cache(_ids(2000), _cache(96))
    m.disk.flush()
    (path,) = list(tmp_path.rglob("*.safetensors"))
    before = path.stat().st_mtime_ns
    writes = []
    real = type(m.disk)._write_payload_impl
    m.disk._write_payload_impl = lambda *a: writes.append(a) or real(m.disk, *a)
    m.close()
    assert writes == []
    assert path.stat().st_mtime_ns == before


def test_disk_budget_still_gates_the_write(tmp_path):
    m = _disk_mgr(tmp_path, persist_on_store=True)

    class Deny:
        def allow_write(self, size):
            return False

        def record_failure(self, e):
            raise AssertionError

        def note_written(self, path):
            raise AssertionError

    m.disk.budget = Deny()
    m.begin_request()
    assert m.store_exact_cache(_ids(2000), _cache(96))
    m.disk.flush()
    assert (
        _files(tmp_path) == []
    )  # cap / free-space reserve refused it; the request is unaffected


def test_full_queue_postpones_instead_of_writing_inline(tmp_path):
    m = _disk_mgr(tmp_path, persist_on_store=True)
    m.disk.queue_max_bytes = 1  # no room
    m.begin_request()
    assert m.store_exact_cache(_ids(2000), _cache(96))
    m.disk.flush()
    assert _files(tmp_path) == []
    m.close()  # close() still persists it
    assert len(_files(tmp_path)) == 1
