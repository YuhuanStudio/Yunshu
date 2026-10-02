"""APC N-level storage: tier specs, device profiling (with a throttled simulated device), the
container codec, mover demotion, restore from lower tiers, the cost model, availability / remount
validation, corruption fallback, and token-identical output after a restore from an encoded tier."""

import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.apc import _sequence_hash  # noqa: E402
from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from yunshu_engine.apc_manager import YunshuAPCManager  # noqa: E402
from yunshu_engine.apc_storage import (  # noqa: E402
    DeviceProfile,
    FileTier,
    ProfileStore,
    TieredDiskStore,
    TierSpec,
    decode_file,
    encode_file,
    parse_tiers,
    probe_device,
    read_container_header,
)

BLOCK = 16
NS = "ns"


def _cache(n: int, seed: int = 0):
    mx.random.seed(seed)
    rec = ArraysCache(2)
    rec.cache = [mx.random.normal((1, 4, 8)), mx.random.normal((1, 2, 3))]
    kv = KVCache()
    kv.update_and_fetch(
        mx.random.normal((1, 2, n, 64)).astype(mx.bfloat16),
        mx.random.normal((1, 2, n, 64)).astype(mx.bfloat16),
    )
    mx.eval(rec.cache, kv.keys, kv.values)
    return [rec, kv]


def _flat(cache):
    out = []
    for c in cache:
        st = c.state
        out += [
            a for a in (st if isinstance(st, (list, tuple)) else [st]) if a is not None
        ]
    return out


def _same(a, b):
    for x, y in zip(_flat(a), _flat(b), strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert mx.array_equal(x, y).item()


def _toks(n, seed):
    r = random.Random(seed)
    return tuple(r.randrange(3, 800) for _ in range(n))


def _profile(read=2e9, write=1e9, lat=0.0001):
    return DeviceProfile("/x", read, write, lat, time.time())


def _store(tmp_path, soft_mb=0.0, lower=(), **kw):
    s = TieredDiskStore(tmp_path / "t0", namespace=NS, num_workers=1, max_bytes=1 << 30)
    s.profile = _profile(8e9, 8e9)
    s.soft_cap_bytes = int(soft_mb * (1 << 20))
    s.prefill_tps = kw.get("tps", 700.0)
    for i, (read, enc) in enumerate(lower):
        spec = TierSpec(tmp_path / f"t{i + 1}")
        spec.path.mkdir(
            parents=True, exist_ok=True
        )  # a mounted volume: its root exists
        s.add_lower(
            FileTier(
                spec,
                NS,
                _profile(read, read),
                cap_bytes=kw.get("cap", 1 << 30),
                name=f"tier{i + 1}",
                encoding=enc,
            )
        )
    return s


def _put(s, toks, seed):
    key = _sequence_hash(tuple(toks), 0, BLOCK)
    assert s.write_now(key, toks, 0, _cache(len(toks), seed), True)
    s.flush()
    return key


def _find(s, toks, extra=(5,)):
    return s.find_exact_prefix(list(toks) + list(extra), extra_hash=0, block_size=BLOCK)


# ── specs, profiles ──────────────────────────────────────────────────────
def test_tiers_on_one_volume_are_told_apart_by_directory(tmp_path):
    s = TieredDiskStore(tmp_path / "t0", namespace=NS, num_workers=1, max_bytes=1 << 30)
    for d in ("t1", "t2"):
        (tmp_path / d).mkdir()
        s.add_lower(FileTier(TierSpec(tmp_path / d), NS, _profile(), cap_bytes=1 << 30))
    names = [s.name, *(t.name for t in s.lower)]
    assert len(set(names)) == 3, names
    assert s.name.endswith("/t0") and any(n.endswith("/t2") for n in names)


def test_parse_tier_specs():
    a, b, c = parse_tiers("/Volumes/A@128, ~/x/y , /mnt/nas@2.5!sim=110/4")
    assert (a.path, a.cap_gib, a.sim) == (Path("/Volumes/A"), 128.0, None)
    assert b.path == Path("~/x/y").expanduser() and b.cap_gib is None
    assert c.path == Path("/mnt/nas") and c.cap_gib == 2.5 and c.sim == (110e6, 0.004)
    assert parse_tiers(None) == [] and parse_tiers(" , ") == []


def test_an_unmounted_volume_is_never_created(tmp_path):
    from yunshu_engine.apc_storage import ensure_root

    assert not ensure_root(Path("/Volumes/yunshu-no-such-volume-xyz/apc"))
    assert not Path("/Volumes/yunshu-no-such-volume-xyz").exists()
    assert ensure_root(tmp_path / "new" / "dir") and (tmp_path / "new" / "dir").is_dir()
    with pytest.raises(OSError):
        probe_device(Path("/Volumes/yunshu-no-such-volume-xyz/apc"), mb=8)


def test_probe_measures_a_throttled_device(tmp_path):
    fast = probe_device(tmp_path / "fast", mb=16)
    slow = probe_device(tmp_path / "slow", mb=8, sim=(40e6, 0.02))
    assert fast.read_bps > 5 * slow.read_bps
    assert 25e6 < slow.read_bps < 45e6 and 25e6 < slow.write_bps < 45e6
    assert 0.015 < slow.latency_s < 0.05 and slow.simulated
    assert not list((tmp_path / "slow").glob(".yunshu-probe-*"))  # probe files cleaned


def test_profile_store_roundtrip_and_ttl(tmp_path):
    store = ProfileStore(tmp_path / "p.json")
    prof = _profile()
    store.put("/mnt/a", prof)
    assert store.get("/mnt/a").read_bps == prof.read_bps
    assert store.get("/mnt/a", ttl_s=-1) is None and store.get("/mnt/b") is None


# ── container codec ──────────────────────────────────────────────────────
def test_encoded_container_roundtrips_and_detects_corruption(tmp_path):
    s = _store(tmp_path)
    key = _put(s, _toks(3000, 1), 1)
    src = s._exact_index[key]
    meta = s._meta_of(src)
    dst = tmp_path / "x.yscx"
    n = encode_file(src, dst, meta)
    head = read_container_header(dst)
    assert head is not None and head["orig_size"] == src.stat().st_size
    assert n == dst.stat().st_size
    out = tmp_path / "x.out"
    decode_file(dst, out, head)
    assert out.read_bytes() == src.read_bytes()
    raw = bytearray(dst.read_bytes())
    raw[len(raw) // 3] ^= 0xFF
    dst.write_bytes(bytes(raw))
    with pytest.raises(Exception):  # noqa: B017
        decode_file(dst, tmp_path / "bad.out", read_container_header(dst))
    dst.write_bytes(bytes(raw[:-5]))  # torn
    assert read_container_header(dst) is None


# ── mover, restore, cascade ──────────────────────────────────────────────
@pytest.mark.parametrize("enc", ["raw", "zstd"])
def test_overflow_is_demoted_and_restored_exactly(tmp_path, enc):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, enc)])
    keys = [_put(s, _toks(1500, i), i + 1) for i in range(3)]
    s.settle()
    t1 = s.lower[0]
    assert t1.index and s.moved >= 1
    assert s._root_used() <= s.soft_cap_bytes + 1 or not s._own_files()
    # the oldest checkpoint now lives below, and is found, cost-checked and restored bit-exact
    toks = _toks(1500, 0)
    hit = _find(s, toks)
    assert hit == (keys[0], 1500) and s._where.get(keys[0]) is t1
    loaded = s.load_exact_cache(keys[0], prefix_len=1500)
    assert loaded is not None and s.last_device == "tier1"
    _same(loaded[2], _cache(1500, 1))
    assert t1.hits == 1 and s.device_hits["tier1"] == 1
    suffixes = {e.path.suffix for e in t1.index.values()}
    assert suffixes == ({".yscx"} if enc == "zstd" else {".safetensors"})


def test_budget_hands_overflow_to_the_mover_instead_of_deleting(tmp_path):
    from yunshu_kv.disk_budget import DiskBudget

    s = _store(tmp_path, soft_mb=1.0, lower=[(1e9, "raw")])
    budget = DiskBudget(
        tmp_path / "t0", cap_bytes=1 << 20, reserve_pct=0, reserve_min_bytes=0
    )
    s.attach_budget(budget)
    for i in range(
        3
    ):  # a checkpoint is 0.8 MB: from the second write the budget is over its cap
        _put(s, _toks(1500, i), i + 1)
        assert len(s._exact_index) + len(s.lower[0].index) == i + 1, (
            "deleted, not demoted"
        )
        s.rebalance()  # the mover keeps up
    assert s.lower[0].index
    # with no lower tier able to take them (slower than a re-prefill) the budget deletes
    s.lower[0].profile = _profile(1e5, 1e5)
    before = len(s._exact_index)
    for i in range(3, 6):
        _put(s, _toks(1500, i), i + 1)
    assert len(s._exact_index) < before + 3


def test_cascade_down_through_three_tiers(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw"), (5e8, "raw")], cap=0)
    s.lower[
        0
    ].cap_bytes = 1_000_000  # tier1 holds about one checkpoint, the rest goes to tier2
    for i in range(4):
        _put(s, _toks(1500, i), i + 1)
        s.settle()
    assert s.lower[0].index and s.lower[1].index
    for i in range(
        4
    ):  # nothing was lost: every checkpoint restores from wherever it is
        toks = _toks(1500, i)
        hit = _find(s, toks)
        assert hit is not None and hit[1] == 1500
        loaded = s.load_exact_cache(hit[0], prefix_len=1500)
        assert loaded is not None
        _same(loaded[2], _cache(1500, i + 1))


def test_fresh_spill_replaces_a_slower_duplicate(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    toks = _toks(1500, 0)
    k = _put(s, toks, 1)
    for i in (1, 2):
        _put(s, _toks(1500, i), i + 1)
    s.settle()
    assert k in s.lower[0].index
    s.write_now(
        k, toks, 0, _cache(1500, 1), True
    )  # RAM evicted it again: written to the primary
    s.flush()
    assert k not in s.lower[0].index and k in s._exact_index


# ── superseded checkpoints are garbage on disk too ───────────────────────
def _mgr_for(disk, hot_entries=1):
    return YunshuAPCManager(
        num_blocks=8,
        block_size=BLOCK,
        disk=disk,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
        max_entries=hot_entries,
    )


def _store_req(m, toks, seed, *, head=False):
    m.begin_request()
    assert m.store_exact_cache(list(toks), _cache(len(toks), seed))
    if head:
        with m._plock:
            m._head_keys.add(_sequence_hash(tuple(toks), 0, BLOCK))


def test_a_grown_conversation_supersedes_its_older_checkpoints_on_disk(tmp_path):
    disk = TieredDiskStore(
        tmp_path / "t0", namespace=NS, num_workers=1, max_bytes=1 << 30
    )
    disk.profile = _profile(8e9, 8e9)
    m = _mgr_for(disk)
    session = _toks(400, 1)
    key = lambda toks: _sequence_hash(tuple(toks), 0, BLOCK)  # noqa: E731
    head, other = session[:100], _toks(300, 2)
    _store_req(m, head, 1, head=True)  # the shared system-turn checkpoint
    _store_req(m, session[:200], 2)  # request 1 of the session
    _store_req(
        m, session[:300], 3
    )  # request 2 grows it: request 1's checkpoint is garbage
    _store_req(m, other, 4)
    _store_req(m, session, 5)  # request 3
    _store_req(m, _toks(250, 6), 6)  # push everything through the SSD
    disk.flush()
    stored = set(disk._exact_index)
    assert key(head) in stored, "heads are never superseded"
    assert key(session[:200]) not in stored and key(session[:300]) not in stored
    assert m.disk_superseded >= 1


def test_supersede_also_works_for_checkpoints_written_straight_to_the_ssd(tmp_path):
    """RAM too small to keep a checkpoint: it is written synchronously inside store_exact_cache,
    before the manager could record its generation afterwards."""
    from yunshu_engine.apc_manager import SpillDiskStore

    disk = SpillDiskStore(tmp_path, namespace=NS, num_workers=1, max_bytes=1 << 30)
    m = YunshuAPCManager(
        num_blocks=8,
        block_size=BLOCK,
        disk=disk,
        overrides={"memory_max_gb": 0.0004, "checkpoint_interval_tokens": 0},
        max_entries=1,
    )
    sessions = [_toks(2000, i) for i in range(3)]
    for turn in range(1, 5):
        for i, toks in enumerate(sessions):
            _store_req(m, toks[: 400 + 300 * turn], i + turn)
    disk.flush()
    assert len(disk._exact_index) == 3, "only each session's newest checkpoint stays"
    assert m.disk_superseded >= 6


def test_supersede_reaches_lower_tiers_and_spares_unknown_files(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    m = _mgr_for(s)
    session = _toks(400, 1)
    _store_req(m, session[:200], 1)
    _store_req(
        m, _toks(300, 2), 2
    )  # session[:200] reaches the SSD, the mover demotes it
    s.flush()
    s.settle()
    old = _sequence_hash(tuple(session[:200]), 0, BLOCK)
    assert old in s.lower[0].index or old in s._exact_index
    # a checkpoint this process never stored or restored (an earlier run) is left to LRU
    with m._plock:
        m._born.pop(old, None)
    _store_req(m, session[:300], 3)
    _store_req(m, _toks(250, 4), 4)
    s.flush()
    assert old in s.lower[0].index or old in s._exact_index
    # once it is a known earlier-request checkpoint it goes, wherever it is
    with m._plock:
        m._born[old] = 0
    _store_req(m, session, 5)
    _store_req(m, _toks(260, 6), 6)
    s.flush()
    s.settle()
    assert old not in s.lower[0].index and old not in s._exact_index


# ── cost model, availability, corruption ─────────────────────────────────
def test_a_tier_slower_than_reprefill_is_not_used_or_filled(tmp_path):
    s = _store(
        tmp_path, soft_mb=0.2, lower=[(1e5, "raw")]
    )  # 0.1 MB/s: far slower than prefill
    for i in range(3):
        _put(s, _toks(1500, i), i + 1)
    s.settle()
    assert not s.lower[0].index and s.dropped_not_worth >= 1
    # and a candidate already there would not be chosen
    s.lower[0].index.clear()


def test_cost_model_prefers_the_cheaper_tier_and_respects_gain(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(2e9, "raw")])
    keys = [_put(s, _toks(1500, i), i + 1) for i in range(3)]
    s.settle()
    toks = _toks(1500, 0)
    assert _find(s, toks) == (keys[0], 1500)
    # a HOT hit nearly as long leaves nothing to gain from a restore
    s.lower[0].profile = _profile(
        1e7, 1e7
    )  # 0.08 s to restore: not worth one more token
    assert (
        s.find_exact_prefix(
            list(toks) + [5], extra_hash=0, min_prefix_tokens=1499, block_size=BLOCK
        )
        is None
    )
    assert _find(s, toks) == (
        keys[0],
        1500,
    )  # but worth the whole 1500 (2 s of prefill)
    s.lower[0].profile = _profile(2e9, 2e9)
    # a prefill 1000x faster makes every restore a loss
    s.prefill_tps = 1e7
    assert _find(s, toks) is None


def test_unavailable_tier_is_skipped_and_validated_on_return(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    keys = [_put(s, _toks(1500, i), i + 1) for i in range(3)]
    s.settle()
    t1 = s.lower[0]
    assert t1.index
    held = [e.path for e in t1.index.values()]
    # unplugged: not available -> never consulted
    t1._avail = (time.monotonic() + 1e6, False)
    assert _find(s, _toks(1500, 0)) is None
    # while it was away one file got corrupted and one torn
    victim, torn = held[0], held[-1]
    data = bytearray(victim.read_bytes())
    data[:16] = b"\0" * 16
    victim.write_bytes(bytes(data))
    if torn != victim:
        torn.write_bytes(torn.read_bytes()[: torn.stat().st_size // 2])
    # remount: re-scan and validate
    t1._avail = (0.0, False)
    t1._was_available = False
    assert t1.available()
    assert t1.invalidated >= 1
    assert all(p.exists() for p in (e.path for e in t1.index.values()))
    for p in (victim, torn):
        assert p not in {e.path for e in t1.index.values()}
        assert not p.exists()
    del keys


def test_a_volume_that_fails_mid_scan_keeps_its_files(tmp_path, monkeypatch):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    for i in range(3):
        _put(s, _toks(1500, i), i + 1)
    s.settle()
    t1 = s.lower[0]
    files = [e.path for e in t1.index.values()]
    assert files
    real_open = open

    def flaky(path, *a, **kw):
        if str(path).endswith(".safetensors") and str(t1.dir) in str(path):
            raise OSError(5, "Input/output error")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", flaky)
    t1._avail, t1._was_available = (0.0, False), False
    assert not t1.available()  # the scan hit an I/O error: unavailable, nothing deleted
    monkeypatch.undo()
    assert all(p.exists() for p in files) and t1.invalidated == 0
    t1._avail = (0.0, False)
    assert t1.available() and len(t1.index) == len(files)


def test_a_busy_volume_is_not_taken_for_an_unplugged_one(tmp_path, monkeypatch):
    import yunshu_engine.apc_storage as mod

    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    for i in range(3):
        _put(s, _toks(1500, i), i + 1)
    s.settle()
    t1 = s.lower[0]
    assert t1.index and t1.available()
    held = dict(t1.index)
    t1.CHECK_TIMEOUT_S = 0.05
    real = mod.ensure_root
    monkeypatch.setattr(mod, "ensure_root", lambda p: (time.sleep(0.5), real(p))[1])
    t1._avail = (
        0.0,
        True,
    )  # due for a re-check; the check now takes 0.5 s (a stat behind a big fsync)
    assert t1.available(), (
        "a slow answer from a volume that answered recently is not an absence"
    )
    assert t1.index == held, "no re-scan, no empty index while it is only busy"
    time.sleep(0.6)
    monkeypatch.setattr(mod, "ensure_root", real)
    t1._avail = (0.0, True)
    assert t1.available() and t1.index == held


def test_observed_restore_speed_overrides_an_optimistic_probe(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(2e9, "raw")])
    keys = [_put(s, _toks(1500, i), i + 1) for i in range(3)]
    s.settle()
    toks = _toks(1500, 0)
    assert _find(s, toks) == (keys[0], 1500)
    t1 = s.lower[0]
    e = t1.index[keys[0]]
    assert t1.restore_s(e) < 0.01  # the probe says: milliseconds
    t1.note_restore(256 << 20, 400.0)  # but a real restore delivered 0.64 MiB/s
    assert t1.restore_s(e) == pytest.approx(e.orig / t1.eff_bps)
    s.prefill_tps = (
        5000.0  # 1500 tokens re-prefill in 0.3 s; the real restore takes 12 ms x ...
    )
    assert _find(s, toks) is None and s.cost_rejected >= 1
    assert s.snapshot()[1]["effective_read_bps"] == pytest.approx((256 << 20) / 400.0)


def test_a_file_being_read_is_not_deleted_under_the_reader(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    keys = [_put(s, _toks(1500, i), i + 1) for i in range(3)]
    s.settle()
    t1 = s.lower[0]
    k = next(iter(t1.index))
    path = t1.index[k].path
    s._lease(path)  # a lookup is loading it
    s.drop_exact(k)  # a newer checkpoint supersedes it meanwhile
    assert k not in t1.index and path.exists(), "gone from the index, still on disk"
    s._release(path)
    assert not path.exists(), "unlinked once the reader is done"
    # the primary store too
    k2 = _put(s, _toks(1500, 9), 9)
    p2 = s._exact_index[k2]
    s._lease(p2)
    assert s.drop_exact(k2) and p2.exists() and k2 not in s._exact_index
    s._release(p2)
    assert not p2.exists()
    del keys


def test_a_read_that_cannot_complete_never_deletes_a_good_file(tmp_path, monkeypatch):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "raw")])
    for i in range(3):
        _put(s, _toks(1500, i), i + 1)
    s.settle()
    t1 = s.lower[0]
    k = next(iter(t1.index))
    path = t1.index[k].path

    def eio(*a, **kw):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(s, "_load_exact_cache_file", eio)
    s._where[k] = t1
    assert s.load_exact_cache(k, prefix_len=len(t1.index[k].tokens)) is None
    assert path.exists() and k in t1.index and t1.invalidated == 0
    monkeypatch.undo()
    # a file that vanished under the reader is a quiet miss
    path.unlink()
    s._where[k] = t1
    assert s.load_exact_cache(k, prefix_len=len(t1.index[k].tokens)) is None
    assert k not in t1.index and t1.invalidated == 0


def test_prefill_speed_is_observed_from_the_tokens_really_computed():
    from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner

    seen = []
    disk = SimpleNamespace(observe_prefill=lambda n, t: seen.append((n, t)))
    runner = SimpleNamespace(
        apc_manager=SimpleNamespace(disk=disk), _active_jobs=lambda: 0
    )
    st = RunStats()
    st.t_admit, st.t_first, st.cache_reload_ms = 10.0, 15.0, 500.0
    st.cached_tokens = 25000
    st.prefill_total = 28000  # counts the cached tokens while the prompt is processed
    job = SimpleNamespace(stats=st, ids=list(range(28000)))
    VLMBatchRunner._observe_prefill(runner, job)
    assert seen == [(3000, pytest.approx(4.5))]


def test_corrupt_lower_file_falls_back_and_is_deleted(tmp_path):
    s = _store(tmp_path, soft_mb=0.2, lower=[(1e9, "zstd")])
    keys = [_put(s, _toks(1500, i), i + 1) for i in range(3)]
    s.settle()
    t1 = s.lower[0]
    e = t1.index[keys[0]]
    raw = bytearray(e.path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    e.path.write_bytes(bytes(raw))
    assert _find(s, _toks(1500, 0)) == (keys[0], 1500)
    assert s.load_exact_cache(keys[0], prefix_len=1500) is None
    assert keys[0] not in t1.index and not e.path.exists() and t1.invalidated == 1


def test_auto_encoding_compresses_only_where_it_pays(tmp_path):
    slow = _store(
        tmp_path / "a", soft_mb=0.2, lower=[(3e7, "auto")]
    )  # 30 MB/s: a network share
    fast = _store(tmp_path / "b", soft_mb=0.2, lower=[(8e9, "auto")])  # NVMe
    slow.lower[0].profile = _profile(3e7, 3e7)
    for st in (slow, fast):
        st.prefill_tps = (
            1.0  # restore always worth it here; only the encoding is under test
        )
        for i in range(3):
            _put(st, _toks(1500, i), i + 1)
        st.settle()
    assert {e.path.suffix for e in slow.lower[0].index.values()} == {".yscx"}
    assert {e.path.suffix for e in fast.lower[0].index.values()} == {".safetensors"}


# ── manager level: provenance + token identity ───────────────────────────
def test_manager_reports_device_and_stays_token_identical(tmp_path):
    from tests.unit.test_apc_warm import _tiny_model
    from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner

    class Stop:
        def __init__(self):
            self.eos = set()

        def add_eos_token_ids(self, ids):
            self.eos |= set(ids or [])

        def __call__(self, token):
            return int(token) in self.eos

    proc = SimpleNamespace(tokenizer=SimpleNamespace(stopping_criteria=Stop()))
    model = _tiny_model()
    rnd = random.Random(11)
    a = [rnd.randrange(3, 800) for _ in range(300)]
    b = [rnd.randrange(3, 800) for _ in range(300)]
    c = [rnd.randrange(3, 800) for _ in range(300)]
    a2 = a + [rnd.randrange(3, 800) for _ in range(40)]

    def run(runner, ids):
        st = RunStats()
        out = list(runner.iter_tokens(ids, max_tokens=8, stats=st, allow_draft=False))
        return out, st

    disk = TieredDiskStore(
        tmp_path / "t0", namespace=NS, num_workers=1, max_bytes=1 << 30
    )
    disk.profile = _profile(8e9, 8e9)
    disk.soft_cap_bytes = 1  # everything the SSD receives moves down
    disk.prefill_tps = 700.0
    (tmp_path / "t1").mkdir()
    disk.add_lower(
        FileTier(
            TierSpec(tmp_path / "t1"),
            NS,
            _profile(1e9, 1e9),
            cap_bytes=1 << 30,
            name="hdd",
            encoding="zstd",
        )
    )
    mgr = YunshuAPCManager(
        num_blocks=64, block_size=16, disk=disk,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0}, max_entries=1,
    )  # fmt: skip
    runner = VLMBatchRunner(model, processor=proc, apc_manager=mgr, apc_semantic_hash=0)
    run(runner, a)
    run(runner, b)  # a leaves RAM for the SSD
    run(runner, c)
    disk.flush()
    disk.settle()
    assert disk.lower[0].index, "nothing was demoted"
    out_low, st_low = run(runner, a2)
    assert st_low.cache_tier == "ssd" and st_low.cache_device == "hdd"
    assert st_low.cached_tokens == len(a) - 1

    ram = YunshuAPCManager(
        num_blocks=64, block_size=16,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0}, max_entries=8,
    )  # fmt: skip
    rr = VLMBatchRunner(model, processor=proc, apc_manager=ram, apc_semantic_hash=0)
    run(rr, a)
    out_ram, st_ram = run(rr, a2)
    assert st_ram.cache_tier == "ram"
    out_cold, _ = run(VLMBatchRunner(model, processor=proc), a2)
    assert out_low == out_ram == out_cold
    assert any(
        t["name"] == "hdd" and t["hits"] == 1 for t in mgr.snapshot()["storage_tiers"]
    )


def test_agent_sessions_leave_one_or_two_checkpoints_each_on_the_ssd(tmp_path):
    """Three interleaved growing sessions through the real runner, RAM too small to hold them:
    restored checkpoints are superseded in RAM, and their SSD copies must still be dropped when
    the session's next checkpoint is written (the generation of a RAM-superseded entry is kept)."""
    from tests.unit.test_apc_warm import _tiny_model
    from yunshu_engine.apc_manager import SpillDiskStore
    from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner

    class Stop:
        def __init__(self):
            self.eos = set()

        def add_eos_token_ids(self, ids):
            self.eos |= set(ids or [])

        def __call__(self, token):
            return int(token) in self.eos

    disk = SpillDiskStore(tmp_path, namespace=NS, num_workers=1, max_bytes=1 << 30)
    mgr = YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        disk=disk,
        overrides={"memory_max_gb": 0.002, "checkpoint_interval_tokens": 128},
        max_entries=4,
        head_marker=(900, 901),
    )
    proc = SimpleNamespace(tokenizer=SimpleNamespace(stopping_criteria=Stop()))
    runner = VLMBatchRunner(
        _tiny_model(), processor=proc, apc_manager=mgr, apc_semantic_hash=12345
    )
    r = random.Random(3)
    head = [r.randrange(3, 800) for _ in range(150)]
    sess = [
        [7, *head, 900, 901] + [r.randrange(3, 800) for _ in range(3000)]
        for _ in range(3)
    ]
    cached = prompt = 0
    for turn in range(1, 6):
        for s in range(3):
            ids = sess[s][: 250 + turn * 200]
            st = RunStats()
            list(runner.iter_tokens(ids, max_tokens=2, stats=st, allow_draft=False))
            if turn > 1:
                cached += st.cached_tokens
                prompt += len(ids)
    disk.flush()
    assert len(disk._exact_index) <= 1 + 3 * 2, "stale checkpoints pile up on the SSD"
    assert cached / prompt > 0.7
    assert mgr.disk_superseded >= 12
