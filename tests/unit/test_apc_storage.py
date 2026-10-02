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
def test_parse_tier_specs():
    a, b, c = parse_tiers("/Volumes/A@128, ~/x/y , /mnt/nas@2.5!sim=110/4")
    assert (a.path, a.cap_gib, a.sim) == (Path("/Volumes/A"), 128.0, None)
    assert b.path == Path("~/x/y").expanduser() and b.cap_gib is None
    assert c.path == Path("/mnt/nas") and c.cap_gib == 2.5 and c.sim == (110e6, 0.004)
    assert parse_tiers(None) == [] and parse_tiers(" , ") == []


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
