"""APC WARM tier: demotion HOT -> WARM -> SSD, promotion on hit, budgets, corruption fallback, and
token-identical output of a WARM-lossless / SSD restore versus a HOT hit (tiny hybrid Qwen3.5)."""

import random
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.apc import _sequence_hash  # noqa: E402
from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from yunshu_engine.apc_manager import SpillDiskStore, YunshuAPCManager  # noqa: E402
from yunshu_engine.apc_warm import WarmTier  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_test_disk_restore(monkeypatch):
    # These tiny roundtrip tests exercise storage, not the host's free-RAM policy.
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")


def _cache(n: int, seed: int = 0, dtype=mx.bfloat16):
    """Hybrid-style prompt cache: float32 recurrent state + bf16 dense KV of n tokens."""
    mx.random.seed(seed)
    rec = ArraysCache(2)
    rec.cache = [mx.random.normal((1, 4, 8)), mx.random.normal((1, 2, 3))]
    kv = KVCache()
    k = mx.random.normal((1, 2, n, 64)).astype(dtype)
    v = mx.random.normal((1, 2, n, 64)).astype(dtype)
    kv.update_and_fetch(k, v)
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
    xs, ys = _flat(a), _flat(b)
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys, strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert mx.array_equal(x, y).item()


def _mgr(mode="lossless", warm_mb=64, hot_entries=1, disk=None, **kw):
    return YunshuAPCManager(
        num_blocks=8,
        block_size=16,
        disk=disk,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
        head_marker=None,
        max_entries=hot_entries,
        warm_mode=mode,
        warm_bytes=warm_mb << 20,
        **kw,
    )


def _toks(n, seed):
    r = random.Random(seed)
    return tuple(r.randrange(3, 800) for _ in range(n))


def _store(m, toks, seed):
    m.begin_request()
    assert m.store_exact_cache(list(toks), _cache(len(toks), seed))


def _settle(m):
    m.warm.wait_idle()
    m._settle_warm()


def test_lossless_roundtrip_is_bit_exact():
    w = WarmTier("lossless", 64 << 20)
    src = _cache(300, 1)
    entry = SimpleNamespace(token_ids=(1, 2, 3), extra_hash=0, prompt_cache=src)
    assert w.demote(1, entry)
    w.wait_idle()
    w.drain()
    assert w.bytes > 0 and len(w.entries) == 1
    cache, _ent = w.take(1)
    _same(src, cache)
    assert w.bytes == 0 and not w.entries
    w.close()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_multi_chunk_arrays_roundtrip_bit_exact(dtype):
    w = WarmTier("lossless", 256 << 20)
    kv = KVCache()
    mx.random.seed(4)
    kv.update_and_fetch(  # 10+ MB per array: three 4 MiB chunks, an odd tail
        mx.random.normal((1, 2, 20011, 64)).astype(dtype),
        mx.random.normal((1, 2, 20011, 64)).astype(dtype),
    )
    mx.eval(kv.keys, kv.values)
    src = [kv]
    assert w.demote(1, SimpleNamespace(token_ids=(1,), extra_hash=0, prompt_cache=src))
    w.wait_idle()
    w.drain()
    cache, _ent = w.take(1)
    _same(src, cache)
    w.close()


def test_demotion_on_eviction_and_promotion_on_hit():
    m = _mgr()
    a, b = _toks(200, 1), _toks(200, 2)
    _store(m, a, 1)
    _store(m, b, 2)  # one HOT entry: a is demoted to WARM
    _settle(m)
    assert m.warm.snapshot()["warm_entries"] == 1
    cache, n = m.lookup_exact_cache(list(a) + [5, 6, 7])
    assert n == 200
    assert m.lookups[-1].tier == "warm"
    _same(cache, _cache(200, 1))
    # promoted: a is HOT again, b went down to WARM (one HOT entry)
    _settle(m)
    assert m.warm.stats.hits == 1
    assert [len(e.token_ids) for e in m._exact_cache.values()] == [200]
    assert m.warm.snapshot()["warm_entries"] == 1
    assert m.tier_hits["warm"] == 1


def test_hit_while_still_compressing_is_served_from_hot_arrays():
    m = _mgr()
    a, b = _toks(120, 3), _toks(120, 4)
    _store(m, a, 3)
    _store(m, b, 4)  # a is in flight (or already settled); either way it is found
    cache, n = m.lookup_exact_cache(list(a) + [9])
    assert n == 120
    _same(cache, _cache(120, 3))


def test_warm_budget_overflow_goes_to_ssd_exact(tmp_path):
    disk = SpillDiskStore(tmp_path, namespace="w", num_workers=1, max_bytes=1 << 30)
    m = _mgr(warm_mb=1, disk=disk)  # 1 MiB: only the last demoted entry fits
    seqs = [_toks(2500, s) for s in (1, 2, 3, 4)]
    for i, t in enumerate(seqs):
        _store(m, t, i + 1)
        _settle(m)
    disk.flush()
    snap = m.warm.snapshot()
    assert snap["warm_bytes"] <= snap["warm_max_bytes"]
    assert snap["warm_evicted_to_ssd"] >= 1
    # the entry pushed out of WARM is on SSD, bit-exact
    m._exact_cache.clear()
    m.warm.entries.clear()
    cache, n = m.lookup_exact_cache(list(seqs[0]) + [1])
    assert n == 2500 and m.lookups[-1].tier == "ssd"
    _same(cache, _cache(2500, 1))


def test_a_longer_ssd_prefix_is_not_preceded_by_a_wasted_warm_decode(tmp_path):
    disk = SpillDiskStore(tmp_path, namespace="w", num_workers=1, max_bytes=1 << 30)
    m = _mgr(disk=disk, hot_entries=1)
    base = _toks(300, 5)
    short, long_ = (
        base[:200],
        base,
    )  # one session: an older (short) and the latest checkpoint
    _store(m, short, 1)
    _store(m, _toks(150, 9), 9)  # short goes down to WARM
    _settle(m)
    assert m.warm.snapshot()["warm_entries"] == 1
    key = _sequence_hash(tuple(long_), 0, 16)
    assert disk.write_now(
        key, long_, 0, _cache(300, 2), True
    )  # the longer one lives on SSD only
    disk.flush()
    cache, n = m.lookup_exact_cache(list(long_) + [7])
    assert n == 300 and m.lookups[-1].tier == "ssd"
    assert m.warm.stats.hits == 0, "the short WARM entry was decoded for nothing"
    del cache


def test_corrupt_warm_entry_falls_back_cold_and_is_dropped():
    m = _mgr()
    a, b = _toks(200, 1), _toks(200, 2)
    _store(m, a, 1)
    _store(m, b, 2)
    _settle(m)
    (ent,) = m.warm.entries.values()
    blob = ent.layers[1].items[0]
    chunk = bytearray(blob.chunks[0])
    chunk[len(chunk) // 2] ^= 0xFF
    blob.chunks[0] = bytes(chunk)
    _cache_out, n = m.lookup_exact_cache(list(a) + [5])
    assert n == 0 and m.lookups[-1].tier == "none"
    assert m.warm.stats.corrupt == 1 and not m.warm.entries


def test_supersede_drops_older_request_warm_entry():
    m = _mgr(hot_entries=1)
    a = _toks(150, 1)
    _store(m, a, 1)
    _store(m, _toks(150, 2), 2)
    _settle(m)
    assert m.warm.snapshot()["warm_entries"] == 1
    m.begin_request()
    assert m.store_exact_cache(list(a) + list(_toks(40, 9)), _cache(190, 5))
    assert m.warm.snapshot()["warm_entries"] == 0


def test_lossy_int8_keeps_ssd_exact_and_drops_on_overflow(tmp_path):
    disk = SpillDiskStore(tmp_path, namespace="l", num_workers=1, max_bytes=1 << 30)
    m = _mgr(mode="int8", warm_mb=64, disk=disk)
    a, b = _toks(256, 1), _toks(256, 2)
    _store(m, a, 1)
    _store(m, b, 2)
    disk.flush()
    ref = _cache(256, 1)
    # lossy WARM is smaller than the exact form and close to it
    assert m.warm.bytes < sum(x.nbytes for x in _flat(ref))
    cache, n = m.lookup_exact_cache(list(a) + [4])
    assert n == 256 and m.lookups[-1].tier == "warm"
    kv_ref, kv_got = ref[1].keys[..., :256, :], cache[1].keys[..., :256, :]
    err = mx.abs(kv_ref.astype(mx.float32) - kv_got.astype(mx.float32)).max().item()
    assert 0 < err < 0.1
    _same([ref[0]], [cache[0]])  # recurrent state stays exact
    # the SSD copy is the exact one
    m._exact_cache.clear()
    m.warm.entries.clear()
    exact, n = m.lookup_exact_cache(list(a) + [4])
    assert n == 256 and m.lookups[-1].tier == "ssd"
    _same(exact, ref)


def test_unstorable_layer_is_rejected_not_lost():
    w = WarmTier("lossless", 64 << 20)

    class Odd:
        pass

    entry = SimpleNamespace(token_ids=(1,), extra_hash=0, prompt_cache=[Odd()])
    assert not w.demote(1, entry)
    assert w.stats.rejected == 1
    w.close()


# ── end to end on a tiny hybrid Qwen3.5: WARM / SSD restores equal a HOT hit ──
def _tiny_model():
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    cfg = dict(
        model_type="qwen3_5", hidden_size=256, intermediate_size=512,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=64,
        linear_value_head_dim=64, linear_conv_kernel_dim=4, num_hidden_layers=4,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=1024, num_key_value_heads=1,
        max_position_embeddings=4096, head_dim=256, tie_word_embeddings=False,
    )  # fmt: skip
    mx.random.seed(3)
    lm = LanguageModel(
        TextConfig(**cfg),
        SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=1020, video_token_id=1021, vision_start_token_id=1022,
        ),
    )  # fmt: skip
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())

    class Embeds:
        def __init__(self, e):
            self.e = e

        def to_dict(self):
            return {"inputs_embeds": self.e}

    return SimpleNamespace(
        language_model=lm,
        config=SimpleNamespace(image_token_index=None),
        get_input_embeddings=lambda ids, pv, mask=None, **kw: Embeds(
            lm.model.embed_tokens(ids)
        ),
    )


def test_warm_lossless_and_ssd_hits_are_token_identical_to_hot(tmp_path):
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
    rnd = random.Random(7)
    a = [rnd.randrange(3, 800) for _ in range(260)]
    b = [rnd.randrange(3, 800) for _ in range(260)]
    a2 = a + [rnd.randrange(3, 800) for _ in range(40)]

    def run(runner, ids):
        st = RunStats()
        out = list(runner.iter_tokens(ids, max_tokens=8, stats=st, allow_draft=False))
        return out, st

    def mk(mode, entries, disk=None):
        mgr = YunshuAPCManager(
            num_blocks=64, block_size=16, disk=disk,
            overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
            max_entries=entries, warm_mode=mode, warm_bytes=256 << 20,
        )  # fmt: skip
        runner = VLMBatchRunner(
            model, processor=proc, apc_manager=mgr, apc_semantic_hash=0
        )
        return mgr, runner

    _hot_mgr, hot = mk("off", 8)
    run(hot, a)
    out_hot, st_hot = run(hot, a2)
    assert st_hot.cache_tier == "ram"

    warm_mgr, warm = mk("lossless", 1)
    run(warm, a)
    run(warm, b)  # a is demoted to WARM
    warm_mgr.warm.wait_idle()
    out_warm, st_warm = run(warm, a2)
    assert st_warm.cache_tier == "warm" and st_warm.cached_tokens == len(a) - 1

    disk = SpillDiskStore(tmp_path, namespace="e", num_workers=1, max_bytes=1 << 30)
    _ssd_mgr, ssd = mk("off", 1, disk)
    run(ssd, a)
    run(ssd, b)
    disk.flush()
    out_ssd, st_ssd = run(ssd, a2)
    assert st_ssd.cache_tier == "ssd"

    cold, _ = run(VLMBatchRunner(model, processor=proc), a2)
    assert out_hot == out_warm == out_ssd == cold
