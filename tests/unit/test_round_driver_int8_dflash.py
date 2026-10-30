"""int8 KV on the DFlash / plain-verify batch (row 17, stage 2) against the in-graph REFERENCE kernels.

The int8 tile / fused tree-attention kernels are written separately; until they land the tests install
``tile_partials_q8_reference`` / ``tree_attention_fused_q8_reference`` (dequantize in-graph, bf16 kernel), which
define the contract the kernels must reproduce bit for bit.
"""

import pytest

from .test_round_driver_dflash import _draft, target  # noqa: F401  (fixtures)

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def reference_kernels(monkeypatch):
    from yunshu_engine import tree_verify as tv

    monkeypatch.setattr(tv, "TILE_PARTIALS_Q8", tv.tile_partials_q8_reference)
    monkeypatch.setattr(
        tv, "TREE_ATTENTION_FUSED_Q8", tv.tree_attention_fused_q8_reference
    )


def _random_cache(n, seed=0, heads=1, d=256):
    """A bf16 KVCache holding ``n`` random keys / values."""
    from mlx_vlm.models.cache import KVCache

    mx.random.seed(seed)
    k = mx.random.normal((1, heads, n, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, heads, n, d)).astype(mx.bfloat16)
    c = KVCache()
    c.update_and_fetch(k, v)
    return c


def test_quantize_dequantize_roundtrip_error_is_small():
    from yunshu_engine.kernels.int8_kv import Int8KVCache, dequantize_kv

    c = _random_cache(200)
    q = Int8KVCache.from_kv(c)
    assert q.keys.dtype == mx.int8 and q.k_scales.dtype == mx.float16
    assert q.keys.shape[2] % 64 == 0 and q.offset == 200
    assert q.keys.shape[:3] == q.k_scales.shape[:3] and q.k_scales.shape[3] == 256 // 32
    k16 = c.keys[..., :200, :].astype(mx.float32)
    kd = dequantize_kv(q.keys[..., :200, :], q.k_scales[..., :200, :]).astype(
        mx.float32
    )
    rel = float(mx.linalg.norm(k16 - kd) / mx.linalg.norm(k16))
    assert rel < 0.02, (
        rel
    )  # int8 per 32-dim group: ~0.5 % relative error on gaussian data
    assert q.nbytes < 0.6 * (c.keys.nbytes + c.values.nbytes)


def test_update_and_fetch_quantizes_on_write_like_from_kv():
    """Appending bf16 rows one window at a time stores the same codes as quantizing the whole cache once."""
    from yunshu_engine.kernels.int8_kv import Int8KVCache

    c = _random_cache(40)
    whole = Int8KVCache.from_kv(c)
    grown = Int8KVCache()
    for lo, hi in ((0, 7), (7, 8), (8, 40)):
        grown.update_and_fetch(c.keys[..., lo:hi, :], c.values[..., lo:hi, :])
    assert grown.offset == 40
    for a, b in ((grown.keys, whole.keys), (grown.k_scales, whole.k_scales)):
        assert bool(mx.array_equal(a[..., :40, :], b[..., :40, :]))
    grown.trim(3)
    assert grown.offset == 37  # a rejected window is a trim, never a re-quantization


@pytest.mark.parametrize("width", [1, 4, 9])
def test_tree_attention_int8_equals_the_dequantized_reference_bit_for_bit(width):
    """The int8 path (shared prefix via the installed hook, window / tail rows dequantized into the bf16 tail
    copy) equals the bf16 kernel over the dequantized buffers, and stays close to bf16 attention."""
    from yunshu_engine.kernels.int8_kv import Int8KVCache
    from yunshu_engine.tree_verify import (
        TreeShape,
        tree_attention,
        tree_attention_int8_reference,
    )

    n0 = 200
    c = _random_cache(n0 + width, seed=1)
    q8 = Int8KVCache.from_kv(c)
    mx.random.seed(5)
    queries = mx.random.normal((1, 4, width, 256)).astype(mx.bfloat16)
    shape = TreeShape(list(range(-1, width - 1)))  # a chain
    scale = 256**-0.5
    got = tree_attention(queries, q8, scale, shape, n0)
    ref = tree_attention_int8_reference(queries, q8, scale, shape, n0)
    assert bool(mx.array_equal(got, ref))
    exact = tree_attention(queries, c, scale, shape, n0)
    err = float(
        mx.linalg.norm(got.astype(mx.float32) - exact.astype(mx.float32))
        / mx.linalg.norm(exact.astype(mx.float32))
    )
    assert err < 0.05, err


def test_a_token_attends_the_same_whatever_the_window_width():
    """Window-width independence: row 0 of a 4-wide window and the 1-wide window at the same position
    give identical bits on the int8 cache (verify == plain decode)."""
    from yunshu_engine.kernels.int8_kv import Int8KVCache
    from yunshu_engine.tree_verify import TreeShape, tree_attention

    n0 = 300
    c = _random_cache(n0 + 4, seed=2)
    wide = Int8KVCache.from_kv(c)
    narrow = Int8KVCache.from_kv(c)
    narrow.trim(3)  # the same codes, one window row stored
    mx.random.seed(6)
    q = mx.random.normal((1, 4, 4, 256)).astype(mx.bfloat16)
    a = tree_attention(q, wide, 256**-0.5, TreeShape([-1, 0, 1, 2]), n0)
    b = tree_attention(q[:, :, :1], narrow, 256**-0.5, TreeShape([-1]), n0)
    assert bool(mx.array_equal(a[:, :, :1], b))


def _tokens(lm, drafter, prompts, kv_format, tokens=12, apc=None):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(
        lm,
        dflash_drafter=drafter,
        plain_verify=drafter is None,
        stop_tokens=set(),
        chunk=16,
        kv_format=kv_format,
        apc=apc,
    )
    out: dict = {}
    for i, p in enumerate(prompts):
        d.add(Request(p, tokens, handle=i))
    while d.busy():
        for e in d.step():
            out.setdefault(e.handle, []).append(e.token)
    return out, d


PROMPTS = [
    list(range(4, 60)),
    list(range(8, 70)),
    [(5 * i + 3) % 500 for i in range(44)],
]


def test_int8_dflash_rows_decode_in_int8_caches_and_hold_less_memory(target):  # noqa: F811
    from yunshu_engine.kernels.int8_kv import Int8KVCache
    from yunshu_engine.round_driver.driver import Request, RoundDriver, cache_buffers

    seen: dict = {}

    def run(fmt):
        d = RoundDriver(
            target,
            dflash_drafter=_draft(target),
            stop_tokens=set(),
            chunk=16,
            kv_format=fmt,
        )
        d.add(Request(PROMPTS[0], 12, handle=0))
        while d.busy():
            d.step()
            for r in d.batch.rows:
                if r.cache and fmt not in seen:
                    seen[fmt] = {
                        "kinds": {type(c).__name__ for c in r.cache},
                        "bytes": sum(a.nbytes for a in cache_buffers(r.cache)),
                    }

    run("int8")
    run("bf16")
    assert (
        "Int8KVCache" in seen["int8"]["kinds"]
        and "Int8KVCache" not in seen["bf16"]["kinds"]
    )
    assert (
        seen["int8"]["bytes"] < 0.9 * seen["bf16"]["bytes"]
    )  # attention layers ~0.53x (GDN state unchanged)
    assert isinstance(Int8KVCache(), Int8KVCache)


def test_int8_solo_equals_concurrent_for_dflash_rows(target):  # noqa: F811
    solo, _ = _tokens(target, _draft(target), PROMPTS[:1], "int8")
    beside, _ = _tokens(target, _draft(target), PROMPTS, "int8")
    assert beside[0] == solo[0]


def test_int8_spec_on_equals_spec_off(target):  # noqa: F811
    on, d = _tokens(target, _draft(target), PROMPTS[:1], "int8")
    off, _ = _tokens(target, None, PROMPTS[:1], "int8")
    assert on[0] == off[0]
    assert d.drafted > 0  # the drafter proposed windows verified against the int8 KV
    on3, _ = _tokens(target, _draft(target), PROMPTS, "int8")
    off3, _ = _tokens(target, None, PROMPTS, "int8")
    assert on3 == off3


def test_int8_apc_hit_equals_miss(target):  # noqa: F811
    from yunshu_engine.apc_manager import YunshuAPCManager

    def manager():
        return YunshuAPCManager(
            num_blocks=64,
            block_size=16,
            overrides={
                "memory_max_gb": 1,
                "checkpoint_entries": 4,
                "checkpoint_interval_tokens": 64,
            },
        )

    prompt = [(7 * i + 11) % 500 for i in range(300)]
    cold, _ = _tokens(target, _draft(target), [prompt], "int8", apc=manager())
    apc = manager()
    first, d = _tokens(target, _draft(target), [prompt], "int8", apc=apc)
    d.flush_checkpoints()
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d2 = RoundDriver(
        target,
        dflash_drafter=_draft(target),
        stop_tokens=set(),
        chunk=16,
        kv_format="int8",
        apc=apc,
    )
    hit = d2.add(Request(prompt, 12, handle=0))
    out = []
    while d2.busy():
        out += [e.token for e in d2.step()]
    assert hit > 0 and first[0] == cold[0] and out == cold[0]
