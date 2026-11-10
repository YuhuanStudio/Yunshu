"""The int8 tile kernel (kernels/int8_tile.py) equals the bf16 tile kernel on the dequantized cache, bit for
bit: PO, PM and PL of every partial, across several 512-key chunks, the last step clamped to the buffer end,
and 1-8 fused tokens."""

import pytest

pytestmark = pytest.mark.m5_gpu

mx = pytest.importorskip("mlx.core")


def _int8_cache(hkv, cap, seed):
    from yunshu_engine.kernels.ragged_attention import quantize_kv_pair

    mx.random.seed(seed)
    k = mx.random.normal((1, hkv, cap, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, cap, 256)).astype(mx.bfloat16)
    return quantize_kv_pair(k, v)


@pytest.mark.parametrize(
    ("hkv", "g", "tokens", "length"),
    [(1, 4, 1, 1300), (2, 4, 4, 1300), (4, 6, 8, 1601), (4, 6, 3, 512), (1, 4, 2, 65)],
)
def test_int8_tile_partials_equal_the_reference_bit_for_bit(hkv, g, tokens, length):
    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import int8_tile
    from yunshu_engine.kernels import ragged_attention as ra

    cap = -(-length // 64) * 64  # the last 64-key step is clamped to the buffer end
    kc, ks, vc, vs = _int8_cache(hkv, cap, seed=length + tokens)
    mx.random.seed(7)
    q = mx.random.normal((1, hkv * g, tokens, 256)).astype(mx.bfloat16)
    q_fused = tv._fuse_tokens(q, hkv, g, tokens)
    nc = -(-length // tv.CK)
    plan = ra._work_list((length,), tokens, nc, True)
    lengths = mx.array([length], dtype=mx.int32)
    slots = mx.array([0], dtype=mx.int32)
    scale = tv._scale_array(256**-0.5)
    got = int8_tile.tile_partials_q8(
        q_fused, kc, ks, vc, vs, lengths, slots, scale, plan, tokens, nc
    )
    ref = tv.tile_partials_q8_reference(
        q_fused, kc, ks, vc, vs, lengths, slots, scale, plan, tokens, nc
    )
    for a, b in zip(got[:3], ref[:3], strict=True):
        assert bool(mx.array_equal(a, b))


def test_tree_attention_with_the_int8_tile_kernel_equals_the_reference(monkeypatch):
    """Through tree_attention with a prefix long enough for shared chunks (n0 >= 512)."""
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import int8_tile
    from yunshu_engine.kernels.int8_kv import Int8KVCache

    monkeypatch.setattr(
        tv, "TREE_ATTENTION_FUSED_Q8", tv.tree_attention_fused_q8_reference
    )
    width, n0 = 4, 1300
    mx.random.seed(3)
    k = mx.random.normal((1, 1, n0 + width, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 1, n0 + width, 256)).astype(mx.bfloat16)
    c = KVCache()
    c.update_and_fetch(k, v)
    q8 = Int8KVCache.from_kv(c)
    queries = mx.random.normal((1, 4, width, 256)).astype(mx.bfloat16)
    shape = tv.TreeShape(list(range(-1, width - 1)))
    scale = 256**-0.5
    monkeypatch.setattr(tv, "TILE_PARTIALS_Q8", tv.tile_partials_q8_reference)
    ref = tv.tree_attention(queries, q8, scale, shape, n0)
    monkeypatch.setattr(tv, "TILE_PARTIALS_Q8", int8_tile.tile_partials_q8)
    got = tv.tree_attention(queries, q8, scale, shape, n0)
    assert bool(mx.array_equal(got, ref))


def test_the_tile_rewrite_still_matches_the_bf16_kernel_source():
    from yunshu_engine.kernels import int8_tile

    src = int8_tile._source()
    assert "kq[e] = bfloat16_t(float(keys[src]) * float(k_scales[sidx]))" in src
    assert "auto ks = tensor(kq" in src and "auto vs = tensor(vq" in src


@pytest.mark.parametrize(("width", "n0"), [(4, 1300), (9, 1300), (8, 700), (3, 520)])
def test_fused_int8_equals_the_fused_reference_bit_for_bit(width, n0):
    """The fused path (one shared-prefix launch, tail copy from a dequantized slice) against the fused
    reference on the whole dequantized buffer."""
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import int8_tile
    from yunshu_engine.kernels.int8_kv import Int8KVCache

    mx.random.seed(width + n0)
    k = mx.random.normal((1, 1, n0 + width, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 1, n0 + width, 256)).astype(mx.bfloat16)
    c = KVCache()
    c.update_and_fetch(k, v)
    q8 = Int8KVCache.from_kv(c)
    queries = mx.random.normal((1, 4, width, 256)).astype(mx.bfloat16)
    shape = tv.TreeShape(list(range(-1, width - 1)))
    rc = tv.RoundContext(shape, n0)
    scale_arr = tv._scale_array(256**-0.5)
    dims = (4, 1, 256)
    keys, ks, values, vs = q8.keys, q8.k_scales, q8.values, q8.v_scales
    ref = tv.tree_attention_fused_q8_reference(
        queries, keys, ks, values, vs, shape, rc, scale_arr, dims
    )
    got = int8_tile.tree_attention_fused_q8(
        queries, keys, ks, values, vs, shape, rc, scale_arr, dims
    )
    assert bool(mx.array_equal(got, ref))
