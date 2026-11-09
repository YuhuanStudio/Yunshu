"""A standing int8 reply cache handed to its follow-up prefills on dequantized bf16, not raw codes."""

import mlx.core as mx
from mlx_vlm.models.cache import KVCache

from yunshu_engine.kernels.int8_kv import Int8KVCache, to_bf16_cache


def _int8_cache(n=100):
    mx.random.seed(3)
    c = KVCache()
    c.update_and_fetch(
        mx.random.normal((1, 2, n, 64)).astype(mx.bfloat16),
        mx.random.normal((1, 2, n, 64)).astype(mx.bfloat16),
    )
    return Int8KVCache.from_kv(c)


def test_standing_cache_converts_to_bf16_and_requantizes_identically():
    q = _int8_cache()
    b = to_bf16_cache(q)
    assert type(b) is KVCache and b.offset == q.offset
    k, v = q.dequantized(q.offset)
    assert mx.array_equal(b.keys, k) and mx.array_equal(b.values, v)
    # the suffix prefill sees real values, not int8 codes
    nk = mx.ones((1, 2, 5, 64), dtype=mx.bfloat16)
    got_k, _ = b.update_and_fetch(nk, nk)
    assert got_k.dtype == mx.bfloat16 and got_k.shape[2] == q.offset + 5
    # joining again quantizes the restored prefix to the very same codes and scales
    r = Int8KVCache.from_kv(to_bf16_cache(q))
    n = q.offset
    assert mx.array_equal(r.keys[..., :n, :], q.keys[..., :n, :])
    assert mx.array_equal(r.k_scales[..., :n, :], q.k_scales[..., :n, :])
    assert mx.array_equal(r.values[..., :n, :], q.values[..., :n, :])
    assert mx.array_equal(r.v_scales[..., :n, :], q.v_scales[..., :n, :])


def test_followup_suffix_attention_matches_dequantized_reference():
    """The suffix prefill's attention over a restored int8 cache equals attention over the
    dequantized prefix (the values decode reads); on the raw cache it attended int8 codes."""
    q = _int8_cache()
    n = q.offset
    sk = mx.random.normal((1, 2, 7, 64)).astype(mx.bfloat16)
    sv = mx.random.normal((1, 2, 7, 64)).astype(mx.bfloat16)
    qq = mx.random.normal((1, 2, 7, 64)).astype(mx.bfloat16)
    pk, pv = q.dequantized(n)
    ref = mx.fast.scaled_dot_product_attention(
        qq,
        mx.concatenate([pk, sk], 2),
        mx.concatenate([pv, sv], 2),
        scale=0.125,
        mask=None,
    )
    b = to_bf16_cache(q)
    fk, fv = b.update_and_fetch(sk, sv)
    got = mx.fast.scaled_dot_product_attention(qq, fk, fv, scale=0.125, mask=None)
    assert mx.array_equal(got, ref)
    # the old path: the raw cache returns int8 codes, which are not the values attention needs
    raw = _int8_cache()
    rk, _ = raw.update_and_fetch(sk, sv)
    assert rk.dtype == mx.int8
