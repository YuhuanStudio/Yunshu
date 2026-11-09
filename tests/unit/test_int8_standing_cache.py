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
