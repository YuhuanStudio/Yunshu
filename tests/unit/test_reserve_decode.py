"""tree_verify.reserve_decode: attention caches grow in large steps while decoding."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from yunshu_engine import tree_verify as tv


def test_decode_growth_copies_once_per_reserved_step():
    # mlx-lm grows 256 rows at a time, copying the whole cache each time (1021c: a
    # 25-60 ms round every 256 tokens at 32K)
    mx.random.seed(0)
    plain, reserved = KVCache(), KVCache()
    prompt = mx.random.normal((1, 2, 300, 8)).astype(mx.bfloat16)
    for c in (plain, reserved):
        c.update_and_fetch(prompt, prompt)
    assert tv.reserve_decode([reserved, ArraysCache(2)], 600) == 768
    sizes = {"plain": set(), "reserved": set()}
    for _ in range(40):
        step = mx.random.normal((1, 2, 16, 8)).astype(mx.bfloat16)
        for name, c in (("plain", plain), ("reserved", reserved)):
            c.update_and_fetch(step, step)
            sizes[name].add(int(c.keys.shape[2]))
    # sizes seen = the prefill's allocation plus one per growth (each a full copy)
    assert len(sizes["plain"]) == 3  # 512 -> 768 -> 1024: a copy per 256 rows
    assert len(sizes["reserved"]) == 2  # 512 -> 1276: one growth covers the reply
    n = plain.offset
    assert n == reserved.offset == 940
    for a, b in ((plain.keys, reserved.keys), (plain.values, reserved.values)):
        assert mx.array_equal(a[..., :n, :], b[..., :n, :])


def test_reserve_is_capped_and_skips_other_caches():
    rotating = RotatingKVCache(max_size=64)
    kv = KVCache()
    assert tv.reserve_decode([kv, rotating], 100_000) == tv.DECODE_STEP_CAP
    assert kv.step == tv.DECODE_STEP_CAP
    assert rotating.step == RotatingKVCache.step
    assert tv.reserve_decode([rotating], 10) == 0
