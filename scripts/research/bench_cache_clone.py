"""Cost of cloning a 32K-token hybrid cache (the APC restore / checkpoint copy), by allocator state.

16 attention layers x (K, V) of [1, 4, T, 256] bf16 + 48 GDN states: concatenate-with-padding clone as
mlx-vlm's ``_clone_prompt_cache_for_apc`` does, with the MLX buffer cache cleared before each clone vs left warm,
and with a raised cache limit.
"""

import json
import time

import mlx.core as mx

T = 32777
layers = [
    (
        mx.random.normal((1, 4, T, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, T, 256)).astype(mx.bfloat16),
    )
    for _ in range(16)
]
states = [mx.random.normal((1, 48, 128, 128)) for _ in range(48)]
mx.eval(layers, states)


def clone(pad=256):
    out = []
    for k, v in layers:
        z = (1, 4, pad, 256)
        out.append(
            (
                mx.concatenate([k, mx.zeros(z, dtype=k.dtype)], axis=2),
                mx.concatenate([v, mx.zeros(z, dtype=v.dtype)], axis=2),
            )
        )
    st = [mx.contiguous(mx.array(s)) for s in states]
    mx.eval(out, st)
    return out, st


def timed(label, clear, reps=5):
    ts = []
    for _ in range(reps):
        if clear:
            mx.clear_cache()
        mx.synchronize()
        t = time.perf_counter()
        o = clone()
        ts.append(round((time.perf_counter() - t) * 1e3, 1))
        del o
    return label, ts


res = [
    timed("cache cleared before each clone", True),
    timed("warm buffer cache", False),
]
old = mx.set_cache_limit(16 << 30)
res.append(timed("cache limit 16 GiB, warm", False))
res.append(timed("cache limit 16 GiB, cleared", True))
mx.set_cache_limit(old)
print(json.dumps(dict(res)))
