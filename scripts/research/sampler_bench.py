"""Cost of the keyed sampler on a verify window: [N, 248320] fp32 rows, per component."""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/python")
from yunshu_engine import keyed_sampling as ks  # noqa: E402

V = 248320


def bench(fn, n=20):
    mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n * 1e3


class P:
    def __init__(self, t, p, k=0, m=0.0):
        self.temperature, self.top_p, self.top_k, self.min_p = t, p, k, m


for rows in (1, 8, 16):
    z = mx.random.normal((rows, V)) * 3
    x = z - mx.logsumexp(z, axis=-1, keepdims=True)
    mx.eval(x)
    pos = mx.arange(rows)
    bases = mx.array(np.full((rows,), ks.seed_base(1234), dtype=np.uint64))
    res = {
        "argmax": bench(lambda: mx.argmax(x, axis=-1)),
        "top_p": bench(lambda: ks.top_p_filter(x, 0.95)),
        "argsort_only": bench(lambda: mx.argsort(-x, axis=-1)),
        "gumbel": bench(lambda: ks.gumbel_rows(bases, pos, V)),
        "sample_p95": bench(lambda: ks.sample_rows(x, P(0.7, 0.95), bases, pos)),
        "sample_p1": bench(lambda: ks.sample_rows(x, P(0.7, 1.0), bases, pos)),
    }
    K = 1024

    def cand_topp():
        lse = mx.logsumexp(x, axis=-1, keepdims=True)
        idx = mx.argpartition(-x, K - 1, axis=-1)[:, :K]
        v = mx.take_along_axis(x, idx, axis=-1)
        o = mx.argsort(-v, axis=-1)
        idx, v = mx.take_along_axis(idx, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
        pr = mx.exp(v - lse)
        before = mx.cumsum(pr, axis=-1) - pr
        keep = (before < 0.95) | (mx.arange(K) == 0)
        h = ks._u64(ks.seed_base(1234))
        pos2 = pos.astype(mx.uint64).reshape(-1, 1)
        g = ks._mix(ks._mix(h ^ (pos2 * ks._u64(ks._C3))) ^ idx.astype(mx.uint64))
        u = ((g >> ks._u64(41)).astype(mx.float32) + 0.5) * (1.0 / (1 << 23))
        sc = mx.where(keep, v / 0.7 - mx.log(-mx.log(u)), -mx.inf)
        pick = mx.argmax(sc, axis=-1, keepdims=True)
        return mx.take_along_axis(idx, pick, axis=-1), before[:, -1]

    res["cand_p95"] = bench(cand_topp)
    res["argpartition"] = bench(lambda: mx.argpartition(-x, K - 1, axis=-1))
    res["topk_vals"] = bench(lambda: mx.topk(x, K, axis=-1))
    res["lse"] = bench(lambda: mx.logsumexp(x, axis=-1))
    print(rows, {k: round(v, 2) for k, v in res.items()}, flush=True)
