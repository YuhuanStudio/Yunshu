"""Is oMLX's NAX packed 4-bit matmul row-invariant across 1..8 rows, and its cost vs decode?"""

import json
import sys
import time

sys.path.insert(0, "python")
import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl


def amort(fn, n=40):
    for _ in range(3):
        mx.eval(fn())
    t0 = time.perf_counter()
    o = [fn() for _ in range(n)]
    mx.eval(o)
    return round((time.perf_counter() - t0) / n * 1e6, 1)


mx.random.seed(2)
for name, (K, N) in {
    "down": (17408, 5120),
    "up": (5120, 17408),
    "q": (5120, 12288),
    "gdn_out": (6144, 5120),
}.items():
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    lin.scales = lin.scales.astype(mx.bfloat16)
    lin.biases = lin.biases.astype(mx.bfloat16)
    assert pl.eligible(lin), name
    stock_us = amort(lambda: lin(mx.zeros((1, 1, K), dtype=mx.bfloat16)))
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
    stock = mx.concatenate([lin(x[t : t + 1]) for t in range(8)], axis=0)
    p = pl._pack([lin])[0]
    outs = {M: p(x[:M]) for M in range(1, 9)}
    mx.eval(list(outs.values()))
    ref = outs[8]
    inv = {M: bool(mx.array_equal(outs[M], ref[:M]).item()) for M in outs}
    err1 = (
        mx.abs(outs[1].astype(mx.float32) - stock[:1].astype(mx.float32)).max()
        / mx.abs(stock[:1].astype(mx.float32)).max()
    ).item()
    us = {M: amort(lambda M=M: p(x[:M])) for M in (1, 2, 4, 6, 8)}
    print(
        json.dumps(
            {
                "shape": name,
                "stock_decode_us": stock_us,
                "packed_us": us,
                "row_invariant_vs_M8": inv,
                "M1_equals_stock": bool(mx.array_equal(outs[1], stock[:1]).item()),
                "M1_rel_err_vs_stock": err1,
            }
        ),
        flush=True,
    )
