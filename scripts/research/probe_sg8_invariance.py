"""Is oMLX's sg8 verify matmul row-invariant (batch-invariant) and how fast?"""

import json
import sys
import time

sys.path.insert(0, "python")
import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.kernels.omlx import qwen35_verify_qmm as q


def amort(fn, n=40):
    for _ in range(3):
        mx.eval(fn())
    t0 = time.perf_counter()
    o = [fn() for _ in range(n)]
    mx.eval(o)
    return round((time.perf_counter() - t0) / n * 1e6, 1)


mx.random.seed(1)
for bits in (4, 5):
    for name, (K, N) in {
        "down": (17408, 5120),
        "up": (5120, 17408),
        "gdn_out": (6144, 5120),
    }.items():
        lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
        lin.scales = lin.scales.astype(mx.bfloat16)
        lin.biases = lin.biases.astype(mx.bfloat16)
        x = (mx.random.normal((8, K)) * 0.5).astype(mx.bfloat16)
        outs = {}
        for M in range(1, 9):
            ok = q.sg8_eligible(M, K, N, bits, 64, x.dtype)
            if not ok:
                continue
            y = q.vk_qmm_sg8(
                x[:M], lin.weight, lin.scales, lin.biases, group_size=64, bits=bits
            )
            mx.eval(y)
            outs[M] = y
        if not outs:
            print(json.dumps({"bits": bits, "shape": name, "sg8": "not eligible"}))
            continue
        Ms = sorted(outs)
        ref = outs[Ms[-1]]
        inv = {M: bool(mx.array_equal(outs[M], ref[:M]).item()) for M in Ms}
        stock = mx.concatenate([lin(x[t : t + 1][None])[0] for t in range(8)], axis=0)
        vs_stock = bool(mx.array_equal(ref, stock[: Ms[-1]]).item())
        # padded: compute M=1 as a zero-padded 8-row call
        pad = mx.concatenate([x[:1], mx.zeros((7, K), dtype=x.dtype)], axis=0)
        p = q.vk_qmm_sg8(
            pad, lin.weight, lin.scales, lin.biases, group_size=64, bits=bits
        )[:1]
        t_sg8 = {
            M: amort(
                lambda M=M: q.vk_qmm_sg8(
                    x[:M], lin.weight, lin.scales, lin.biases, group_size=64, bits=bits
                )
            )
            for M in Ms
        }
        print(
            json.dumps(
                {
                    "bits": bits,
                    "shape": name,
                    "eligible_M": Ms,
                    "row_invariant": inv,
                    "padded_row0_equals": bool(mx.array_equal(p, ref[:1]).item()),
                    "equals_stock_decode": vs_stock,
                    "sg8_us": t_sg8,
                    "stock_decode_us": amort(lambda: lin(x[:1][None])),
                }
            ),
            flush=True,
        )
