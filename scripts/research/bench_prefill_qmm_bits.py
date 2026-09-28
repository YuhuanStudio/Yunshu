"""Prefill-shaped quantized matmul cost by bit width on this GPU (NAX coverage check).

Times ``nn.QuantizedLinear`` on (M, K) activations for Qwen3.8 projection shapes
at 4/5/6/8 bits, plus bf16 dense, reporting effective TFLOPS. If a width is much
slower than 4-bit after scaling for bytes, stock MLX likely has no tensor-unit
(NAX) GEMM for it and mixed-precision checkpoints pay that in cold prefill.
"""

import json
import time

import mlx.core as mx
import mlx.nn as nn

M = 2048
SHAPES = {"down": (17408, 5120), "up": (5120, 17408), "gdn_out": (6144, 5120)}


def amort(fn, n=10):
    for _ in range(2):
        mx.eval(fn())
    t0 = time.perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / n


mx.random.seed(0)
for name, (k, n) in SHAPES.items():
    x = (mx.random.normal((1, M, k)) * 0.5).astype(mx.bfloat16)
    flops = 2 * M * k * n
    dense = nn.Linear(k, n, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    row = {"shape": name, "M": M, "bf16_tflops": round(flops / amort(lambda: dense(x)) / 1e12, 1)}
    for bits in (4, 5, 6, 8):
        lin = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
        lin.scales = lin.scales.astype(mx.bfloat16)
        lin.biases = lin.biases.astype(mx.bfloat16)
        row[f"q{bits}_tflops"] = round(flops / amort(lambda lin=lin: lin(x)) / 1e12, 1)
    print(json.dumps(row), flush=True)
