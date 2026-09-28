"""Exactness + speed of Yunshu verify kernels vs mlx-vlm base (Qwen3.8 shapes)."""

import json
import sys
import time

sys.path.insert(0, "python")
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import quantized_verifier as qv

from yunshu_engine.kernels import verify_select as vs

src = (
    Path("scripts/research/bench_verify_qmv_variants.py")
    .read_text()
    .split("mx.random.seed(0)")[0]
)
ns = {}
exec(src, ns)
run = ns["run"]


def amort(fn, n=40):
    for _ in range(3):
        mx.eval(fn())
    t0 = time.perf_counter()
    o = [fn() for _ in range(n)]
    mx.eval(o)
    return round((time.perf_counter() - t0) / n * 1e6, 1)


mx.random.seed(0)
for bits in (4, 5):
    for name, (kin, nout) in {
        "down": (17408, 5120),
        "up": (5120, 17408),
        "q": (5120, 12288),
        "gdn_out": (6144, 5120),
    }.items():
        lin = nn.QuantizedLinear(kin, nout, bias=False, group_size=64, bits=bits)
        lin.scales = lin.scales.astype(mx.bfloat16)
        lin.biases = lin.biases.astype(mx.bfloat16)
        x = (mx.random.normal((1, 8, kin)) * 0.5).astype(mx.bfloat16)
        ref = mx.concatenate([lin(x[:, t : t + 1]) for t in range(8)], axis=1)
        mx.eval(ref)
        dec = amort(lambda: lin(x[:, :1]))
        row = {"bits": bits, "shape": name, "decode_T1_us": dec}
        for T in (2, 3, 4, 5, 6, 8):
            xt = x[:, :T]
            f = vs.streamed_fixed(qv, lin, xt)
            mx.eval(f)
            exact = bool(mx.array_equal(f, ref[:, :T]).item())
            u1 = vs.unpacked(qv, lin, xt, 1)
            u4 = vs.unpacked(qv, lin, xt, 4)
            mx.eval(u1, u4)
            row[f"T{T}"] = {
                "fixed": amort(lambda: vs.streamed_fixed(qv, lin, xt)),
                "base": amort(lambda: run(lin, xt, "base")),
                "unp1": amort(lambda: vs.unpacked(qv, lin, xt, 1)),
                "unp4": amort(lambda: vs.unpacked(qv, lin, xt, 4)),
                "exact": exact,
                "unp_exact": bool(mx.array_equal(u1, ref[:, :T]).item())
                and bool(mx.array_equal(u4, ref[:, :T]).item()),
            }
        print(json.dumps(row), flush=True)
