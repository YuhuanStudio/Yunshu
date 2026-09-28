"""Stock QuantizedLinear vs tensor-unit projections per shape, 4- and 5-bit.

Random weights at Qwen3.8-27B projection shapes; times one call at each row
count (median of ``--iters``). Paths (``--paths``):

- ``stock``: MLX ``QuantizedLinear`` (qmv / qmm)
- ``packed``: our NAX ``PackedLinear`` (4-bit: oMLX kernels; 5-bit: the
  dequantize-to-bf16 GEMM, ``YUNSHU_PACKED_5BIT=1``)
- ``int``: TensorFold's integer-code lane matmul (5/6/8-bit, MLX layout),
  ``int_tiled`` the same on 32-column tiled codes (``YUNSHU_PACKED_5BIT=int``)

    python scripts/research/bench_packed_5bit.py --bits 5 --rows 1 2 4 8 16 \
        --shapes qkv6144 mlp17408 down5120 --paths stock packed int int_tiled \
        --output runs/packed-5bit.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from yunshu_engine.kernels.int_code_linear import IntCodeLinear  # noqa: E402
from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl  # noqa: E402

# (name, K, N) for Qwen3.8-27B (hidden 5120, MLP 17408).
SHAPES = [
    ("mlp_gate_up", 5120, 34816),
    ("mlp_down", 17408, 5120),
    ("attn_q", 5120, 12288),
    ("attn_o", 6144, 5120),
    ("gdn_qkv", 5120, 10240),
    # Shapes named in the 5-bit comparison request.
    ("qkv6144", 5120, 6144),
    ("mlp17408", 5120, 17408),
    ("down5120", 17408, 5120),
]


def make(K, N, bits):
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    q, s, b = mx.quantize(mx.random.normal((N, K)) * 0.02, group_size=64, bits=bits)
    lin.weight, lin.scales = q, s.astype(mx.bfloat16)
    lin.biases = b.astype(mx.bfloat16)
    mx.eval(lin.parameters())
    return lin


def timeit(fn, x, iters):
    for _ in range(3):
        mx.eval(fn(x))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn(x))
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rows", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--bits", type=int, nargs="*", default=[4, 5])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--shapes", nargs="*", help="subset of shape names")
    ap.add_argument(
        "--paths",
        nargs="*",
        default=["stock", "packed"],
        choices=["stock", "packed", "int", "int_tiled"],
    )
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    out = a.output.open("a") if a.output else None
    for name, K, N in SHAPES:
        if a.shapes and name not in a.shapes:
            continue
        for bits in a.bits:
            lin = make(K, N, bits)
            impls = {"stock": lin}
            if "packed" in a.paths:
                saved = pl.PACKED_BITS
                pl.PACKED_BITS = (4, 5)
                try:
                    (impls["packed"],) = pl._pack([lin])
                finally:
                    pl.PACKED_BITS = saved
            if bits != 4 and "int" in a.paths:
                impls["int"] = IntCodeLinear(lin)
            if bits != 4 and "int_tiled" in a.paths:
                impls["int_tiled"] = IntCodeLinear(lin, tiled=True)
            mx.eval([m.parameters() for m in impls.values()])
            for rows in a.rows:
                x = (mx.random.normal((rows, K)) * 0.5).astype(mx.bfloat16)
                us = {
                    path: round(timeit(impls[path], x, a.iters) * 1e6, 1)
                    for path in a.paths
                    if path in impls
                }
                gb = N * K * bits / 8 / 1e9
                best = min(us, key=us.get)
                row = {
                    "shape": name,
                    "K": K,
                    "N": N,
                    "bits": bits,
                    "rows": rows,
                    **{f"{p}_us": v for p, v in us.items()},
                    "best": best,
                    "best_gbps": round(gb / (us[best] / 1e6), 1),
                }
                print(json.dumps(row), flush=True)
                if out:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            del lin, impls
            mx.clear_cache()


if __name__ == "__main__":
    main()
