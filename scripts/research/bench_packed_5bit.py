"""Stock QuantizedLinear vs NAX PackedLinear per projection, 4- and 5-bit.

Random weights at Qwen3.8-27B projection shapes; times one call at each row
count (median of ``--iters``) so the 5-bit packed kernels can be judged
against MLX's own quantized matmul before a full-model run.

    python scripts/research/bench_packed_5bit.py --rows 1 2 4 8 16 \
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

from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl  # noqa: E402

# (name, K, N) for Qwen3.8-27B (hidden 5120, MLP 17408).
SHAPES = [
    ("mlp_gate_up", 5120, 34816),
    ("mlp_down", 17408, 5120),
    ("attn_q", 5120, 12288),
    ("attn_o", 6144, 5120),
    ("gdn_qkv", 5120, 10240),
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
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    out = a.output.open("a") if a.output else None
    for name, K, N in SHAPES:
        if a.shapes and name not in a.shapes:
            continue
        for bits in a.bits:
            lin = make(K, N, bits)
            (packed,) = pl._pack([lin])
            for rows in a.rows:
                x = (mx.random.normal((rows, K)) * 0.5).astype(mx.bfloat16)
                stock = timeit(lin, x, a.iters)
                pk = timeit(packed, x, a.iters)
                gb = N * K * bits / 8 / 1e9
                row = {
                    "shape": name,
                    "K": K,
                    "N": N,
                    "bits": bits,
                    "rows": rows,
                    "stock_us": round(stock * 1e6, 1),
                    "packed_us": round(pk * 1e6, 1),
                    "speedup": round(stock / pk, 2),
                    "packed_gbps": round(gb / pk, 1),
                }
                print(json.dumps(row), flush=True)
                if out:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            del lin, packed
            mx.clear_cache()


if __name__ == "__main__":
    main()
