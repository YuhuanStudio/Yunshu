"""Per-kernel latency floor of this GPU/MLX build for the op shapes of a Flash-Next decode step.

A decode step is ~3000 small dependent kernels (profile graph census).  If one dependent tiny kernel costs
~7 us, the step is launch/dependency bound and only fewer kernels help.  Measures, each as the median of
``--reps`` evals after warm-up:

  chain_mul       K dependent tiny elementwise ops on [1, 2560] bf16
  chain_compiled  the same chain under mx.compile (one fused kernel)
  chain_sum       K dependent reductions (sum -> broadcast) -- the RMSNorm pattern
  chain_qmm       K dependent 4-bit g64 quantized matmuls [1,2560] x [2560,2560]
  indep_mul       K independent tiny ops reduced at the end

Run through gpuq only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def per_op_us(total_s: float, k: int) -> float:
    return total_s / k * 1e6


def median(values):
    return statistics.median(values)


def timed(fn, reps):
    import mlx.core as mx

    mx.eval(fn())  # warm-up (kernel compile)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        out.append(time.perf_counter() - t0)
    return median(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import mlx.core as mx

    x = mx.random.normal((1, 2560)).astype(mx.bfloat16)
    mx.eval(x)
    rows = {}

    def chain_mul():
        y = x
        for _ in range(a.k):
            y = y * 1.0009765625
        return y

    ops = lambda v: ((v * 1.0009765625 + 0.0001) * v) - v * 0.5  # noqa: E731
    fused = mx.compile(ops)

    def chain_compiled():
        y = x
        for _ in range(a.k // 4):
            y = fused(y)
        return y

    def chain_sum():
        y = x
        for _ in range(a.k):
            y = y / (mx.sum(y * y, axis=-1, keepdims=True) + 1.0).astype(mx.bfloat16)
        return y

    w = mx.random.normal((2560, 2560)).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(w, group_size=64, bits=4)
    mx.eval(qw, qs, qb)

    def chain_qmm():
        y = x
        for _ in range(a.k // 4):
            y = mx.quantized_matmul(
                y, qw, qs, qb, transpose=True, group_size=64, bits=4
            )
            y = y * 0.5
        return y

    def indep_mul():
        ys = [x * (1.0 + i * 1e-3) for i in range(a.k)]
        return mx.stack(ys).sum(axis=0)

    for name, fn, n in (
        ("chain_mul", chain_mul, a.k),
        ("chain_compiled", chain_compiled, a.k // 4),
        ("chain_sum", chain_sum, a.k * 3),
        ("chain_qmm", chain_qmm, a.k // 4 * 2),
        ("indep_mul", indep_mul, a.k),
    ):
        total = timed(fn, a.reps)
        rows[name] = {
            "total_ms": round(total * 1e3, 3),
            "kernels": n,
            "us_per_kernel": round(per_op_us(total, n), 2),
        }
        print(json.dumps({name: rows[name]}), flush=True)
    a.out.write_text(
        json.dumps({"kind": "kernel_latency", **rows, "complete": True}) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
