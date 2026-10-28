"""Flash-Next expert gather_qmm at prefill shapes: TFLOP/s of MLX's sorted gather vs bits, rows and skew.

Real shapes: hidden 2560, moe_intermediate 640, 512 experts, top10. One chunk = 2048 tokens = 20480
(token, expert) rows sorted by expert. gate/up are [E, 640, 2560], down is [E, 2560, 640].
Eight calls per eval (an eval per call overstates small kernels). Prints JSON lines.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

E, D, I = 512, 2560, 640


def rows_per_expert(rows: int, skew: float, rng) -> np.ndarray:
    """Sorted expert ids for ``rows`` pairs: uniform when skew is 0, Zipf-like otherwise."""
    p = (1.0 / np.arange(1, E + 1)) ** skew
    p = p / p.sum()
    ids = rng.choice(E, size=rows, p=p)
    return np.sort(ids).astype(np.uint32)


def bench(kind: str, bits: int, rows: int, skew: float, repeat: int = 8) -> dict:
    rng = np.random.default_rng(0)
    n, k = (I, D) if kind == "gate" else (D, I)
    w = mx.random.normal((E, n, k)).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=bits)
    ids = rows_per_expert(rows, skew, rng)
    touched = len(set(ids.tolist()))
    idx = mx.array(ids)
    x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
    mx.eval(wq, scales, biases, idx, x)

    def call():
        return mx.gather_qmm(
            x,
            wq,
            scales,
            biases,
            rhs_indices=idx,
            transpose=True,
            group_size=64,
            bits=bits,
            sorted_indices=True,
        )

    mx.eval(call())  # compile
    best = None
    for _ in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval([call() for _ in range(repeat)])
        dt = (time.perf_counter() - t0) / repeat
        best = dt if best is None else min(best, dt)
    flops = 2.0 * rows * n * k
    return dict(
        kind=kind,
        bits=bits,
        rows=rows,
        skew=skew,
        experts_touched=touched,
        rows_per_expert=round(rows / touched, 1),
        ms=round(best * 1e3, 3),
        tflops=round(flops / best / 1e12, 1),
        weight_gb_s=round(touched * n * k * bits / 8 / best / 1e9, 1),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cells = []
    for kind in ("gate", "down"):
        for bits in (2, 3, 4, 8):
            for rows in (20480,):
                for skew in (0.0, 0.5):
                    cells.append((kind, bits, rows, skew))
    for rows in (5120, 10240, 40960, 81920):  # tile fill vs rows per expert
        cells.append(("gate", 2, rows, 0.0))
    with open(a.out, "w") as fh:
        for cell in cells:
            row = bench(*cell)
            fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        fh.write(json.dumps(dict(complete=True)) + "\n")


if __name__ == "__main__":
    main()
