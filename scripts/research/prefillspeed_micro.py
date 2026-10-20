"""Prefill SDPA / dense-GEMM roofline microbench (GPU jobs go through gpuq).

    prefillspeed_micro.py --mode sdpa|peak --out F.jsonl [--keys 4096 ...] [--dry-run]

``sdpa``: stock ``mx.fast.scaled_dot_product_attention`` for one 2048-row prefill chunk of the 27B
          full-attention layer (24 q heads, 4 kv heads, D=256, bf16, causal) at several key lengths.
``peak``: dense bf16 matmul throughput at large shapes (the NAX compute ceiling).
Fails closed: any exception exits nonzero; the last record is ``{"type":"complete"}`` only on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

H, HKV, D, CHUNK = 24, 4, 256, 2048


def sdpa_flops(q_rows: int, keys: int, heads: int = H, d: int = D) -> float:
    """Causal QK+PV FLOPs for ``q_rows`` queries ending at key ``keys`` (offset = keys - q_rows)."""
    off = keys - q_rows
    pairs = q_rows * off + q_rows * (q_rows + 1) / 2
    return 4.0 * pairs * heads * d


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sdpa", "peak"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--keys", type=int, nargs="+", default=[4096, 16384, 32768, 65536, 131072]
    )
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    return p


def timed(mx, fn, n: int) -> float:
    for _ in range(2):
        mx.eval(fn())
    t0 = time.perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / n


def run_sdpa(mx, keys: list[int], reps: int, emit) -> None:
    mx.random.seed(0)
    for lk in keys:
        q = (mx.random.normal((1, H, CHUNK, D)) * 0.5).astype(mx.bfloat16)
        k = (mx.random.normal((1, HKV, lk, D)) * 0.5).astype(mx.bfloat16)
        v = (mx.random.normal((1, HKV, lk, D)) * 0.5).astype(mx.bfloat16)
        mx.eval(q, k, v)
        n = max(2, min(reps, int(2e12 // max(sdpa_flops(CHUNK, lk), 1))))
        dt = timed(
            mx,
            lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=D**-0.5, mask="causal"
            ),
            n,
        )
        fl = sdpa_flops(CHUNK, lk)
        emit(
            dict(
                type="sdpa",
                keys=lk,
                ms=round(dt * 1e3, 3),
                tflops=round(fl / dt / 1e12, 2),
                kv_gbps=round(2 * HKV * lk * D * 2 / dt / 1e9, 1),
            )
        )


def run_peak(mx, reps: int, emit) -> None:
    for m, k, n in (
        (2048, 5120, 17408),
        (2048, 17408, 5120),
        (4096, 8192, 8192),
        (8192, 8192, 8192),
    ):
        a = mx.random.normal((m, k)).astype(mx.bfloat16)
        w = mx.random.normal((k, n)).astype(mx.bfloat16)
        mx.eval(a, w)
        dt = timed(mx, lambda: a @ w, reps * 2)
        emit(
            dict(type="gemm", m=m, k=k, n=n, tflops=round(2 * m * k * n / dt / 1e12, 2))
        )


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        print(
            json.dumps(
                dict(mode=a.mode, keys=a.keys, flops=sdpa_flops(CHUNK, a.keys[0]))
            )
        )
        return 0
    import mlx.core as mx

    n = 0
    with a.out.open("w") as f:

        def emit(rec: dict) -> None:
            nonlocal n
            n += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)

        emit(
            dict(
                type="engagement",
                device=mx.device_info().get("device_name"),
                mlx=mx.__version__,
            )
        )
        (
            run_sdpa(mx, a.keys, a.reps, emit)
            if a.mode == "sdpa"
            else run_peak(mx, a.reps, emit)
        )
        if n < 2:
            return 1
        f.write(json.dumps(dict(type="complete", success=True, records=n)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
