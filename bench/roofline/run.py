"""Apple GPU Roofline Benchmark — GEMM throughput vs matrix size.

Measures the compute roofline for Apple Silicon GPU using MLX GEMM.
This establishes the theoretical maximum throughput for the inference engine.

Run: PYTHONPATH=. uv run python bench/roofline/run.py
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def bench_gemm(size: int, dtype: mx.Dtype, num_iters: int) -> dict:
    """Benchmark a single GEMM size."""
    a = mx.random.normal((size, size), dtype=dtype)
    b = mx.random.normal((size, size), dtype=dtype)
    # Warmup
    _ = a @ b
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        c = a @ b
    mx.synchronize()
    elapsed = time.perf_counter() - t0

    flops = 2.0 * size ** 3 * num_iters
    return {
        "size": size,
        "flops": flops,
        "elapsed_s": elapsed,
        "tflops": flops / elapsed / 1e12,
        "time_ms": elapsed / num_iters * 1000,
    }


def run_roofline(
    dtype: str = "float16",
    min_size: int = 64,
    max_size: int = 8192,
    steps: int = 10,
    output_dir: str = "bench/roofline",
):
    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    mx_dtype = dtype_map[dtype]

    sizes = np.logspace(
        np.log10(min_size), np.log10(max_size), steps, dtype=int
    ).tolist()

    print(f"=== Apple GPU Roofline Benchmark ===")
    print(f"dtype: {dtype}, sizes: {len(sizes)}")
    print(f"{'Size':>8} {'TFLOPS':>10} {'Time (ms)':>10}")
    print("-" * 32)

    results = []
    for size in sizes:
        num_iters = max(1, min(200, 2**24 // (size * size)))
        result = bench_gemm(size, mx_dtype, num_iters)
        results.append(result)
        print(f"{size:8d} {result['tflops']:10.2f} {result['time_ms']:10.3f}")

    # Find peak
    peak = max(results, key=lambda r: r["tflops"])
    print(f"\nPeak: {peak['tflops']:.2f} TFLOPS at M=N=K={peak['size']}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "device": "Apple GPU",
        "dtype": dtype,
        "peak_tflops": peak["tflops"],
        "peak_size": peak["size"],
        "results": results,
    }
    with open(output_path / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path / 'report.json'}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--max-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    run_roofline(args.dtype, args.min_size, args.max_size, args.steps)
