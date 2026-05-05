"""KIVI Metal Kernel Benchmark.

Benchmarks the KIVI 2-bit quantization for KV cache compression.

Run: PYTHONPATH=. uv run python bench/kivi_metal/run.py
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def bench_kivi(num_tokens: int, num_heads: int, head_dim: int, num_iters: int) -> dict:
    """Benchmark KIVI quantize/dequantize."""
    from python.yunshu_engine.metal_kernels import MetalKernelManager

    mgr = MetalKernelManager()
    keys = mx.random.normal((num_tokens, num_heads, head_dim)).astype(mx.float16)

    # Warmup
    for _ in range(3):
        qk, s, zp = mgr.kivi_quantize(keys)
        dq = mgr.kivi_dequantize(qk, s, zp, head_dim)
    mx.synchronize()

    # Benchmark quantize
    t0 = time.perf_counter()
    for _ in range(num_iters):
        qk, s, zp = mgr.kivi_quantize(keys)
    mx.synchronize()
    quant_time = (time.perf_counter() - t0) / num_iters

    # Benchmark dequantize
    qk, s, zp = mgr.kivi_quantize(keys)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        dq = mgr.kivi_dequantize(qk, s, zp, head_dim)
    mx.synchronize()
    dequant_time = (time.perf_counter() - t0) / num_iters

    original_bytes = keys.size * 2
    compressed_bytes = qk.size + s.size * 2 + zp.size * 2

    return {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "quant_time_ms": quant_time * 1000,
        "dequant_time_ms": dequant_time * 1000,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": original_bytes / compressed_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="bench/kivi_metal")
    args = parser.parse_args()

    print("=== KIVI 2-bit Quantization Benchmark ===\n")

    configs = [
        (64, 32, 128),   # Small
        (256, 32, 128),  # Medium
        (1024, 32, 128), # Large
        (4096, 32, 128), # XL
    ]

    results = []
    for num_tokens, num_heads, head_dim in configs:
        result = bench_kivi(num_tokens, num_heads, head_dim, num_iters=20)
        results.append(result)
        print(
            f"  tokens={num_tokens:5d}: "
            f"quant={result['quant_time_ms']:.2f}ms, "
            f"dequant={result['dequant_time_ms']:.2f}ms, "
            f"ratio={result['compression_ratio']:.1f}x"
        )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "report.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nResults saved to {output_path / 'report.json'}")


if __name__ == "__main__":
    main()
