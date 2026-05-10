"""Yunshu Distributed Benchmark — measures collective ops bandwidth.

Runs on a 2+ process ring mesh via launch_mesh.py:
  PYTHONPATH=. uv run python scripts/launch_mesh.py -n 2 -- python scripts/bench_distributed.py

Tests all_reduce, all_gather, sum_scatter at various tensor sizes,
measures latency (avg/p50/p99) and effective bandwidth.
"""
from __future__ import annotations

import argparse
import json
import sys

import mlx.core as mx

from yunshu_mesh.collective import CollectiveOps


def main():
    parser = argparse.ArgumentParser(description="Yunshu distributed benchmark")
    parser.add_argument("--sizes", type=str, default="1k,10k,100k,1m,4m,16m",
                        help="Comma-separated tensor sizes (k=1024, m=1024*1024)")
    parser.add_argument("--iters", type=int, default=50, help="Iterations per size")
    parser.add_argument("--ops", type=str, default="all_reduce,all_gather,sum_scatter",
                        help="Comma-separated ops to benchmark")
    parser.add_argument("--backend", type=str, default="ring", help="Distributed backend")
    args = parser.parse_args()

    def parse_size(s: str) -> int:
        s = s.strip().lower()
        if s.endswith("k"):
            return int(s[:-1]) * 1024
        if s.endswith("m"):
            return int(s[:-1]) * 1024 * 1024
        return int(s)

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    ops = [op.strip() for op in args.ops.split(",")]

    ops_instance = CollectiveOps(backend=args.backend)
    if not ops_instance.initialize(backend=args.backend):
        print("ERROR: Failed to initialize distributed backend")
        print("Run via: launch_mesh.py -n 2 -- python scripts/bench_distributed.py")
        sys.exit(1)

    rank = ops_instance.rank
    world_size = ops_instance.size
    print(f"Distributed benchmark: rank={rank}, world_size={world_size}")
    print(f"Backend: {args.backend}, iters: {args.iters}")
    print("=" * 80)

    results = []
    for op_name in ops:
        for tensor_size in sizes:
            bench = ops_instance.bench_collective(
                op_name=op_name,
                tensor_size=tensor_size,
                num_iters=args.iters,
            )
            results.append(bench)
            print(
                f"  {op_name:15s} size={tensor_size:>10,}  "
                f"avg={bench['avg_latency_ms']:>8.3f}ms  "
                f"p50={bench['p50_ms']:>8.3f}ms  "
                f"p99={bench['p99_ms']:>8.3f}ms  "
                f"BW={bench['bandwidth_gbs']:>6.2f} GB/s"
            )

    print("=" * 80)

    # Quick benchmark (10 iterations, fixed size)
    print("\nQuick benchmark (all ops, 1M elements, 10 iters):")
    quick = ops_instance.run_benchmark(size=1024 * 1024)
    for k, v in quick.items():
        print(f"  {k}: {v}")

    ops_instance.shutdown()


if __name__ == "__main__":
    main()
