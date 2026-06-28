#!/usr/bin/env python3
"""JACCL baseline benchmark — inter-node communication performance.

Benchmarks all_reduce, all_gather, send/recv, and ring allreduce bandwidth
over JACCL/Thunderbolt 5 interconnect. When multi-node is not available,
runs single-node baselines measuring local mx.array copy + operation latency.

Usage:
    # Full benchmark suite
    uv run python scripts/bench_jaccl.py

    # Custom sizes
    uv run python scripts/bench_jaccl.py --sizes 1K,64K,1M,16M,64M

    # Save results
    uv run python scripts/bench_jaccl.py --output jaccl_baseline.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "python"))

import mlx.core as mx

logger = logging.getLogger("bench_jaccl")

# Standard tensor sizes for the benchmark (in elements)
DEFAULT_SIZES = [
    1_024,          # 4 KB (float32)
    16_384,         # 64 KB
    65_536,         # 256 KB
    262_144,        # 1 MB
    1_048_576,      # 4 MB
    4_194_304,      # 16 MB
    16_777_216,     # 64 MB
    67_108_864,     # 256 MB
]


def _parse_size(s: str) -> int:
    """Parse a human-readable size string like '1K', '64M', '1G'."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def _fmt_bytes(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.1f} GB"
    if nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


def _fmt_elements(n: int) -> str:
    if n >= 10**6:
        return f"{n / 10**6:.1f}M"
    if n >= 10**3:
        return f"{n / 10**3:.1f}K"
    return str(n)


# ── Config ──


@dataclass
class JACCLBenchConfig:
    """Configuration for JACCL baseline benchmarks."""

    tensor_sizes: list[int] = field(default_factory=lambda: list(DEFAULT_SIZES))
    num_warmup: int = 5
    num_iters: int = 100
    backend: str = "any"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> JACCLBenchConfig:
        sizes = [_parse_size(s.strip()) for s in args.sizes.split(",") if s.strip()]
        return cls(
            tensor_sizes=sizes if sizes else list(DEFAULT_SIZES),
            num_warmup=args.warmup,
            num_iters=args.iters,
            backend=args.backend,
        )


# ── Benchmark runner ──


class JACCLBenchmark:
    """Benchmark JACCL/Thunderbolt inter-node communication.

    In single-node mode, measures local mx.array operation latency as a
    baseline reference point. These baselines represent the lower bound
    of what distributed communication would add.
    """

    def __init__(self, config: JACCLBenchConfig) -> None:
        self._config = config
        self._distributed = False
        self._group: object | None = None

    def _try_init_distributed(self) -> bool:
        """Try to initialise mx.distributed. Returns True only if world_size > 1."""
        try:
            if not mx.distributed.is_available():
                return False
            group = mx.distributed.init(backend=self._config.backend)
            # A singleton group (size=1) cannot do real collective communication
            if group.size() <= 1:
                logger.info("mx.distributed returned singleton group — single-node baseline")
                return False
            self._group = group
            self._distributed = True
            logger.info(
                "mx.distributed initialised: rank=%d size=%d",
                self._group.rank(),
                self._group.size(),
            )
            return True
        except Exception as exc:
            logger.warning("mx.distributed init failed (%s) — single-node baseline", exc)
            return False

    # ── Individual benchmarks ──

    def bench_all_reduce(self, size: int, group: object | None = None) -> dict:
        """Benchmark all_reduce at a given tensor size.

        Returns: {op, size, bytes, avg_ms, p50_ms, p99_ms, bandwidth_gbs, mode}
        """
        x = mx.ones((size,), dtype=mx.float32)
        mx.eval(x)

        # Warmup
        for _ in range(self._config.num_warmup):
            if self._distributed:
                r = mx.distributed.all_sum(x, group=group)
            else:
                r = x + mx.zeros_like(x)
            mx.eval(r)

        # Timed iterations
        lats = []
        for _ in range(self._config.num_iters):
            t0 = time.monotonic()
            if self._distributed:
                r = mx.distributed.all_sum(x, group=group)
            else:
                r = x + mx.zeros_like(x)
            mx.eval(r)
            lats.append((time.monotonic() - t0) * 1000.0)

        return self._build_result("all_reduce", size, lats)

    def bench_all_gather(self, size: int, group: object | None = None) -> dict:
        """Benchmark all_gather at a given tensor size."""
        x = mx.ones((size,), dtype=mx.float32)
        mx.eval(x)

        # Warmup
        for _ in range(self._config.num_warmup):
            if self._distributed:
                r = mx.distributed.all_gather(x, group=group)
            else:
                r = mx.concatenate([x], axis=0)
            mx.eval(r)

        # Timed iterations
        lats = []
        for _ in range(self._config.num_iters):
            t0 = time.monotonic()
            if self._distributed:
                r = mx.distributed.all_gather(x, group=group)
            else:
                r = mx.concatenate([x], axis=0)
            mx.eval(r)
            lats.append((time.monotonic() - t0) * 1000.0)

        return self._build_result("all_gather", size, lats)

    def bench_send_recv(self, size: int, group: object | None = None) -> dict:
        """Benchmark point-to-point send/recv latency.

        In single-node mode, measures a local copy + add operation as baseline.
        """
        x = mx.ones((size,), dtype=mx.float32)
        mx.eval(x)

        # Warmup
        for _ in range(self._config.num_warmup):
            if self._distributed:
                rank = self._group.rank()
                world = self._group.size()
                dst = (rank + 1) % world
                src = (rank - 1) % world
                mx.distributed.send(x, dst=dst, group=group)
                r = mx.distributed.recv(x.shape, x.dtype, src=src, group=group)
            else:
                r = x + mx.zeros_like(x)
            mx.eval(r)

        # Timed iterations
        lats = []
        for _ in range(self._config.num_iters):
            t0 = time.monotonic()
            if self._distributed:
                rank = self._group.rank()
                world = self._group.size()
                dst = (rank + 1) % world
                src = (rank - 1) % world
                mx.distributed.send(x, dst=dst, group=group)
                r = mx.distributed.recv(x.shape, x.dtype, src=src, group=group)
            else:
                r = x + mx.zeros_like(x)
            mx.eval(r)
            lats.append((time.monotonic() - t0) * 1000.0)

        return self._build_result("send_recv", size, lats)

    def bench_ring_bandwidth(self, group: object | None = None) -> dict:
        """Measure ring allreduce bandwidth across all configured tensor sizes.

        Returns aggregated results with size-vs-bandwidth data points.
        """
        results = []
        for size in self._config.tensor_sizes:
            result = self.bench_all_reduce(size, group)
            results.append(result)

        # Compute bandwidth summary
        bandwidths = [r["bandwidth_gbs"] for r in results if r.get("bandwidth_gbs", 0) > 0]
        return {
            "op": "ring_allreduce_sweep",
            "mode": results[0]["mode"] if results else "single_node",
            "sizes_tested": len(results),
            "peak_bandwidth_gbs": max(bandwidths) if bandwidths else 0.0,
            "avg_bandwidth_gbs": statistics.mean(bandwidths) if bandwidths else 0.0,
            "per_size": results,
        }

    # ── Full benchmark run ──

    def run(self, group: object | None = None) -> list[dict]:
        """Run all benchmarks across all configured tensor sizes.

        Returns a flat list of result dicts, one per (op, size) pair.
        """
        self._try_init_distributed()
        group = group or self._group
        mode = "distributed" if self._distributed else "single_node"
        logger.info("Running JACCL benchmarks: mode=%s sizes=%d", mode, len(self._config.tensor_sizes))

        results: list[dict] = []
        ops = [
            ("all_reduce", self.bench_all_reduce),
            ("all_gather", self.bench_all_gather),
            ("send_recv", self.bench_send_recv),
        ]

        for op_name, bench_fn in ops:
            logger.info("Benchmarking %s ...", op_name)
            for size in self._config.tensor_sizes:
                result = bench_fn(size, group)
                results.append(result)
                logger.debug(
                    "  %s %s: %.3f ms (bw: %.2f GB/s)",
                    op_name, _fmt_elements(size), result["avg_ms"], result["bandwidth_gbs"],
                )

        return results

    def format_results(self, results: list[dict]) -> str:
        """Format benchmark results as a readable table."""
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 90)
        lines.append(f"  JACCL Baseline Benchmark  (mode: {results[0]['mode'] if results else 'N/A'})")
        lines.append("=" * 90)
        lines.append(
            f"  {'Op':<15} {'Elements':>10} {'Bytes':>12} "
            f"{'Avg (ms)':>10} {'P50 (ms)':>10} {'P99 (ms)':>10} {'BW (GB/s)':>10}"
        )
        lines.append("-" * 90)

        for r in results:
            lines.append(
                f"  {r['op']:<15} {_fmt_elements(r['size']):>10} {_fmt_bytes(r['bytes']):>12} "
                f"{r['avg_ms']:>10.3f} {r['p50_ms']:>10.3f} {r['p99_ms']:>10.3f} {r['bandwidth_gbs']:>10.2f}"
            )

        lines.append("=" * 90)
        lines.append("  Note: single_node mode measures local mx.array operation latency.")
        lines.append("  These represent the lower-bound baseline for distributed ops.")
        lines.append("=" * 90)
        lines.append("")
        return "\n".join(lines)

    # ── Helpers ──

    def _build_result(self, op_name: str, size: int, lats: list[float]) -> dict:
        """Build a result dict from measured latencies."""
        nbytes = size * 4  # float32
        sorted_lats = sorted(lats)
        avg_ms = statistics.mean(lats) if lats else 0.0
        p50_ms = sorted_lats[len(sorted_lats) // 2] if sorted_lats else 0.0
        p99_idx = min(int(len(sorted_lats) * 0.99), len(sorted_lats) - 1) if sorted_lats else 0
        p99_ms = sorted_lats[p99_idx] if sorted_lats else 0.0

        # Bandwidth = bytes / latency. For all_reduce, data volume = 2 * size * sizeof(dtype)
        # (reduce-scatter + all-gather). For local ops, it's just the copy size.
        data_volume = nbytes * 2 if op_name == "all_reduce" else nbytes
        bandwidth_gbs = (data_volume / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0.0

        return {
            "op": op_name,
            "size": size,
            "bytes": nbytes,
            "avg_ms": round(avg_ms, 4),
            "p50_ms": round(p50_ms, 4),
            "p99_ms": round(p99_ms, 4),
            "bandwidth_gbs": round(bandwidth_gbs, 3),
            "mode": "distributed" if self._distributed else "single_node",
        }


# ── CLI ──


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JACCL baseline benchmark — inter-node communication performance",
    )
    parser.add_argument(
        "--sizes", type=str, default="1K,64K,256K,1M,4M,16M,64M,256M",
        help="Comma-separated tensor sizes (e.g. 1K,64K,1M,16M)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Number of timed iterations (default: 100)",
    )
    parser.add_argument(
        "--backend", type=str, default="any",
        help="mx.distributed backend: any, jaccl, ring, mpi (default: any)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = JACCLBenchConfig.from_args(args)
    bench = JACCLBenchmark(config)
    results = bench.run()

    # Print formatted table
    print(bench.format_results(results))

    # Save JSON report
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": {
                "tensor_sizes": config.tensor_sizes,
                "num_warmup": config.num_warmup,
                "num_iters": config.num_iters,
                "backend": config.backend,
            },
            "results": results,
        }
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
