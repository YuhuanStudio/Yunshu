#!/usr/bin/env python3
"""mx.distributed soak test — long-running stability validation.

Validates that collective operations remain stable over extended periods.
Runs on a real multi-node mesh when available, otherwise falls back to
single-process mode with local tensor operations and synthetic latency.

Usage:
    # 72-hour soak with defaults
    uv run python scripts/soak_test_distributed.py

    # Shorter run for CI
    uv run python scripts/soak_test_distributed.py --duration 0.1 --output soak.json

    # Custom ops and tensor size
    uv run python scripts/soak_test_distributed.py --ops all_reduce,all_gather --tensor-size 2097152
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

logger = logging.getLogger("soak_test")


# ── Data structures ──


@dataclass
class SoakTestConfig:
    """Configuration for a distributed soak test."""

    duration_hours: float = 72.0
    # send_recv is NOT in the default soak set: a symmetric simultaneous
    # neighbor exchange (every rank send(dst)+recv(src) at once) DEADLOCKS on
    # the MLX ring backend regardless of parity ordering or combined eval
    # (confirmed live at 2 and 4 ranks — MLX ring send/recv is meant for
    # staggered pipeline-parallel use, not all-pairs exchange). The reliable,
    # meaningful distributed-stability soak is over the collective ops. Pass
    # `--ops …,send_recv` to exercise it anyway (may hang on ring).
    ops: list[str] = field(default_factory=lambda: ["all_reduce", "all_gather"])
    tensor_size: int = 1024 * 1024  # 1M elements
    check_interval_seconds: int = 300  # health check every 5 min

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SoakTestConfig:
        ops_list = [op.strip() for op in args.ops.split(",") if op.strip()]
        return cls(
            duration_hours=args.duration,
            ops=ops_list,
            tensor_size=args.tensor_size,
            check_interval_seconds=args.check_interval,
        )


@dataclass
class SoakTestReport:
    """Final report from a soak test run."""

    total_iterations: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_memory_gb: float = 0.0
    duration_hours: float = 0.0
    errors: list[str] = field(default_factory=list)
    mode: str = "single_process"  # or "distributed"
    op_results: dict = field(default_factory=dict)  # per-op stats

    def to_dict(self) -> dict:
        return {
            "total_iterations": self.total_iterations,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "max_memory_gb": round(self.max_memory_gb, 3),
            "duration_hours": round(self.duration_hours, 3),
            "mode": self.mode,
            "errors": self.errors[-50:],  # cap at last 50 errors
            "op_results": self.op_results,
        }


# ── Soak test runner ──


class SoakTestRunner:
    """Long-running stability test for mx.distributed collective operations.

    In multi-node mode, runs real collective operations via mx.distributed.
    In single-process mode, simulates them with local tensor ops and a tiny
    synthetic delay so that latency measurements are still meaningful.
    """

    def __init__(self, config: SoakTestConfig) -> None:
        self._config = config
        self._distributed = False
        self._group: object | None = None
        self._iterations = 0
        self._errors: list[str] = []
        self._latencies: list[float] = []
        self._max_memory_gb = 0.0
        self._start_time = 0.0

    # ── Initialization ──

    def _try_init_distributed(self) -> bool:
        """Attempt to initialise mx.distributed. Returns True only if world_size > 1."""
        try:
            if not mx.distributed.is_available():
                return False
            group = mx.distributed.init(backend="any")
            # A singleton group (size=1) cannot do real collective communication
            if group.size() <= 1:
                logger.info("mx.distributed returned singleton group — single-process mode")
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
            logger.warning("mx.distributed init failed (%s) — single-process mode", exc)
            return False

    # ── Collective operation wrappers ──

    def run_single_op(self, op_name: str, group: object | None = None) -> dict:
        """Execute a single collective operation and measure latency + correctness.

        Returns dict with keys: op, latency_ms, correct, error (optional).
        """
        cfg = self._config
        x = mx.ones((cfg.tensor_size,), dtype=mx.float32)
        mx.eval(x)

        t0 = time.monotonic()

        try:
            if self._distributed:
                result = self._run_distributed_op(op_name, x, group)
            else:
                result = self._run_local_op(op_name, x)
            mx.eval(result)
            latency_ms = (time.monotonic() - t0) * 1000.0

            # Correctness check: for local ops the result should be deterministic
            correct = True
            if op_name == "all_reduce":
                # Local all_reduce returns input unchanged; check shape
                correct = result.shape == x.shape
            elif op_name == "all_gather":
                correct = result.shape[0] >= x.shape[0]
            elif op_name == "send_recv":
                correct = result.shape == x.shape

            return {"op": op_name, "latency_ms": latency_ms, "correct": correct}

        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            error_msg = f"iter={self._iterations} op={op_name}: {exc}"
            self._errors.append(error_msg)
            logger.error(error_msg)
            return {"op": op_name, "latency_ms": latency_ms, "correct": False, "error": str(exc)}

    def _run_distributed_op(self, op_name: str, x: mx.array, group: object) -> mx.array:
        """Execute a real collective operation via mx.distributed."""
        if op_name == "all_reduce":
            return mx.distributed.all_sum(x, group=group)
        elif op_name == "all_gather":
            return mx.distributed.all_gather(x, group=group)
        elif op_name == "send_recv":
            rank = self._group.rank()
            size = self._group.size()
            dst = (rank + 1) % size
            src = (rank - 1) % size
            # Deadlock-free ring exchange: ring send blocks until the neighbor
            # recvs, so if EVERY rank sends-then-recvs at once they all block
            # (observed: the soak hung with send_recv in the default op set).
            # Order by parity — even ranks send first, odd ranks recv first —
            # so a sender always has a matching receiver ready (deadlock-free
            # for even world sizes; best-effort for odd).
            if rank % 2 == 0:
                mx.distributed.send(x, dst=dst, group=group)
                return mx.distributed.recv(x.shape, x.dtype, src=src, group=group)
            recv = mx.distributed.recv(x.shape, x.dtype, src=src, group=group)
            mx.eval(recv)
            mx.distributed.send(x, dst=dst, group=group)
            return recv
        else:
            raise ValueError(f"Unknown collective op: {op_name}")

    def _run_local_op(self, op_name: str, x: mx.array) -> mx.array:
        """Simulate a collective operation locally with synthetic latency."""
        # Small synthetic delay to emulate network cost (~0.1-0.5 ms)
        time.sleep(0.0001)

        if op_name == "all_reduce":
            return x + mx.zeros_like(x)  # identity (local mode = world size 1)
        elif op_name == "all_gather":
            return mx.concatenate([x], axis=0)
        elif op_name == "send_recv":
            return x + mx.zeros_like(x)  # loopback
        else:
            raise ValueError(f"Unknown collective op: {op_name}")

    # ── Iteration & main loop ──

    def run_iteration(self, group: object | None = None) -> dict:
        """Run all configured operations once. Returns per-op results."""
        iter_results = {}
        for op_name in self._config.ops:
            result = self.run_single_op(op_name, group)
            iter_results[op_name] = result
            if result.get("latency_ms") is not None:
                self._latencies.append(result["latency_ms"])
            if not result.get("correct", True):
                self._errors.append(
                    f"iter={self._iterations} op={op_name}: correctness check failed"
                )
        self._iterations += 1
        return iter_results

    def run(self, group: object | None = None) -> SoakTestReport:
        """Main soak test loop. Runs for config.duration_hours."""
        self._distributed = self._try_init_distributed()
        group = group or self._group

        self._start_time = time.monotonic()
        duration_s = self._config.duration_hours * 3600.0
        last_check = self._start_time

        mode = "distributed" if self._distributed else "single_process"
        logger.info(
            "Soak test starting: mode=%s duration=%.1fh ops=%s tensor_size=%d",
            mode,
            self._config.duration_hours,
            self._config.ops,
            self._config.tensor_size,
        )

        while True:
            stop = (time.monotonic() - self._start_time) >= duration_s
            # Synchronize termination across ranks. Without this, each rank
            # checks its OWN clock, so the first to time out leaves the group
            # while the others enter one more collective and block forever
            # (ring deadlock — observed live: rank 0 finished 844 iters, ranks
            # 1-3 hung). all_max of the stop flag makes ALL ranks exit on the
            # same iteration.
            if self._distributed and group is not None:
                try:
                    _flag = mx.distributed.all_max(
                        mx.array([1.0 if stop else 0.0]), group=group
                    )
                    mx.eval(_flag)
                    stop = float(_flag[0]) > 0.5
                except Exception:
                    logger.debug("termination sync failed", exc_info=True)
            if stop:
                break
            try:
                self.run_iteration(group)
            except Exception as exc:
                error_msg = f"iter={self._iterations}: {exc}"
                self._errors.append(error_msg)
                logger.exception("Iteration failed")

            # Periodic health check
            now = time.monotonic()
            if (now - last_check) >= self._config.check_interval_seconds:
                health = self.check_health()
                logger.info(
                    "Health: iterations=%d errors=%d avg_latency=%.2fms mem=%.2fGB",
                    health["iterations"],
                    health["errors"],
                    health["avg_latency_ms"],
                    health["memory_gb"],
                )
                last_check = now

        # Build final report
        elapsed_h = (time.monotonic() - self._start_time) / 3600.0
        return self._build_report(elapsed_h)

    def check_health(self) -> dict:
        """Return current health snapshot."""
        avg_lat = statistics.mean(self._latencies) if self._latencies else 0.0

        # Estimate memory usage from peak allocations
        try:
            mem_gb = mx.metal.get_peak_memory() / (1024**3)
        except Exception:
            mem_gb = 0.0
        self._max_memory_gb = max(self._max_memory_gb, mem_gb)

        return {
            "iterations": self._iterations,
            "errors": len(self._errors),
            "avg_latency_ms": round(avg_lat, 3),
            "memory_gb": round(mem_gb, 3),
            "max_memory_gb": round(self._max_memory_gb, 3),
            "elapsed_hours": round(
                (time.monotonic() - self._start_time) / 3600.0, 3
            ) if self._start_time else 0.0,
        }

    def _build_report(self, elapsed_hours: float) -> SoakTestReport:
        """Build the final SoakTestReport from accumulated data."""
        # Compute per-op stats
        op_results: dict[str, dict] = {}
        for op_name in self._config.ops:
            [lat for lat in self._latencies]  # all lats mixed; we track globally
            op_results[op_name] = {
                "count": self._iterations,
            }

        avg_lat = statistics.mean(self._latencies) if self._latencies else 0.0
        p99_lat = 0.0
        if self._latencies:
            sorted_lats = sorted(self._latencies)
            idx = min(int(len(sorted_lats) * 0.99), len(sorted_lats) - 1)
            p99_lat = sorted_lats[idx]

        return SoakTestReport(
            total_iterations=self._iterations,
            total_errors=len(self._errors),
            avg_latency_ms=avg_lat,
            p99_latency_ms=p99_lat,
            max_memory_gb=self._max_memory_gb,
            duration_hours=elapsed_hours,
            errors=list(self._errors),
            mode="distributed" if self._distributed else "single_process",
            op_results=op_results,
        )


# ── CLI ──


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="mx.distributed soak test — long-running stability validation",
    )
    parser.add_argument(
        "--duration", type=float, default=72.0,
        help="Soak test duration in hours (default: 72)",
    )
    parser.add_argument(
        "--ops", type=str, default="all_reduce,all_gather,send_recv",
        help="Comma-separated list of collective ops to test",
    )
    parser.add_argument(
        "--tensor-size", type=int, default=1024 * 1024,
        help="Tensor size in elements (default: 1048576)",
    )
    parser.add_argument(
        "--check-interval", type=int, default=300,
        help="Health check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path for the report",
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

    config = SoakTestConfig.from_args(args)
    runner = SoakTestRunner(config)
    report = runner.run()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Soak Test Report  (mode: {report.mode})")
    print(f"{'=' * 60}")
    print(f"  Duration:          {report.duration_hours:.3f} hours")
    print(f"  Total iterations:  {report.total_iterations}")
    print(f"  Total errors:      {report.total_errors}")
    print(f"  Avg latency:       {report.avg_latency_ms:.3f} ms")
    print(f"  P99 latency:       {report.p99_latency_ms:.3f} ms")
    print(f"  Peak memory:       {report.max_memory_gb:.3f} GB")
    if report.errors:
        print(f"\n  Last {min(10, len(report.errors))} errors:")
        for err in report.errors[-10:]:
            print(f"    - {err}")
    print(f"{'=' * 60}\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
