#!/usr/bin/env python3
"""Phase 0 Platform Validation — comprehensive single-script runner.

Runs all Phase 0 benchmarks and validations:
  1. Roofline benchmark (MLX bandwidth + GEMM throughput)
  2. Metal kernel validation
  3. KIVI Metal kernel benchmark
  4. ANE micro-benchmark
  5. CoreML validation
  6. mx.distributed soak test (short 10-second version)
  7. JACCL baseline benchmark

Collects all results into a single JSON report and prints a summary.

Usage:
    PYTHONPATH=. uv run python scripts/run_phase0_validation.py
    PYTHONPATH=. uv run python scripts/run_phase0_validation.py --output phase0_report.json
    PYTHONPATH=. uv run python scripts/run_phase0_validation.py --skip ane,coreml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _section_pass(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


# ── Individual validation steps ──


def run_roofline() -> dict[str, Any]:
    """Run roofline benchmark (MLX bandwidth + GEMM throughput)."""
    _header("Step 1: Roofline Benchmark")
    result: dict[str, Any] = {"name": "roofline", "passed": False, "details": {}}

    try:
        from yunshu_engine.roofline import measure_roofline, RooflineModel

        # Run empirical measurements
        measurements = measure_roofline()
        result["details"]["measurements"] = measurements

        # Also run analytical model
        rm = RooflineModel()
        analytical = rm.estimate_max_throughput("Qwen2.5-0.5B")
        result["details"]["analytical"] = analytical

        # Validate: bandwidth should be > 50 GB/s for modern Apple Silicon
        bw_ok = measurements["bandwidth_gbs"] > 50
        tflops_ok = measurements["tflops_peak"] > 0.5

        result["passed"] = bw_ok and tflops_ok
        _section_pass(
            "Roofline benchmark",
            result["passed"],
            f"BW={measurements['bandwidth_gbs']:.1f} GB/s, "
            f"Peak={measurements['tflops_peak']:.2f} TFLOPS, "
            f"Bound={measurements['bound_type']}",
        )

    except Exception as e:
        result["error"] = str(e)
        _section_pass("Roofline benchmark", False, str(e))

    return result


def run_metal_kernels() -> dict[str, Any]:
    """Run Metal kernel validation."""
    _header("Step 2: Metal Kernel Validation")
    result: dict[str, Any] = {"name": "metal_kernels", "passed": False, "details": {}}

    try:
        # Import and run the validation script's main function
        sys.path.insert(0, str(_ROOT / "scripts"))
        import validate_metal_kernels as vmk

        # Run each validation individually
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        results = {}
        results["kivi_quantize"] = vmk.validate_kivi_quantize(mgr)
        results["kivi_roundtrip"] = vmk.validate_kivi_roundtrip(mgr)
        results["paged_attention"] = vmk.validate_paged_attention(mgr)
        results["gemv"] = vmk.validate_gemv(mgr)
        results["sdpa"] = vmk.validate_sdpa(mgr)

        result["details"]["kernel_results"] = {k: "PASS" if v else "FAIL" for k, v in results.items()}

        total = len(results)
        passed = sum(1 for v in results.values() if v)
        result["passed"] = all(results.values())
        _section_pass(
            "Metal kernel validation",
            result["passed"],
            f"{passed}/{total} passed",
        )

    except Exception as e:
        result["error"] = str(e)
        _section_pass("Metal kernel validation", False, str(e))

    return result


def run_kivi_benchmark() -> dict[str, Any]:
    """Run KIVI Metal kernel benchmark."""
    _header("Step 3: KIVI Metal Benchmark")
    result: dict[str, Any] = {"name": "kivi_metal", "passed": False, "details": {}}

    try:
        import mlx.core as mx
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        configs = [
            (64, 32, 128),
            (256, 32, 128),
            (1024, 32, 128),
        ]

        results = []
        for num_tokens, num_heads, head_dim in configs:
            keys = mx.random.normal((num_tokens, num_heads, head_dim)).astype(mx.float16)

            # Warmup
            for _ in range(3):
                qk, s, zp = mgr.kivi_quantize(keys)
                dq = mgr.kivi_dequantize(qk, s, zp, head_dim)
            mx.synchronize()

            # Benchmark
            num_iters = 20
            t0 = time.perf_counter()
            for _ in range(num_iters):
                qk, s, zp = mgr.kivi_quantize(keys)
            mx.synchronize()
            quant_time_ms = (time.perf_counter() - t0) / num_iters * 1000

            qk, s, zp = mgr.kivi_quantize(keys)
            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_iters):
                dq = mgr.kivi_dequantize(qk, s, zp, head_dim)
            mx.synchronize()
            dequant_time_ms = (time.perf_counter() - t0) / num_iters * 1000

            original_bytes = keys.size * 2
            compressed_bytes = qk.size + s.size * 2 + zp.size * 2
            ratio = original_bytes / compressed_bytes

            results.append({
                "tokens": num_tokens,
                "quant_ms": round(quant_time_ms, 3),
                "dequant_ms": round(dequant_time_ms, 3),
                "compression_ratio": round(ratio, 2),
            })

        result["details"]["results"] = results
        # Pass if compression ratio > 2x and times are reasonable (< 100ms)
        all_ok = all(
            r["compression_ratio"] > 2.0 and r["quant_ms"] < 100
            for r in results
        )
        result["passed"] = all_ok
        _section_pass(
            "KIVI benchmark",
            result["passed"],
            f"{len(results)} configs tested",
        )

    except Exception as e:
        result["error"] = str(e)
        _section_pass("KIVI benchmark", False, str(e))

    return result


def run_ane_benchmark() -> dict[str, Any]:
    """Run ANE micro-benchmark."""
    _header("Step 4: ANE Micro-Benchmark")
    result: dict[str, Any] = {"name": "ane_benchmark", "passed": False, "details": {}}

    try:
        sys.path.insert(0, str(_ROOT / "scripts"))
        import bench_ane

        config = bench_ane.ANEBenchConfig(
            model_sizes=[10, 50],
            seq_lengths=[32, 128],
            num_warmup=2,
            num_iters=10,
        )
        bench = bench_ane.ANEBenchmark(config)
        results = bench.run()

        result["details"]["num_results"] = len(results)
        ok_results = sum(1 for r in results if r.get("status") == "ok")
        result["details"]["gpu_ok"] = ok_results

        # Pass if we got at least some GPU results
        result["passed"] = ok_results > 0
        _section_pass(
            "ANE benchmark",
            result["passed"],
            f"{ok_results}/{len(results)} successful",
        )

    except Exception as e:
        result["error"] = str(e)
        _section_pass("ANE benchmark", False, str(e))

    return result


def run_coreml_validation() -> dict[str, Any]:
    """Run CoreML validation."""
    _header("Step 5: CoreML Validation")
    result: dict[str, Any] = {"name": "coreml", "passed": False, "details": {}}

    try:
        sys.path.insert(0, str(_ROOT / "scripts"))
        import validate_coreml

        report = validate_coreml.validate_coreml()
        result["details"] = report

        # Pass if coremltools is available and compilation succeeded
        result["passed"] = report.get("compilation_success", False)
        status_detail = (
            f"coremltools={report.get('coremltools_available')}, "
            f"compilation={'OK' if report.get('compilation_success') else 'FAILED'}, "
            f"ANE={report.get('ane_available')}"
        )
        _section_pass("CoreML validation", result["passed"], status_detail)

    except Exception as e:
        result["error"] = str(e)
        _section_pass("CoreML validation", False, str(e))

    return result


def run_distributed_soak() -> dict[str, Any]:
    """Run distributed soak test (short 10-second version)."""
    _header("Step 6: mx.distributed Soak Test (10s)")
    result: dict[str, Any] = {"name": "distributed_soak", "passed": False, "details": {}}

    try:
        sys.path.insert(0, str(_ROOT / "scripts"))
        import soak_test_distributed as soak

        config = soak.SoakTestConfig(
            duration_hours=10.0 / 3600.0,  # 10 seconds
            ops=["all_reduce", "all_gather", "send_recv"],
            tensor_size=1024 * 1024,
            check_interval_seconds=5,
        )
        runner = soak.SoakTestRunner(config)
        report = runner.run()

        result["details"] = report.to_dict()
        result["passed"] = report.total_errors == 0
        _section_pass(
            "Distributed soak test",
            result["passed"],
            f"{report.total_iterations} iters, {report.total_errors} errors, "
            f"mode={report.mode}",
        )

    except Exception as e:
        result["error"] = str(e)
        _section_pass("Distributed soak test", False, str(e))

    return result


def run_jaccl_benchmark() -> dict[str, Any]:
    """Run JACCL baseline benchmark."""
    _header("Step 7: JACCL Baseline Benchmark")
    result: dict[str, Any] = {"name": "jaccl", "passed": False, "details": {}}

    try:
        sys.path.insert(0, str(_ROOT / "scripts"))
        import bench_jaccl

        config = bench_jaccl.JACCLBenchConfig(
            tensor_sizes=[1024, 65536, 1_048_576],
            num_warmup=3,
            num_iters=20,
        )
        bench = bench_jaccl.JACCLBenchmark(config)
        results = bench.run()

        result["details"]["num_results"] = len(results)
        result["details"]["results"] = results
        result["passed"] = len(results) > 0
        _section_pass(
            "JACCL benchmark",
            result["passed"],
            f"{len(results)} benchmark configs",
        )

    except Exception as e:
        result["error"] = str(e)
        _section_pass("JACCL benchmark", False, str(e))

    return result


# ── Main orchestrator ──


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Yunshu Phase 0 — Platform Validation Runner",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save JSON report to this file",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of steps to skip (roofline,metal,kivi,ane,coreml,soak,jaccl)",
    )
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print("=" * 60)
    print("  Yunshu Phase 0 — Platform Validation")
    print("=" * 60)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Root: {_ROOT}")
    if skip:
        print(f"  Skip: {', '.join(sorted(skip))}")

    all_results: list[dict[str, Any]] = []

    # Run each step
    steps = [
        ("roofline", "roofline", run_roofline),
        ("metal", "metal_kernels", run_metal_kernels),
        ("kivi", "kivi_metal", run_kivi_benchmark),
        ("ane", "ane_benchmark", run_ane_benchmark),
        ("coreml", "coreml", run_coreml_validation),
        ("soak", "distributed_soak", run_distributed_soak),
        ("jaccl", "jaccl", run_jaccl_benchmark),
    ]

    for skip_key, name, runner in steps:
        if skip_key in skip:
            _header(f"Skipping: {name}")
            print("  (skipped by --skip flag)")
            all_results.append({"name": name, "passed": None, "skipped": True})
            continue
        all_results.append(runner())

    # ── Summary ──
    _header("Phase 0 Validation Summary")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("passed") is True)
    failed = sum(1 for r in all_results if r.get("passed") is False)
    skipped = sum(1 for r in all_results if r.get("skipped"))

    for r in all_results:
        name = r["name"]
        if r.get("skipped"):
            print(f"  [SKIP] {name}")
        elif r.get("passed"):
            print(f"  [PASS] {name}")
        else:
            error = r.get("error", "")
            detail = f" — {error}" if error else ""
            print(f"  [FAIL] {name}{detail}")

    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    # Save JSON report
    report = {
        "phase": "phase0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "overall_pass": failed == 0,
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  Report saved to: {args.output}")

    print(f"\n{'='*60}")

    if failed > 0:
        print("  *** PHASE 0 VALIDATION FAILED ***")
        return 1
    else:
        print("  *** PHASE 0 VALIDATION PASSED ***")
        return 0


if __name__ == "__main__":
    sys.exit(main())
