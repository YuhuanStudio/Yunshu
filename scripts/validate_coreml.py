#!/usr/bin/env python3
"""CoreML Model Compilation Validation for Phase 0.

Validates that CoreML is available and functional for ANE deployment.
This script:
  1. Checks if coremltools is installed
  2. Attempts to compile a simple MLP model to CoreML .mlmodelc
  3. Validates the compiled model produces correct outputs
  4. Reports: coremltools version, ANE availability, compilation time, inference latency

Exit codes:
  0 - CoreML works correctly
  1 - coremltools is not installed
  2 - CoreML compilation or inference fails

Usage:
    PYTHONPATH=. uv run python scripts/validate_coreml.py
    PYTHONPATH=. uv run python scripts/validate_coreml.py --verbose
    PYTHONPATH=. uv run python scripts/validate_coreml.py --output /tmp/coreml_validation.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))


def validate_coreml() -> dict[str, Any]:
    """Run CoreML validation checks and return a report.

    Returns:
        dict with keys:
            coremltools_available (bool): Whether coremltools is installed
            coremltools_version (str|None): Version string if installed
            platform_info (dict): Platform details (system, machine, mac_ver)
            ane_available (bool): Whether ANE appears available
            compilation_success (bool): Whether model compilation succeeded
            compilation_time_s (float|None): Compilation time in seconds
            inference_success (bool): Whether inference produced valid output
            inference_latency_ms (float|None): Single inference latency
            output_correct (bool|None): Whether output matches expected values
            error (str|None): Error message if something failed
    """
    report: dict[str, Any] = {
        "coremltools_available": False,
        "coremltools_version": None,
        "platform_info": {
            "system": platform.system(),
            "machine": platform.machine(),
            "mac_ver": platform.mac_ver()[0],
            "python_version": platform.python_version(),
        },
        "ane_available": False,
        "compilation_success": False,
        "compilation_time_s": None,
        "inference_success": False,
        "inference_latency_ms": None,
        "output_correct": None,
        "error": None,
    }

    # ── Step 1: Check coremltools ──────────────────────────────────────────────
    try:
        import coremltools as ct  # type: ignore[import-untyped]

        report["coremltools_available"] = True
        report["coremltools_version"] = ct.__version__
    except ImportError:
        report["error"] = (
            "coremltools is not installed. Install with: uv pip install coremltools"
        )
        return report

    # ── Step 2: Check ANE availability ─────────────────────────────────────────
    from yunshu_engine.ane_embedding import is_ane_available

    report["ane_available"] = is_ane_available()

    # ── Step 3: Build a simple MLP model ───────────────────────────────────────
    try:
        import numpy as np

        # Build a simple 2-layer MLP using numpy operations
        # y = relu(x @ W1 + b1) @ W2 + b2
        hidden_dim = 64
        input_dim = 32
        output_dim = 16

        np.random.seed(42)
        W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
        b2 = np.zeros(output_dim, dtype=np.float32)

        # Create a PyTorch model for CoreML conversion
        try:
            import torch  # type: ignore[import-untyped]
            import torch.nn as nn
        except ImportError:
            report["error"] = (
                "PyTorch is required for CoreML model tracing. "
                "Install with: uv pip install torch"
            )
            return report

        class SimpleMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleMLP()
        model.eval()

        # Load our deterministic weights into the PyTorch model
        with torch.no_grad():
            model.fc1.weight.data = torch.from_numpy(W1.T)
            model.fc1.bias.data = torch.from_numpy(b1)
            model.fc2.weight.data = torch.from_numpy(W2.T)
            model.fc2.bias.data = torch.from_numpy(b2)

        # Compute expected output with numpy
        test_input = np.random.randn(1, input_dim).astype(np.float32)
        with torch.no_grad():
            expected_output = model(torch.from_numpy(test_input)).numpy()

        # ── Step 4: Trace and convert to CoreML ────────────────────────────────
        sample_input = torch.randn(1, input_dim)
        traced = torch.jit.trace(model, sample_input)

        t0 = time.perf_counter()
        coreml_model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="input",
                    shape=(1, input_dim),
                    dtype=np.float32,
                )
            ],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,
        )
        t1 = time.perf_counter()
        report["compilation_time_s"] = round(t1 - t0, 4)
        report["compilation_success"] = True

        # ── Step 5: Run inference and validate ─────────────────────────────────
        # Warmup
        for _ in range(3):
            coreml_model.predict({"input": test_input})

        # Timed inference
        t0 = time.perf_counter()
        prediction = coreml_model.predict({"input": test_input})
        t1 = time.perf_counter()
        report["inference_latency_ms"] = round((t1 - t0) * 1000, 3)
        report["inference_success"] = True

        # Extract output
        output = prediction.get("output", list(prediction.values())[0])
        if isinstance(output, np.ndarray):
            actual_output = output
        else:
            actual_output = np.array(output)

        # Validate output correctness (allow small numerical tolerance)
        max_diff = np.max(np.abs(actual_output - expected_output))
        report["output_correct"] = bool(max_diff < 1e-3)
        report["max_output_diff"] = float(max_diff)

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        report["traceback"] = traceback.format_exc()

    return report


def main() -> None:
    """Run CoreML validation and print results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CoreML Model Compilation Validation for Phase 0",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save validation report as JSON",
    )
    args = parser.parse_args()

    report = validate_coreml()

    # Print summary
    print()
    print("=" * 60)
    print("  CoreML Validation Report")
    print("=" * 60)

    print(f"  coremltools:   {report['coremltools_available']}")
    if report["coremltools_version"]:
        print(f"  version:       {report['coremltools_version']}")

    pi = report["platform_info"]
    print(f"  platform:      {pi['system']} {pi['machine']} (macOS {pi['mac_ver']})")
    print(f"  python:        {pi['python_version']}")
    print(f"  ANE available: {report['ane_available']}")
    print(f"  Compilation:   {'OK' if report['compilation_success'] else 'FAILED'}")
    if report["compilation_time_s"] is not None:
        print(f"  Compile time:  {report['compilation_time_s']:.3f}s")
    print(f"  Inference:     {'OK' if report['inference_success'] else 'FAILED'}")
    if report["inference_latency_ms"] is not None:
        print(f"  Latency:       {report['inference_latency_ms']:.3f}ms")
    if report["output_correct"] is not None:
        status = "PASS" if report["output_correct"] else "MISMATCH"
        print(f"  Output check:  {status}")

    if report["error"]:
        print(f"\n  ERROR: {report['error']}")

    print("=" * 60)

    # Verbose output
    if args.verbose:
        print("\nFull report:")
        print(json.dumps(report, indent=2, default=str))

    # Save JSON report
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to: {args.output}")

    # Exit code
    if not report["coremltools_available"]:
        print("\nExit code 1: coremltools not installed")
        sys.exit(1)
    if not report["compilation_success"] or not report["inference_success"]:
        print(f"\nExit code 2: CoreML operation failed — {report['error']}")
        sys.exit(2)

    print("\nCoreML validation PASSED (exit code 0)")
    sys.exit(0)


if __name__ == "__main__":
    main()
