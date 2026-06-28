#!/usr/bin/env python3
"""CLI tool for Apple Silicon roofline analysis.

Usage:
    uv run python scripts/roofline.py --chip M4_Max --model Qwen2.5-9B
    uv run python scripts/roofline.py --plot roofline.png
    uv run python scripts/roofline.py --chip M3_Max --model Qwen2.5-7B --context 2048
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from yunshu_engine.roofline import MODEL_CONFIGS, RooflineModel


def _fmt_num(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def cmd_estimate(args: argparse.Namespace) -> None:
    rm = RooflineModel(chip_name=args.chip)
    result = rm.estimate_max_throughput(args.model, context_len=args.context)
    print(f"\n{'=' * 60}")
    print(f"  Roofline Estimate: {result['model']} on {result['chip']}")
    print(f"{'=' * 60}")
    print(f"  Context length:    {args.context}")
    print(f"  Bound:             {result['bound']}")
    print(f"  Max throughput:    {result['tokens_per_sec']:.2f} tokens/sec")
    print(f"  GFLOP/token:       {result['gflops_per_token']:.2f}")
    print(f"  Operational int.:  {result['operational_intensity']:.4f} FLOP/byte")
    print(f"  Bandwidth:         {result['bandwidth_gbs']:.0f} GB/s")
    print(f"  Peak compute:      {result['compute_tflops']:.1f} TFLOP/s")
    print(f"{'=' * 60}\n")

    if args.json:
        print(json.dumps(result, indent=2))


def cmd_decode(args: argparse.Namespace) -> None:
    rm = RooflineModel(chip_name=args.chip)
    config = rm._resolve_model_config(args.model)
    result = rm.compute_decode_roofline(config, context_len=args.context)

    print(f"\n{'=' * 60}")
    print(f"  Decode Roofline: {args.model} on {rm.chip_key}")
    print(f"{'=' * 60}")
    print(f"  Total FLOPs:    {_fmt_num(result.total_flops)}")
    print(f"  Total bytes:    {_fmt_num(result.total_bytes)}")
    print(f"  Op intensity:   {result.operational_intensity:.4f} FLOP/byte")
    print(f"  Predicted:      {result.predicted_gflops:.2f} GFLOP/s")
    print(f"  Bound:          {result.bound}")

    # Per-op breakdown (group by layer type, not every single op)
    print(f"\n  {'Op':<30s} {'FLOPs':>10s} {'Bytes':>10s} {'OI':>10s} {'Bound':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for op in result.ops:
        print(
            f"  {op.label:<30s} "
            f"{_fmt_num(op.flops):>10s} "
            f"{_fmt_num(op.bytes_accessed):>10s} "
            f"{op.operational_intensity:>10.4f} "
            f"{op.bound:>8s}"
        )
    print(f"{'=' * 60}\n")


def cmd_plot(args: argparse.Namespace) -> None:
    rm = RooflineModel(chip_name=args.chip)
    path = rm.plot_roofline(args.plot)
    print(f"Roofline plot saved to: {path}")


def cmd_list_models(_: argparse.Namespace) -> None:
    print("\nKnown model configurations:")
    for name, cfg in sorted(MODEL_CONFIGS.items()):
        params = (
            f"H={cfg['hidden_size']}, L={cfg['num_layers']}, "
            f"NH={cfg['num_heads']}, FFN={cfg['ffn_dim']}"
        )
        print(f"  {name:<20s}  {params}")
    print()


def cmd_list_chips(_: argparse.Namespace) -> None:
    from yunshu_engine.roofline import CHIP_PARAMS
    print("\nKnown chip configurations:")
    print(f"  {'Chip':<12s} {'BW (GB/s)':>10s} {'FP16 TFLOP/s':>14s}")
    print(f"  {'-'*12} {'-'*10} {'-'*14}")
    for name, params in sorted(CHIP_PARAMS.items()):
        print(f"  {name:<12s} {params['bandwidth_gbps']:>10.0f} {params['compute_tflops_fp16']:>14.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apple Silicon roofline analysis for LLM inference",
    )
    parser.add_argument(
        "--chip", default=None,
        help="Chip name (e.g. M3_Max, M4_Pro). Auto-detects if omitted.",
    )

    sub = parser.add_subparsers(dest="command")

    # estimate
    p_est = sub.add_parser("estimate", help="Estimate max tokens/sec for a model")
    p_est.add_argument("model", help="Model name (e.g. Qwen2.5-9B)")
    p_est.add_argument("--context", type=int, default=0, help="Context length (default: 0)")
    p_est.add_argument("--json", action="store_true", help="Print JSON output")

    # decode
    p_dec = sub.add_parser("decode", help="Full decode step roofline breakdown")
    p_dec.add_argument("model", help="Model name")
    p_dec.add_argument("--context", type=int, default=0, help="Context length")

    # plot
    p_plot = sub.add_parser("plot", help="Generate roofline plot")
    p_plot.add_argument("plot", nargs="?", default="roofline.png", help="Output path")

    # list
    sub.add_parser("list-models", help="List known model configurations")
    sub.add_parser("list-chips", help="List known chip configurations")

    # Convenience: --plot as top-level flag
    parser.add_argument("--plot", default=None, help="Generate roofline plot (shortcut)")
    parser.add_argument("--model", default=None, help="Model name (shortcut for estimate)")
    parser.add_argument("--context", type=int, default=0, help="Context length")

    args = parser.parse_args()

    # Handle shortcuts
    if args.plot is not None:
        cmd_plot(args)
        return

    if args.model is not None and args.command is None:
        args.json = False
        cmd_estimate(args)
        return

    if args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "decode":
        cmd_decode(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "list-chips":
        cmd_list_chips(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
