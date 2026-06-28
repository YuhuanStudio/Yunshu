#!/usr/bin/env python3
"""Yunshu Benchmark CLI — measure LLM inference performance.

Usage examples:
    # Run the default suite
    uv run python scripts/bench.py --model Qwen2.5-0.5B-Instruct-4bit --suite default

    # Single request with specific prompt length
    uv run python scripts/bench.py --model Qwen2.5-0.5B-Instruct-4bit --prompt-len 1024 --max-tokens 512

    # Batch benchmark
    uv run python scripts/bench.py --model Qwen2.5-0.5B-Instruct-4bit --batch 4 --output bench_results.json

    # Prefill-vs-decode sweep
    uv run python scripts/bench.py --model Qwen2.5-0.5B-Instruct-4bit --sweep prefill

No model weights are required if --mock is passed (for testing the harness itself).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Yunshu Benchmark CLI — measure LLM inference performance",
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name or path (e.g. Qwen2.5-0.5B-Instruct-4bit)",
    )
    parser.add_argument(
        "--suite",
        choices=["default"],
        default=None,
        help="Run a predefined benchmark suite",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Approximate prompt length in tokens (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum output tokens (default: 256)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Number of concurrent requests for batch benchmark",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum concurrency (defaults to --batch value)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--sweep",
        choices=["prefill"],
        default=None,
        help="Run a sweep (e.g. prefill-vs-decode)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save results as JSON to this file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a mock engine (no real model required)",
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=None,
        help="Number of requests (for batch mode, overrides --batch)",
    )
    return parser


async def _run_with_mock(args: argparse.Namespace) -> None:
    """Run benchmarks with a deterministic mock engine."""
    from unittest.mock import AsyncMock

    from yunshu_engine.benchmark import BenchmarkRunner

    # Build a mock engine that produces deterministic results
    engine = AsyncMock()

    # Track call count for varying output
    _call_count = {"n": 0}

    async def _fake_generate(prompt: str, max_tokens: int = 256, **kwargs):
        _call_count["n"] += 1
        _call_count["n"]
        return type("GenResult", (), {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": max_tokens,
            "text": "mock output " * max_tokens,
            "finished": True,
            "finish_reason": "length",
        })()

    async def _fake_stream(prompt: str, max_tokens: int = 256, temperature: float = 0.0, **kwargs):
        _call_count["n"] += 1
        for i in range(max_tokens):
            chunk = type("Chunk", (), {
                "new_text": "word ",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": i + 1,
                "finished": i == max_tokens - 1,
                "finish_reason": "length" if i == max_tokens - 1 else None,
            })()
            yield chunk

    engine.generate = _fake_generate
    engine.stream_generate = _fake_stream

    runner = BenchmarkRunner(engine)

    if args.sweep == "prefill":
        print("Running prefill-vs-decode sweep with mock engine...")
        points = await runner.bench_prefill_vs_decode(
            max_decode_tokens=args.max_tokens,
        )
        print(f"\n{'Prompt Len':>12} {'Prefill (ms)':>14} {'Decode (tok/s)':>16} {'Decode (ms)':>14}")
        print("-" * 60)
        for p in points:
            print(f"{p.prompt_length:>12} {p.prefill_time_ms:>14.1f} {p.decode_tps:>16.1f} {p.decode_time_ms:>14.1f}")
        if args.output:
            data = [{"prompt_length": p.prompt_length, "prefill_ms": p.prefill_time_ms,
                      "decode_tps": p.decode_tps} for p in points]
            args.output.write_text(json.dumps(data, indent=2))
            print(f"\nResults saved to {args.output}")
        return

    if args.batch is not None:
        num = args.num_requests or args.batch
        print(f"Running batch benchmark: {num} requests, concurrency={args.concurrency or num} (mock)...")
        result = await runner.bench_batch(
            prompts=num,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            prompt_tokens=args.prompt_len,
            temperature=args.temperature,
        )
        print(BenchmarkRunner.format_results(result))
        if args.output:
            args.output.write_text(json.dumps(result.to_dict(), indent=2))
            print(f"\nResults saved to {args.output}")
        return

    if args.suite:
        print(f"Running suite '{args.suite}' with mock engine on {args.model}...")
        suite = await runner.run_suite(args.model, suite_name=args.suite)
        print(BenchmarkRunner.format_results(suite))
        if args.output:
            args.output.write_text(json.dumps(suite.to_dict(), indent=2))
            print(f"\nResults saved to {args.output}")
        return

    # Default: single request
    print(f"Running single request: prompt_len={args.prompt_len}, max_tokens={args.max_tokens} (mock)...")
    result = await runner.bench_single_request(
        prompt_tokens=args.prompt_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(BenchmarkRunner.format_results(result))
    if args.output:
        args.output.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {args.output}")


async def _run_with_real_model(args: argparse.Namespace) -> None:
    """Run benchmarks with a real BatchedEngine."""
    from yunshu_engine.benchmark import BenchmarkRunner

    try:
        from yunshu_engine.batched_engine import BatchedEngine
    except ImportError:
        print("ERROR: yunshu_engine is not installed. Run `uv sync` first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}...")
    engine = BatchedEngine(model_name=args.model)

    try:
        await engine.start()
    except Exception as e:
        print(f"ERROR: Failed to load model '{args.model}': {e}", file=sys.stderr)
        print("\nHint: Make sure the model is available locally or via HuggingFace.", file=sys.stderr)
        print("Use --mock to test the benchmark harness without a real model.", file=sys.stderr)
        sys.exit(1)

    try:
        runner = BenchmarkRunner(engine)

        if args.sweep == "prefill":
            print("Running prefill-vs-decode sweep...")
            points = await runner.bench_prefill_vs_decode(
                max_decode_tokens=args.max_tokens,
            )
            print(f"\n{'Prompt Len':>12} {'Prefill (ms)':>14} {'Decode (tok/s)':>16} {'Decode (ms)':>14}")
            print("-" * 60)
            for p in points:
                print(f"{p.prompt_length:>12} {p.prefill_time_ms:>14.1f} {p.decode_tps:>16.1f} {p.decode_time_ms:>14.1f}")
            if args.output:
                data = [{"prompt_length": p.prompt_length, "prefill_ms": p.prefill_time_ms,
                          "decode_tps": p.decode_tps} for p in points]
                args.output.write_text(json.dumps(data, indent=2))
                print(f"\nResults saved to {args.output}")
            return

        if args.batch is not None:
            num = args.num_requests or args.batch
            print(f"Running batch benchmark: {num} requests, concurrency={args.concurrency or num}...")
            result = await runner.bench_batch(
                prompts=num,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                prompt_tokens=args.prompt_len,
                temperature=args.temperature,
            )
            print(BenchmarkRunner.format_results(result))
            if args.output:
                args.output.write_text(json.dumps(result.to_dict(), indent=2))
                print(f"\nResults saved to {args.output}")
            return

        if args.suite:
            print(f"Running suite '{args.suite}' on {args.model}...")
            suite = await runner.run_suite(args.model, suite_name=args.suite)
            print(BenchmarkRunner.format_results(suite))
            if args.output:
                args.output.write_text(json.dumps(suite.to_dict(), indent=2))
                print(f"\nResults saved to {args.output}")
            return

        # Default: single request
        print(f"Running single request: prompt_len={args.prompt_len}, max_tokens={args.max_tokens}...")
        result = await runner.bench_single_request(
            prompt_tokens=args.prompt_len,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(BenchmarkRunner.format_results(result))
        if args.output:
            args.output.write_text(json.dumps(result.to_dict(), indent=2))
            print(f"\nResults saved to {args.output}")

    finally:
        await engine.stop()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mock:
        asyncio.run(_run_with_mock(args))
    else:
        asyncio.run(_run_with_real_model(args))


if __name__ == "__main__":
    main()
