"""Latency benchmark — end-to-end request latency against running server.

Measures P50/P95/P99 latency and tokens/sec for the inference server.

Run: PYTHONPATH=. uv run python bench/latency/run.py
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np


def run_latency_benchmark(
    url: str = "http://localhost:8000",
    model: str = "Qwen3.5-9B-MLX-4bit",
    num_requests: int = 20,
    prompt_tokens: int = 32,
    max_tokens: int = 64,
    output_dir: str = "bench/latency",
):
    print(f"=== Latency Benchmark ===")
    print(f"Server: {url}")
    print(f"Model: {model}")
    print(f"Requests: {num_requests}")
    print(f"Prompt tokens: ~{prompt_tokens}, Max output: {max_tokens}")
    print()

    prompt = "Hello! Tell me a short story about a robot. " * (prompt_tokens // 10 + 1)

    latencies = []
    comp_tokens = []
    errors = 0

    for i in range(num_requests):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.7,
        }

        t0 = time.perf_counter()
        try:
            resp = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
            elapsed = time.perf_counter() - t0

            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                ct = usage.get("completion_tokens", max_tokens)
                latencies.append(elapsed)
                comp_tokens.append(ct)
                print(f"  [{i+1:3d}/{num_requests}] {elapsed:.2f}s, {ct} tokens, {ct/elapsed:.1f} tok/s")
            else:
                errors += 1
                print(f"  [{i+1:3d}/{num_requests}] ERROR: {resp.status_code}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1:3d}/{num_requests}] ERROR: {e}")

    if not latencies:
        print("\nNo successful requests!")
        return

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)]
    avg = statistics.mean(latencies)
    total_tokens = sum(comp_tokens)
    avg_tok_per_s = statistics.mean([c / l for c, l in zip(comp_tokens, latencies)])

    print(f"\n=== Results ===")
    print(f"Successful: {len(latencies)}/{num_requests}, Errors: {errors}")
    print(f"Avg latency: {avg:.3f}s")
    print(f"P50: {p50:.3f}s, P95: {p95:.3f}s, P99: {p99:.3f}s")
    print(f"Avg throughput: {avg_tok_per_s:.1f} tok/s")
    print(f"Total tokens: {total_tokens}")

    report = {
        "url": url,
        "model": model,
        "num_requests": num_requests,
        "successful": len(latencies),
        "errors": errors,
        "avg_latency_s": avg,
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "p99_latency_s": p99,
        "avg_tokens_per_second": avg_tok_per_s,
        "total_tokens": total_tokens,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path / 'report.json'}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen3.5-9B-MLX-4bit")
    parser.add_argument("-n", "--requests", type=int, default=20)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    run_latency_benchmark(args.url, args.model, args.requests, args.prompt_tokens, args.max_tokens)
