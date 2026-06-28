"""Throughput benchmark — concurrent request throughput against running server.

Measures tokens/sec at different concurrency levels.

Run: PYTHONPATH=. uv run python bench/throughput/run.py
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


def _single_request(
    url: str, model: str, prompt: str, max_tokens: int, idx: int
) -> dict:
    """Execute a single request and return timing info."""
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
            ct = data.get("usage", {}).get("completion_tokens", max_tokens)
            return {"elapsed": elapsed, "tokens": ct, "success": True, "idx": idx}
        return {"elapsed": elapsed, "tokens": 0, "success": False, "idx": idx}
    except Exception as e:
        return {"elapsed": time.perf_counter() - t0, "tokens": 0, "success": False, "idx": idx, "error": str(e)}


def run_throughput_benchmark(
    url: str = "http://localhost:8000",
    model: str = "Qwen3.5-9B-MLX-4bit",
    concurrency_levels: list[int] | None = None,
    num_requests: int = 10,
    prompt_tokens: int = 128,
    max_tokens: int = 128,
    output_dir: str = "bench/throughput",
):
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4]

    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10 + 1)

    print(f"=== Throughput Benchmark ===")
    print(f"Server: {url}, Model: {model}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Requests per level: {num_requests}")
    print()

    results = []

    for conc in concurrency_levels:
        print(f"--- Concurrency: {conc} ---")
        wall_start = time.perf_counter()

        total_tokens = 0
        successes = 0
        with ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [
                executor.submit(_single_request, url, model, prompt, max_tokens, i)
                for i in range(num_requests)
            ]
            for f in as_completed(futures):
                r = f.result()
                if r["success"]:
                    total_tokens += r["tokens"]
                    successes += 1

        wall_time = time.perf_counter() - wall_start
        req_per_s = successes / wall_time if wall_time > 0 else 0
        tok_per_s = total_tokens / wall_time if wall_time > 0 else 0

        print(f"  Wall time: {wall_time:.2f}s")
        print(f"  Success: {successes}/{num_requests}")
        print(f"  Requests/s: {req_per_s:.1f}")
        print(f"  Tokens/s: {tok_per_s:.1f}")

        results.append({
            "concurrency": conc,
            "wall_time_s": round(wall_time, 3),
            "successful_requests": successes,
            "total_tokens": total_tokens,
            "requests_per_second": round(req_per_s, 2),
            "tokens_per_second": round(tok_per_s, 1),
        })

    peak = max(results, key=lambda r: r["tokens_per_second"]) if results else None
    print(f"\nPeak throughput: {peak['tokens_per_second']} tok/s at concurrency={peak['concurrency']}")

    report = {
        "url": url,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "results": results,
        "peak_throughput_tok_s": peak["tokens_per_second"] if peak else 0,
        "peak_concurrency": peak["concurrency"] if peak else 0,
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
    parser.add_argument("-c", "--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("-n", "--requests", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    run_throughput_benchmark(
        args.url, args.model, args.concurrency, args.requests,
        args.prompt_tokens, args.max_tokens,
    )
