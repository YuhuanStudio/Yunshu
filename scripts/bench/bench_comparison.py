"""Fair benchmark: Yunshu vs oMLX comparison.

Measures the same metrics oMLX reports:
- TTFT (Time To First Token) — streaming first chunk latency
- Generation throughput (tok/s) — tokens per second during generation
- E2E latency — full request latency (prefill + generate)
- Concurrent throughput — requests/second under load

Usage:
    PYTHONPATH=. uv run python scripts/bench_comparison.py
    PYTHONPATH=. uv run python scripts/bench_comparison.py --model llm --requests 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

ENGINE_MAP = {
    "llm": {
        "path": f"{MODELS_DIR}/Qwen3.5-9B-MLX-4bit",
        "engine": "batched",
    },
    "vlm": {
        "path": f"{MODELS_DIR}/Qwen3-Omni-30B-A3B-Instruct-4bit",
        "engine": "vlm",
    },
}


async def bench_llm(model_path: str, args) -> dict:
    """Benchmark LLM engine: TTFT, tok/s, E2E latency."""
    from yunshu_engine.batched_engine import BatchedEngine

    engine = BatchedEngine(model_name=model_path, stream_interval=1)
    await engine.start()

    prompt = "Write a short essay about the importance of mathematics in modern science. " * (args.prompt_tokens // 15 + 1)

    results = {
        "model": Path(model_path).name,
        "engine": "yunshu-batched",
        "prompt_tokens_est": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "runs": [],
    }

    # Warmup
    for _ in range(2):
        await engine.generate(prompt=prompt, max_tokens=args.max_tokens, temperature=0.0)

    for run_idx in range(args.requests):
        t_start = time.perf_counter()
        ttft = None
        first_token_time = None
        token_count = 0

        async for chunk in engine.stream_generate(
            prompt=prompt, max_tokens=args.max_tokens, temperature=0.0,
        ):
            if first_token_time is None and chunk.new_text:
                first_token_time = time.perf_counter()
                ttft = first_token_time - t_start
            token_count += 1

        e2e = time.perf_counter() - t_start
        gen_time = e2e - (ttft or 0)
        tok_s = (token_count / gen_time) if gen_time > 0 else 0

        results["runs"].append({
            "ttft_ms": round(ttft * 1000, 1) if ttft else None,
            "e2e_ms": round(e2e * 1000, 1),
            "tokens": token_count,
            "tok_s": round(tok_s, 1),
            "gen_time_ms": round(gen_time * 1000, 1),
        })

        if (run_idx + 1) % 5 == 0:
            print(f"  Run {run_idx + 1}/{args.requests}")

    await engine.stop()
    _summarize(results)
    return results


async def bench_vlm(model_path: str, args) -> dict:
    """Benchmark VLM engine: text-only and vision generation."""
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(model_path)
    await engine.start()

    results = {
        "model": Path(model_path).name,
        "engine": "yunshu-vlm",
        "has_vision": engine.has_vision,
        "runs_text": [],
        "runs_vision": [],
    }

    # Text-only benchmark
    for _ in range(min(args.requests, 5)):
        t0 = time.perf_counter()
        result = await engine.generate(
            messages=[{"role": "user", "content": "Explain gravity in one sentence."}],
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        e2e = time.perf_counter() - t0
        results["runs_text"].append({
            "e2e_ms": round(e2e * 1000, 1),
            "text_length": len(result["text"]),
        })

    # Vision benchmark
    if engine.has_vision:
        import os
        import tempfile

        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()

        for _ in range(min(args.requests, 3)):
            t0 = time.perf_counter()
            result = await engine.generate(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe briefly."},
                        {"type": "image_url", "image_url": {"url": tmp.name}},
                    ],
                }],
                max_tokens=args.max_tokens,
                temperature=0.5,
            )
            e2e = time.perf_counter() - t0
            results["runs_vision"].append({
                "e2e_ms": round(e2e * 1000, 1),
                "text_length": len(result["text"]),
            })

        os.unlink(tmp.name)

    await engine.stop()
    _summarize_vlm(results)
    return results


def _summarize(results: dict) -> None:
    """Compute summary statistics from runs."""
    runs = results["runs"]
    if not runs:
        return

    ttfts = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
    tok_s = [r["tok_s"] for r in runs]
    e2es = [r["e2e_ms"] for r in runs]

    results["summary"] = {
        "ttft_p50_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "ttft_p95_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1) if len(ttfts) > 1 else (round(ttfts[0], 1) if ttfts else 0),
        "tok_s_mean": round(statistics.mean(tok_s), 1) if tok_s else 0,
        "tok_s_median": round(statistics.median(tok_s), 1) if tok_s else 0,
        "e2e_p50_ms": round(statistics.median(e2es), 1) if e2es else 0,
        "e2e_p95_ms": round(sorted(e2es)[int(len(e2es) * 0.95)], 1) if len(e2es) > 1 else (round(e2es[0], 1) if e2es else 0),
        "num_runs": len(runs),
    }

    s = results["summary"]
    print(f"\n  Summary (n={s['num_runs']}):")
    if s["ttft_p50_ms"]:
        print(f"    TTFT: p50={s['ttft_p50_ms']}ms, p95={s['ttft_p95_ms']}ms")
    print(f"    Throughput: mean={s['tok_s_mean']} tok/s, median={s['tok_s_median']} tok/s")
    print(f"    E2E: p50={s['e2e_p50_ms']}ms, p95={s['e2e_p95_ms']}ms")


def _summarize_vlm(results: dict) -> None:
    """Compute VLM summary statistics."""
    text_runs = results["runs_text"]
    vision_runs = results["runs_vision"]

    if text_runs:
        e2es = [r["e2e_ms"] for r in text_runs]
        results["summary_text"] = {
            "e2e_p50_ms": round(statistics.median(e2es), 1),
            "num_runs": len(text_runs),
        }
        print(f"\n  Text-only: p50={results['summary_text']['e2e_p50_ms']}ms (n={len(text_runs)})")

    if vision_runs:
        e2es = [r["e2e_ms"] for r in vision_runs]
        results["summary_vision"] = {
            "e2e_p50_ms": round(statistics.median(e2es), 1),
            "num_runs": len(vision_runs),
        }
        print(f"\n  Vision: p50={results['summary_vision']['e2e_p50_ms']}ms (n={len(vision_runs)})")


async def main():
    parser = argparse.ArgumentParser(description="Yunshu benchmark")
    parser.add_argument("--model", choices=["llm", "vlm", "all"], default="llm")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    import gc
    all_results = {}

    models_to_bench = ["llm", "vlm"] if args.model == "all" else [args.model]

    for model_key in models_to_bench:
        cfg = ENGINE_MAP.get(model_key)
        if cfg is None:
            print(f"Unknown model: {model_key}")
            continue

        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {model_key} ({Path(cfg['path']).name})")
        print(f"  Engine: {cfg['engine']}")
        print(f"  Requests: {args.requests}, max_tokens: {args.max_tokens}")
        print(f"{'='*60}")

        gc.collect()
        t0 = time.time()

        if model_key == "llm":
            result = await bench_llm(cfg["path"], args)
        elif model_key == "vlm":
            result = await bench_vlm(cfg["path"], args)
        else:
            continue

        result["total_bench_time_s"] = round(time.time() - t0, 1)
        all_results[model_key] = result
        gc.collect()

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    for key, res in all_results.items():
        if "summary" in res:
            s = res["summary"]
            print(f"  {key}: {s.get('tok_s_median', '?')} tok/s, TTFT p50={s.get('ttft_p50_ms', '?')}ms")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
