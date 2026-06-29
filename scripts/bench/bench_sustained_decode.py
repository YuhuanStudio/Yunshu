"""Sustained-decode thermal-decay bench — the `cool/` trend producer.

Single model, single stream, ONE long generation (no servers, no co-loaded
models — safe on the 36GB Mac). Measures whether decode tok/s DECAYS over a
sustained run: splits the generated tokens into a first vs last window and
reports the decay %. On this 30-core M3 Max raw decode is bandwidth-bound and
holds flat (~0% decay across 8 back-to-back iters) — a large positive
decay flags thermal throttling during pure decode, which the gpu_tflops matmul
tag (a compute-bound signal) does NOT capture.

Feeds docs/PERF_TREND.md `cool/` via perf_history.snapshot_from_kpis so the
signal is tracked every run instead of the single stale 0602 hand-built point.

Run:
  PYTHONPATH=. uv run python scripts/bench_sustained_decode.py
  PYTHONPATH=. YUNSHU_COOL_MODEL=./models/Qwen2.5-3B-Instruct-bf16 \
    YUNSHU_COOL_TOKENS=1200 uv run python scripts/bench_sustained_decode.py
"""
from __future__ import annotations

import json
import os
import time

MODEL = os.environ.get("YUNSHU_COOL_MODEL", "./models/Qwen2.5-3B-Instruct-bf16")
N_TOKENS = int(os.environ.get("YUNSHU_COOL_TOKENS", "1000"))


def main() -> int:
    if not os.path.exists(MODEL):
        print(f"SKIP: {MODEL} not mounted")
        return 0
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, tokenizer = load(MODEL)
    prompt = tokenizer.encode("Write a long, detailed essay about the history "
                              "of computing, covering every decade in depth.")
    prompt = mx.array(prompt)

    # Warm one step (compile + first-token prefill cost out of the measurement).
    g = generate_step(prompt, model, max_tokens=N_TOKENS)
    stamps: list[float] = []
    n = 0
    for _tok, _logprob in g:
        stamps.append(time.perf_counter())
        n += 1
        if n >= N_TOKENS:
            break
    mx.clear_cache()

    if len(stamps) < 40:
        print(f"SKIP: only {len(stamps)} tokens decoded (need >=40)")
        return 0

    # Per-step intervals → tok/s in the first vs last quarter of the run.
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    q = max(5, len(deltas) // 4)
    first_tps = q / sum(deltas[:q])
    last_tps = q / sum(deltas[-q:])
    # Positive decay = slower at the end (thermal throttle). Negative = warm-up.
    decay_pct = round((first_tps - last_tps) / first_tps * 100, 1)
    overall_tps = round(len(deltas) / sum(deltas), 1)

    name = os.path.basename(MODEL.rstrip("/")).replace("-MLX", "").replace("-Instruct", "")
    print(f"sustained decode: {len(stamps)} tok | overall {overall_tps} tok/s | "
          f"first-q {first_tps:.1f} → last-q {last_tps:.1f} tok/s | decay {decay_pct}%")
    summary = {
        "model": name, "tokens": len(stamps), "overall_tps": overall_tps,
        "first_q_tps": round(first_tps, 1), "last_q_tps": round(last_tps, 1),
        "sustained_decode_decay_pct": decay_pct,
    }
    print("@@COOL@@ " + json.dumps(summary))

    # Feed the cool/ trend (same append-only history as the rest of PERF_TREND).
    try:
        import sys
        # scripts/bench/ → perf_history is one level up at scripts/ (reorg).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from perf_history import snapshot_from_kpis
        kpis = {
            f"cool/{name}/decode_tps": overall_tps,
            f"cool/{name}/sustained_decode_decay_pct": decay_pct,
        }
        snapshot_from_kpis(kpis, source="bench_sustained_decode")
    except Exception as e:
        print(f"(perf-history feed skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
