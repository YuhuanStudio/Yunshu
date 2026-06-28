"""Run bench_comprehensive.py across all models (sequential subprocesses) and
render the combined length / cache-hit-ratio / concurrency dataset + dump JSON.

Run: PYTHONPATH=. uv run python scripts/bench/bench_comprehensive_all.py
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [
    "./models/Qwen3.5-0.8B-MLX-bf16",
    "./models/Qwen3.5-2B-MLX-bf16",
    "./models/Qwen3.5-9B-MLX-4bit",
    "./models/Qwen2.5-3B-Instruct-bf16",
    "./models/gemma-4-e4b-it-bf16",
    "./models/GLM-OCR-bf16",
    # Qwen3-Omni-30B intentionally omitted from the concurrency sweep: the 30B MoE
    # GPU-hangs under N=32 concurrent load on M3 Max. Its tier data is in bench_all.
]


def run(model):
    env = dict(os.environ, PYTHONPATH="reference/mlx-vlm:.", YUNSHU_BENCH_MODEL=model)
    try:
        p = subprocess.run([sys.executable, "scripts/bench/bench_comprehensive.py"],
                           env=env, cwd=REPO, capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    m = re.search(r"@@COMPREHENSIVE@@ (\{.*\})", p.stdout + p.stderr)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {"status": "PARSE ERR"}
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-2:])
    return {"status": f"NO RESULT ({tail[:80]})"}


def main():
    models = sys.argv[1:] or MODELS
    results = []
    for mp in models:
        if not os.path.exists(os.path.join(REPO, mp)):
            print(f"SKIP (missing) {mp}")
            continue
        print(f"\n>>> {mp}", flush=True)
        r = run(mp)
        r["_path"] = mp
        results.append(r)
        print("   ", r.get("status", "ok"))
    json.dump(results, open("/tmp/bench_comprehensive.json", "w"), indent=2)

    # ── render ──
    def tbl(title, rows_key, cols):
        print(f"\n{'='*100}\n{title}\n{'='*100}")
        for r in results:
            if rows_key not in r:
                continue
            print(f"\n### {r['model']} ({r.get('engine','?')})")
            print(" ".join(h.ljust(w) for h, w, _ in cols))
            for row in r[rows_key]:
                print(" ".join(str(row.get(k, '-')).ljust(w) for _, w, k in cols))

    tbl("PROMPT-LENGTH sweep — TTFT + prefill tok/s + pure decode tok/s (cold, single-req)",
        "length_sweep",
        [("prompt_tok", 11, "prompt_tok"), ("TTFT_ms", 9, "ttft_ms"),
         ("prefill_t/s", 12, "prefill_tps"), ("decode_t/s", 11, "decode_tps")])
    tbl("CACHE-HIT-RATIO sweep — reuse benefit vs shared-prefix fraction",
        "hit_ratio_sweep",
        [("hit_ratio", 10, "hit_ratio"), ("prompt_tok", 11, "prompt_tok"),
         ("cached", 8, "cached"), ("TTFT_ms", 9, "ttft_ms"),
         ("speedup", 9, "speedup_vs_cold_base")])
    tbl("CONCURRENCY sweep — fast path aggregate tok/s (serialises; engine-loop batching in bench_matrix.py)",
        "concurrency_sweep",
        [("N", 5, "N"), ("agg_tok/s", 10, "agg_tps"), ("mean_TTFT_ms", 13, "mean_ttft_ms")])
    print(f"\nfull json -> /tmp/bench_comprehensive.json")


if __name__ == "__main__":
    main()
