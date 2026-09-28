"""Fused chunked prefill vs separate prefill: same greedy tokens? (small model)

Deterministic schedule on the VLM runner (no executor, one thread): rows A and B
start decoding, then a longer prompt C arrives and prefills while they decode.
Runs it with fused prefill off (reference) and at each budget, and reports per
request whether the greedy tokens match and where they first differ. Not a
benchmark (timings are incidental).

    python scripts/research/check_fused_prefill.py /Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16 \
        --budgets 64 256 --lines 300
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--lines", type=int, default=300, help="filler lines in prompt C")
    ap.add_argument("--kv", default="bf16", help="ragged KV precision")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument(
        "--apc",
        action="store_true",
        help="prefix cache on (checkpoints while C prefills), then C2 = C's prefix "
        "with another question arrives and prefills warm while A/B still decode",
    )
    ap.add_argument(
        "--image",
        default=None,
        help="make prompt C an image prompt (mRoPE positions through the fused step)",
    )
    ap.add_argument(
        "--float32",
        action="store_true",
        help="cast the model to float32 (stock KV, no ragged kernels): rounding "
        "differences from the packed matmul shapes shrink to ~1e-6, so a "
        "remaining token difference points at a logic error",
    )
    a = ap.parse_args()

    from mlx_vlm import load

    from yunshu_engine.kernels import ragged_kv
    from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner

    model, proc = load(a.model_dir)
    tok = proc.tokenizer
    if a.float32:
        import mlx.core as mx

        model.set_dtype(mx.float32)
        a.kv = None
    ragged_kv.install()
    ragged_kv.enable(None)

    def ids(text):
        s = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tok.encode(s, add_special_tokens=False)

    c_text = (
        "".join(
            f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n"
            for i in range(a.lines)
        )
        + "\nWhich sensor had the highest reading? Answer, then explain."
    )
    # A follow-up turn of C's conversation: its prompt extends C's (prefix hit).
    prompts_c2 = tok.encode(
        tok.apply_chat_template(
            [
                {"role": "user", "content": c_text},
                {"role": "assistant", "content": "Sensor 16."},
                {"role": "user", "content": "And the lowest?"},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        add_special_tokens=False,
    )

    prompts = {
        "A": ids("Write a long story about a lighthouse keeper."),
        "B": ids("Explain how a CPU cache works, in detail."),
        "C": ids(c_text),
    }
    media = {}
    if a.image:
        text = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this chart in detail."},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    print({k: len(v) for k, v in prompts.items()}, flush=True)

    def run(budget):
        apc = None
        if a.apc:
            from mlx_vlm.apc import APCManager, semantic_extra_hash

            apc = APCManager(
                num_blocks=512, block_size=16, overrides={"memory_max_gb": 2.0}
            )
            salt = semantic_extra_hash(
                image_hash=0,
                media={"audio": None, "video": None},
                model=model.language_model,
                processor=proc,
            )
        r = VLMBatchRunner(
            model, proc, apc_manager=apc, apc_semantic_hash=salt if apc else None
        )
        r.stop_tokens = {tok.eos_token_id}
        r.ragged_kv = a.kv
        r.fused_prefill_tokens = budget
        out = {k: [] for k in prompts}
        gens = {}
        ab_tokens = a.max_tokens * (3 if a.apc else 1)
        gens["A"] = r.iter_tokens(prompts["A"], max_tokens=ab_tokens)
        out["A"].append(next(gens["A"]))
        gens["B"] = r.iter_tokens(prompts["B"], max_tokens=ab_tokens)
        out["B"].append(next(gens["B"]))
        for _ in range(5):
            out["A"].append(next(gens["A"]))
            out["B"].append(next(gens["B"]))
        t = time.perf_counter()
        if a.image:
            c_ids, c_kwargs, _ = r.prepare_media(text, [a.image])
            media["C"] = len(c_ids)
            gens["C"] = r.iter_tokens(
                c_ids, max_tokens=a.max_tokens // 2, prompt_kwargs=c_kwargs
            )
        else:
            gens["C"] = r.iter_tokens(prompts["C"], max_tokens=a.max_tokens // 2)
        out["C"].append(next(gens["C"]))
        ttft = time.perf_counter() - t
        if a.apc:
            out["C"].extend(gens.pop("C"))
            out["C2"] = []
            stats = RunStats()
            gens["C2"] = r.iter_tokens(
                prompts_c2, max_tokens=a.max_tokens // 2, stats=stats
            )
            out["C2"].append(next(gens["C2"]))
            print(f"  C2 cached tokens {stats.cached_tokens}", flush=True)
        for k, g in gens.items():
            out[k].extend(g)
        return out, ttft

    from yunshu_engine import fused_prefill

    ref, t_ref = run(0)
    if media:
        print(f"image prompt C: {media['C']} tokens", flush=True)
    for budget in a.budgets:
        before = fused_prefill.stats()
        got, t = run(budget)
        after = fused_prefill.stats()
        print(
            f"budget={budget}: fused steps {after['steps'] - before['steps']}, "
            f"prefill tokens in them {after['tokens'] - before['tokens']}",
            flush=True,
        )
        for k in ref:
            x, y = ref[k], got[k]
            diff = next((i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q), None)
            print(
                f"budget={budget} {k}: tokens {len(x)}/{len(y)} same={x == y} "
                f"first_diff={diff}",
                flush=True,
            )
        print(f"budget={budget} C first token {t_ref:.2f}s -> {t:.2f}s", flush=True)


if __name__ == "__main__":
    main()
