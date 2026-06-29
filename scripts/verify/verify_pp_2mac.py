"""Turnkey Pipeline-Parallel correctness validation — the final 2-Mac step for PP.

CORRECTNESS TRUTH: the ONLY correct pipeline parallelism is mlx-lm NATIVE
(sharded_load(pipeline_group=group) → model.pipeline() + the model's pipeline-aware forward
that does recv→layers→send→all_gather→norm so EVERY rank produces identical logits). This
exists ONLY for models with PipelineMixin — deepseek_v3 / deepseek_v2 / deepseek_v32 /
glm4_moe / glm4_moe_lite / ministral3. For any other model (Qwen, Llama, …) mlx-lm REFUSES
("model does not support pipelining"), and Yunshu's old custom layer-wrapper (PipelineLastLayer)
produced WRONG logits on non-last ranks → made it fail-loud. So PP is: native-correct
where supported, safely refused otherwise. This script validates the native path.

Two-step (works on ONE machine for self-test + on 2 Macs for the real test), TP-style:

  # 1. Reference (single machine, FULL model):
  PYTHONPATH=. uv run python scripts/verify/verify_pp_2mac.py reference \
      --model <a PipelineMixin model dir> --out /tmp/pp_ref.json

  # 2a. Self-test on ONE machine (world_size=1 → trivial pipeline → must match reference):
  PYTHONPATH=. uv run python scripts/verify/verify_pp_2mac.py distributed \
      --model <same> --ref /tmp/pp_ref.json

  # 2b. REAL test on 2 Macs:
  mlx.launch --hostfile hosts.txt -n 2 -- \
      python scripts/verify/verify_pp_2mac.py distributed --model <same> --ref /tmp/pp_ref.json

A model WITHOUT PipelineMixin reports cleanly that PP is N/A for it (use tensor-parallel
instead) — that is the correct, honest outcome, not a script failure.
"""
import argparse
import json
import sys

PROMPT = "The capital of France is"
N_TOKENS = 24


def _greedy(model, tokenizer, prompt: str, n: int) -> list[int]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    ids = tokenizer.encode(prompt)
    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)[:, -1, :]
    out: list[int] = []
    for _ in range(n):
        nt = int(mx.argmax(logits, axis=-1).item())
        out.append(nt)
        logits = model(mx.array([[nt]]), cache=cache)[:, -1, :]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "distributed"])
    ap.add_argument("--model", required=True,
                    help="a PipelineMixin model dir (deepseek_v3/glm4_moe/ministral-class)")
    ap.add_argument("--out", default="/tmp/pp_ref.json")
    ap.add_argument("--ref", default="/tmp/pp_ref.json")
    a = ap.parse_args()

    if a.mode == "reference":
        from mlx_lm.utils import load
        model, tok = load(a.model)
        tokens = _greedy(model, tok, PROMPT, N_TOKENS)
        with open(a.out, "w") as f:
            json.dump({"model": a.model, "prompt": PROMPT, "tokens": tokens}, f)
        print(f"reference: {len(tokens)} greedy tokens → {a.out}  first8={tokens[:8]}")
        return 0

    import mlx.core as mx
    from mlx_lm.utils import sharded_load

    group = mx.distributed.init()
    ws, rank = group.size(), group.rank()
    try:
        model, tok = sharded_load(a.model, pipeline_group=group)
    except ValueError as e:
        if "does not support pipelining" in str(e):
            print(f"ℹ️  {a.model} has no PipelineMixin → native PP is N/A for it (correct, "
                  "not a bug). mlx-lm refuses; use tensor-parallel (verify_tp_2mac.py) instead.")
            return 0
        raise

    tokens = _greedy(model, tok, PROMPT, N_TOKENS)
    if rank != 0:
        return 0  # only rank 0 holds the all_gathered final output to compare

    with open(a.ref) as f:
        ref_tokens = json.load(f)["tokens"]
    print(f"distributed PP: world_size={ws} rank={rank}")
    print(f"  sharded first8: {tokens[:8]}")
    print(f"  ref     first8: {ref_tokens[:8]}")
    if tokens != ref_tokens:
        n_match = sum(1 for x, y in zip(tokens, ref_tokens) if x == y)
        print(f"\n❌ PP MISMATCH — sharded != single-node ({n_match}/{len(ref_tokens)}). "
              "The all_gather/pipeline math is wrong (or a non-PipelineMixin model slipped through).")
        return 1
    tag = "(world_size=1 self-test)" if ws == 1 else f"(REAL {ws}-rank PP)"
    print(f"\n✅ PP correct {tag}: native-pipeline greedy == single-node, token-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
