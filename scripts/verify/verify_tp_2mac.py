"""Turnkey Tensor-Parallel correctness validation — the final 2-Mac step for TP.

TP shards a model's attention/MLP weights across ranks and all-reduces. It has only ever run
at world_size=1 (an identity no-op that proves nothing). This validates the REAL property:
the sharded model's greedy output is TOKEN-IDENTICAL to the single-node model.

Two-step (so it works on ONE machine for self-test AND on 2 Macs for the real test):

  # 1. Reference (any single machine) — greedy-decode the fixed prompt with the FULL model:
  PYTHONPATH=. uv run python scripts/verify/verify_tp_2mac.py reference --out /tmp/tp_ref.json

  # 2a. Self-test on ONE machine (world_size=1 → identity shard → must match reference):
  PYTHONPATH=. uv run python scripts/verify/verify_tp_2mac.py distributed --ref /tmp/tp_ref.json

  # 2b. REAL test on 2 Macs (sharded across 2 ranks → must STILL match reference):
  mlx.launch --hostfile hosts.txt -n 2 -- \
      python scripts/verify/verify_tp_2mac.py distributed --ref /tmp/tp_ref.json

If rank-0's sharded tokens == the single-node reference, TP is correct. A mismatch means the
shard/all-reduce math is wrong. world_size=1 self-test validates the script + comparison logic
without a second machine; the 2-rank run is the actual TP correctness proof.
"""
import argparse
import json
import sys

MODEL = "models/Qwen3.5-0.8B-MLX-bf16"
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
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="/tmp/tp_ref.json")
    ap.add_argument("--ref", default="/tmp/tp_ref.json")
    a = ap.parse_args()

    if a.mode == "reference":
        from mlx_lm.utils import load
        model, tok = load(a.model)
        tokens = _greedy(model, tok, PROMPT, N_TOKENS)
        with open(a.out, "w") as f:
            json.dump({"model": a.model, "prompt": PROMPT, "tokens": tokens}, f)
        print(f"reference: {len(tokens)} greedy tokens → {a.out}")
        print(f"  first 8: {tokens[:8]}")
        return 0

    # distributed mode — sharded load across ranks (world_size=1 = identity self-test)
    import mlx.core as mx
    from mlx_lm.utils import sharded_load

    group = mx.distributed.init()
    ws, rank = group.size(), group.rank()
    model, tok = sharded_load(a.model, tensor_group=group)
    tokens = _greedy(model, tok, PROMPT, N_TOKENS)

    if rank != 0:
        return 0  # only rank 0 holds the combined (all-reduced) output to compare

    with open(a.ref) as f:
        ref = json.load(f)
    ref_tokens = ref["tokens"]
    print(f"distributed: world_size={ws} rank={rank}")
    print(f"  sharded first 8: {tokens[:8]}")
    print(f"  ref     first 8: {ref_tokens[:8]}")
    if tokens != ref_tokens:
        n_match = sum(1 for x, y in zip(tokens, ref_tokens) if x == y)
        print(f"\n❌ TP MISMATCH — sharded != single-node ({n_match}/{len(ref_tokens)} tokens match). "
              "The shard/all-reduce math is WRONG.")
        return 1
    tag = "(world_size=1 identity self-test)" if ws == 1 else f"(REAL {ws}-rank TP)"
    print(f"\n✅ TP correct {tag}: sharded greedy == single-node, token-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
