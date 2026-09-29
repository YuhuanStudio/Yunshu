"""Round driver in-process: parity and throughput by concurrent rows.

Loads a Qwen3.5-family checkpoint, converts its projections to lane matmuls
and builds the round driver with the checkpoint's MTP head. Then:

1. parity: every prompt's greedy tokens alone without drafts are the
   reference; alone with MTP drafts, all rows together with drafts, and rows
   joining one by one must reproduce them exactly;
2. rows sweep: ``--rows`` concurrent greedy rows (distinct prompts), with and
   without drafts: aggregate tok/s, steps, drafted / accepted, per-row tok/s.

    python scripts/research/sweep_round_driver.py <ckpt> --rows 1 2 4 8 \
        --tokens 256 --output runs/round-driver-sweep.jsonl

``--quantize`` quantizes a bf16 checkpoint to 4 bits in memory (small models
for smoke tests); ``--context`` prepends that many filler tokens.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

PROMPTS = [
    "Write a Python function that merges two sorted lists, with a docstring.",
    "Explain step by step why the sky looks blue during the day.",
    "List twelve European capitals and one fact about each.",
    "Write a short story about a lighthouse keeper who finds a map.",
    "Describe how a hash map handles collisions, with an example.",
    "Give a JSON object describing three fictional books.",
    "Summarize the causes of the French Revolution in five points.",
    "Solve: a train leaves at 3pm at 60 km/h; another at 4pm at 80 km/h. When does the second catch up?",
]


def load(ckpt: str, quantize: bool):
    import mlx.nn as nn
    from mlx_vlm import load as vlm_load

    import yunshu_engine.mlxvlm_mtp as m
    from yunshu_engine.kernels import lane_linear

    model, processor = vlm_load(ckpt)
    lm = model.language_model
    if quantize:
        nn.quantize(
            lm,
            group_size=64,
            bits=4,
            class_predicate=lambda _p, mod: (
                isinstance(mod, nn.Linear) and mod.weight.shape[-1] % 64 == 0
            ),
        )
    lanes = lane_linear.convert(lm)
    if lm.args.tie_word_embeddings:
        lm._yunshu_lane_head = lane_linear.lane_head(lm.model.embed_tokens)
    extra = Path(ckpt) / "mtp-weights.safetensors"
    if extra.exists():
        weights = mx.load(str(extra))
        m._load_mtp_head_tensors = lambda _p: {
            k.removeprefix("mtp."): v for k, v in weights.items()
        }
    drafter = m._load_drafter_in_memory(ckpt)
    return model, processor, drafter, lanes


def filler_text(tok, context: int) -> str:
    """Filler of ``context`` tokens under ``tok`` (numbered notes, cut at the
    token count). "note 1234." is several tokens, so sizing the filler by
    characters or words overshoots: 32768 once meant ~76K-token prompts."""
    if context <= 0:
        return ""
    words, ids = context, []
    while len(ids) < context:
        ids = tok.encode(
            " ".join(f"note {i}." for i in range(words)), add_special_tokens=False
        )
        words *= 2
    return tok.decode(ids[:context])


def encode(tok, text, context):
    msgs = [{"role": "user", "content": text}]
    if context:
        msgs[0]["content"] = filler_text(tok, context) + "\n\n" + text
    text = tok.apply_chat_template(
        msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    return list(tok.encode(text, add_special_tokens=False))


def run(model, drafter, prompts, tokens, stop, stagger=False):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(model, drafter=drafter, stop_tokens=stop)
    out = {i: [] for i in range(len(prompts))}
    first = {}
    todo = list(range(len(prompts)))
    t0 = time.perf_counter()
    if not stagger:
        for i in todo:
            d.add(Request(prompts[i], tokens, handle=i))
        todo = []
    steps = 0
    while d.busy() or todo:
        if todo and steps % 2 == 0:
            d.add(
                Request(
                    prompts[todo.pop(0)], tokens, handle=len(prompts) - len(todo) - 1
                )
            )
        for e in d.step():
            out[e.handle].append(e.token)
            first.setdefault(e.handle, time.perf_counter() - t0)
        steps += 1
    wall = time.perf_counter() - t0
    return [out[i] for i in range(len(prompts))], d, wall, first


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument(
        "--parity-rows",
        type=int,
        default=None,
        help="prompts in the parity check (default: max --rows); each runs 5 ways",
    )
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    model, processor, drafter, lanes = load(a.ckpt, a.quantize)
    tok = processor.tokenizer
    extra = getattr(tok, "eos_token_ids", None) or []
    stop = {tok.eos_token_id} | set([extra] if isinstance(extra, int) else extra)
    prompts = [encode(tok, p, a.context) for p in PROMPTS]
    n_parity = a.parity_rows or max(a.rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:

        def put(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)

        put(
            {
                "kind": "meta",
                "ckpt": a.ckpt,
                "tokens": a.tokens,
                "context": a.context,
                "prompt_tokens": [len(p) for p in prompts],
                # prefill work of the whole sweep: parity (5 runs of n
                # prompts) + each rows setting with and without drafts
                "prefill_tokens_total": 5 * sum(len(p) for p in prompts[:n_parity])
                + 2 * sum(sum(len(p) for p in prompts[:r]) for r in a.rows),
                "lane_converted": lanes["converted"],
                "lane_skipped": lanes["skipped"],
            }
        )
        n = a.parity_rows or max(a.rows)
        ref = [run(model, None, [p], a.tokens, stop)[0][0] for p in prompts[:n]]
        checks = {
            "alone_mtp": [
                run(model, drafter, [p], a.tokens, stop)[0][0] for p in prompts[:n]
            ],
            "batch_mtp": run(model, drafter, prompts[:n], a.tokens, stop)[0],
            "stagger_mtp": run(
                model, drafter, prompts[:n], a.tokens, stop, stagger=True
            )[0],
            "batch_ar": run(model, None, prompts[:n], a.tokens, stop)[0],
        }
        put(
            {
                "kind": "parity",
                **{k: v == ref for k, v in checks.items()},
                "first_diff": {
                    k: next(
                        (
                            (i, j)
                            for i, (x, y) in enumerate(zip(v, ref, strict=True))
                            for j, (p, q) in enumerate(zip(x, y, strict=False))
                            if p != q
                        ),
                        None,
                    )
                    for k, v in checks.items()
                },
            }
        )
        for rows in a.rows:
            for drafts in (False, True):
                outs, d, wall, first = run(
                    model, drafter if drafts else None, prompts[:rows], a.tokens, stop
                )
                total = sum(len(o) for o in outs)
                put(
                    {
                        "kind": "rows",
                        "rows": rows,
                        "mtp": drafts,
                        "tokens": total,
                        "wall_s": round(wall, 3),
                        "aggregate_tps": round(total / wall, 1),
                        "per_row_tps": round(total / wall / rows, 1),
                        "steps": d.steps,
                        "drafted": d.drafted,
                        "accepted": d.accepted,
                        "depth_acceptance": [
                            round(l / max(n, 1), 2)
                            for n, l in zip(
                                d.depth_drafted, d.depth_landed, strict=True
                            )
                        ],
                        "chain_ms": round(d.chain_ms, 2),
                        "mean_first_s": round(sum(first.values()) / len(first), 3),
                    }
                )


if __name__ == "__main__":
    main()
