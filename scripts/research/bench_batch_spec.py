"""Aggregate decode throughput of one upstream BatchGenerator batch: AR vs MTP.

B distinct greedy prompts are inserted together (a cohort), so this measures
the ceiling of multi-row speculative decoding against plain batched decode on
the same kernels, before building a scheduler around either.

    python scripts/research/bench_batch_spec.py <model_dir> --batches 1 2 4 8 \
        --blocks 0 3 6 --kernels exact --output runs/batch-spec.jsonl

``--kernels``: exact (Yunshu default verify kernels) | none (upstream).
Block 0 = AR.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory  # noqa: E402
from yunshu_engine.mrope import clear_rope_state  # noqa: E402

TOPICS = [
    "an LRU cache class",
    "a trie with insert and prefix search",
    "a thread-safe bounded queue",
    "a JSON pretty printer",
    "a matrix class with multiplication",
    "an interval tree",
    "a rate limiter (token bucket)",
    "a union-find with path compression",
    "a min-heap priority queue",
    "a simple tokenizer for arithmetic expressions",
    "a ring buffer",
    "a Dijkstra shortest path function",
    "a markdown table formatter",
    "an event bus with subscribe/publish",
    "a retry decorator with backoff",
    "a CSV parser handling quotes",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--blocks", type=int, nargs="*", default=[0, 3, 6])
    ap.add_argument("--kernels", default="exact", choices=["exact", "none"])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--pack", action="store_true", help="oMLX NAX packed projections")
    ap.add_argument(
        "--dflash", default=None, help="external DFlash drafter dir instead of MTP"
    )
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")

    if a.kernels != "none":
        from yunshu_engine.kernels import omlx as yk

        yk.apply()
        if a.kernels == "exact":
            from yunshu_engine.kernels.verify_select import install as install_streamed5

            install_streamed5()
    model, processor = load(a.model_dir)
    tok = processor.tokenizer
    packed = 0
    if a.pack:
        from yunshu_engine.kernels.omlx import pack_projections

        packed = pack_projections(model)
    if a.dflash:
        from mlx_vlm.speculative.drafters import load_drafter

        drafter, draft_kind = load_drafter(a.dflash, kind="dflash")
    else:
        drafter, draft_kind = _load_drafter_in_memory(a.model_dir), "mtp"
    lm = model.language_model

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    emit(
        {
            "kind": "meta",
            "model": a.model_dir,
            "kernels": a.kernels,
            "tokens": a.tokens,
            "packed": packed,
            "draft": draft_kind,
        }
    )

    def prompt_ids(i):
        msgs = [
            {
                "role": "user",
                "content": f"Write {TOPICS[i % len(TOPICS)]} in Python with docstrings. Output code only.",
            }
        ]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return tok.encode(text, add_special_tokens=False)

    for block in a.blocks:
        for b in a.batches:
            gen = BatchGenerator(
                lm,
                processor,
                max_tokens=a.tokens,
                draft_model=drafter if block else None,
                draft_kind=draft_kind if block else None,
                draft_block_size=block or None,
                greedy_sampling=True,
                compute_logprobs=False,
            )
            clear_rope_state(model)
            ids = [prompt_ids(i) for i in range(b)]
            kws = [
                model.get_input_embeddings(mx.array(x)[None], None, mask=None).to_dict()
                for x in ids
            ]
            t0 = time.perf_counter()
            uids = gen.insert(ids, max_tokens=a.tokens, prompt_kwargs=kws)
            counts = dict.fromkeys(uids, 0)
            first = {}
            done = set()
            try:
                while len(done) < b:
                    _, resps = gen.next()
                    now = time.perf_counter()
                    for r in resps:
                        if r.uid not in counts or r.token is None:
                            if r.finish_reason is not None:
                                done.add(r.uid)
                            continue
                        counts[r.uid] += 1
                        first.setdefault(r.uid, now)
                        if r.finish_reason is not None:
                            done.add(r.uid)
            finally:
                gen.close()
            wall = time.perf_counter() - t0
            total = sum(counts.values())
            decode_start = max(first.values()) if first else t0
            emit(
                {
                    "kind": "run",
                    "block": block,
                    "batch": b,
                    "tokens": total,
                    "wall_s": round(wall, 2),
                    "aggregate_tps": round(total / wall, 1),
                    "aggregate_decode_tps": round(
                        (total - b) / max(1e-6, time.perf_counter() - decode_start), 1
                    ),
                    "per_row_tps": round(total / wall / b, 1),
                }
            )
            mx.clear_cache()


if __name__ == "__main__":
    main()
