"""Cost of one DFlash2 draft (lattice) by part on the served lane stack.

Each part is timed alone (barrier after it), averaged over --steps calls; the
last line is the whole lattice + tree search as the round runs it.

    python scripts/research/profile_dflash_draft.py MODEL_DIR DRAFTER_DIR --output F.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from spec_bench_snapshot import refuse_contended

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def avg(fn, steps):
    for _ in range(3):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    return (time.perf_counter() - t0) / steps * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("drafter_dir")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--hidden-rows", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compile-conv", action="store_true")
    ap.add_argument("--bits", type=int, choices=[4, 6, 8], default=8)
    a = ap.parse_args()
    if refuse_contended(a.output):
        return

    from mlx_vlm.speculative.drafters import (
        load_drafter,
        validate_drafter_compatibility,
    )
    from probe_wide_verify import setup

    from yunshu_engine import dflash_context, dflash_tree

    if a.compile_conv:
        import mlx_vlm.speculative.drafters.dflash2.dflash2 as d2

        raw = d2._grouped_dynamic_convolve
        compiled = mx.compile(raw)

        def conv(hidden, dynamic, base, group_size):
            return compiled(hidden, dynamic, base, group_size)

        d2._grouped_dynamic_convolve = conv
    model, lm, _ids = setup(a.model_dir)
    drafter, kind = load_drafter(a.drafter_dir)
    validate_drafter_compatibility(model, drafter, kind)
    converted = dflash_tree.quantize_drafter(drafter, a.bits)
    dflash_context.install(lm)
    cache = drafter.reset(model)
    cfg = drafter.config
    positions = 7
    d = len(cfg.target_layer_ids) * cfg.hidden_size
    mx.random.seed(0)
    context_rows = max(1, int(cfg.sliding_window or 257) - 1)
    ctx = mx.random.normal((1, context_rows, d)).astype(mx.bfloat16)
    hid = mx.random.normal((1, a.hidden_rows, d)).astype(mx.bfloat16)
    mx.eval(ctx, hid)
    anchor = 1000
    lat = dflash_tree.compute_lattice_gpu(drafter, anchor, ctx, cache, positions)
    mx.eval(lat.cands)

    sel = drafter.candidate_selector
    inputs = mx.array([[anchor] + [int(cfg.mask_token_id)] * positions], dtype=mx.int32)

    def part_hidden():
        mx.eval(drafter._hidden(inputs, hid, cache)[:, 1:])

    dh = drafter._hidden(inputs, hid, cache)[:, 1:]
    mx.eval(dh)
    logits = drafter._logits(dh)
    mx.eval(logits)

    def part_logits():
        mx.eval(drafter._logits(dh))

    def part_topk():
        c = mx.argpartition(logits, -sel.top_k, axis=-1)[0, ..., -sel.top_k :]
        mx.eval(c, mx.take_along_axis(logits[0], c, axis=-1))

    cands = mx.argpartition(logits, -sel.top_k, axis=-1)[0, ..., -sel.top_k :]
    mx.eval(cands)

    def part_selector():
        mx.eval(
            sel.hidden_projection(dh)[0],
            sel.successor_codebook(cands),
            sel.predecessor_codebook(cands),
        )

    def part_lattice():
        l2 = dflash_tree.compute_lattice_gpu(drafter, anchor, hid, cache, positions)
        mx.eval(l2.cands, l2.unary, l2.hproj, l2.succ, l2.pred, l2.anchor)

    def part_search():
        t, p = dflash_tree.search_tree(lat, 15)
        mx.eval(t, p)

    def part_total():
        l2 = dflash_tree.compute_lattice_gpu(drafter, anchor, hid, cache, positions)
        t, p = dflash_tree.search_tree(l2, 15)
        mx.eval(t, p)

    def part_chain():
        tokens = drafter.draft_block_greedy(
            anchor, hid, cache, positions + 1, lambda lg: mx.argmax(lg, axis=-1)
        )
        mx.eval(tokens)

    def part_selector_chain():
        tokens = drafter.draft_block(
            anchor, hid, cache, positions + 1, lambda lg: mx.argmax(lg, axis=-1)
        )
        mx.eval(tokens)

    row = {
        "bits": a.bits,
        "converted": converted,
        "hidden_rows": a.hidden_rows,
        "context_rows": context_rows,
    }
    for name, fn in (
        ("hidden(5 layers+fc)", part_hidden),
        ("lm_head logits", part_logits),
        ("argpartition top16", part_topk),
        ("selector proj+codebooks", part_selector),
        ("lattice (all)", part_lattice),
        ("search_tree", part_search),
        ("lattice+search", part_total),
        ("greedy chain draft", part_chain),
        ("selector chain draft", part_selector_chain),
    ):
        row[name] = round(avg(fn, a.steps), 2)
        print(json.dumps({name: row[name]}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        f.write(json.dumps(row) + "\n")
        from yunshu_engine.dflash_draft import install_quantized_context

        base_context = drafter._project_context_kv
        if not install_quantized_context(drafter):
            raise RuntimeError("quantized context fusion did not engage")
        fused_context = drafter._project_context_kv
        for rep in range(3):
            for fused in [False, True] if rep % 2 == 0 else [True, False]:
                object.__setattr__(
                    drafter,
                    "_project_context_kv",
                    fused_context if fused else base_context,
                )
                result = {
                    "bits": a.bits,
                    "rep": rep,
                    "fused_context": fused,
                    "lattice+search": avg(part_total, a.steps),
                    "greedy chain draft": avg(part_chain, a.steps),
                    "selector chain draft": avg(part_selector_chain, a.steps),
                }
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(json.dumps(result), flush=True)
        f.write(json.dumps({"complete": True, "mode": kind, "bits": a.bits}) + "\n")


if __name__ == "__main__":
    main()
