"""Where does a verify forward's growth with context go? Ablations of one tree forward (run via gpuq only).

Variants per (context, rows): full; noattn (tree_attention replaced by zeros, so the
tile kernel and tail gather vanish while cache update, rope and projections remain);
commit_prefix / commit_compact (tree_commit cost for a prefix path vs a path that
needs key/value compaction). Writes JSON lines and a final complete record.
"""

import argparse
import json
import sys
import time
from pathlib import Path

VARIANTS = ("full", "noattn", "commit_prefix", "commit_compact")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", nargs="?", default="")
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 65536])
    ap.add_argument("--rows", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main():
    a = parse()
    if a.dry_run:
        print(
            json.dumps(
                {
                    "cells": len(a.contexts) * len(a.rows) * len(VARIANTS),
                    "variants": list(VARIANTS),
                }
            )
        )
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mlx.core as mx
    import probe_wide_verify as pw

    from yunshu_engine import tree_verify as tv

    _, lm, ids = pw.setup(a.model_dir)
    real_attention = tv.tree_attention

    def fake_attention(queries, *_args, **_kw):
        return mx.zeros(queries.shape, dtype=mx.bfloat16)

    rows = []
    for ctx in a.contexts:
        cache = pw.prefill(lm, ids, ctx)
        for w in a.rows:
            toks = mx.array(ids[ctx : ctx + w])[None]
            shape = tv.TreeShape(pw.shape_parents("chain", w))
            for variant in VARIANTS:
                tv.tree_attention = (
                    fake_attention if variant == "noattn" else real_attention
                )

                def once(variant=variant, toks=toks, shape=shape, w=w):
                    res = tv.tree_forward(lm, toks, shape, cache)
                    tgt = lm.speculative_argmax_from_hidden(res.hidden)
                    if variant.startswith("commit"):
                        path = (
                            list(range(w // 2))
                            if variant == "commit_prefix"
                            else [0, 2, 4, 6][: max(2, w // 4)]
                        )
                        mx.eval(tgt)
                        t0 = time.perf_counter()
                        tv.tree_commit(lm, cache, res, path)
                        mx.eval([c.state for c in cache])
                        spent = time.perf_counter() - t0
                        # restore: drop the committed rows again
                        for c in cache:
                            if getattr(c, "keys", None) is not None and hasattr(
                                c, "trim"
                            ):
                                c.trim(len(path))
                        return None, spent
                    return tgt, lambda: tv.tree_abort(cache, res)

                times = []
                for rep in range(3 + a.steps):
                    if variant.startswith("commit"):
                        _, spent = once()
                        if rep >= 3:
                            times.append(spent * 1e3)
                    else:
                        t0 = time.perf_counter()
                        tgt, abort = once()
                        mx.eval(tgt)
                        dt = (time.perf_counter() - t0) * 1e3
                        abort()
                        if rep >= 3:
                            times.append(dt)
                times.sort()
                row = {
                    "context": ctx,
                    "rows": w,
                    "variant": variant,
                    "median_ms": round(times[len(times) // 2], 3),
                }
                print(json.dumps(row), flush=True)
                rows.append(row)
            tv.tree_attention = real_attention
        mx.clear_cache()
    with a.output.open("x") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
        out.write(json.dumps({"complete": True}) + "\n")


if __name__ == "__main__":
    main()
