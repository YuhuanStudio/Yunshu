"""Phase B packed-tree primitive parity pilot. Run only through gpuq.

Compares each window to its solo lane call, including captured layers, logits
and state after a branched commit. No request content is written to results.
--dry-run validates the plan without importing MLX.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", nargs="?")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def plan(a):
    if not a.tiny and not a.model:
        raise ValueError("model path or --tiny required")
    if a.context < 4 or a.nodes < 1 or not a.rows:
        raise ValueError("context >= 4 and positive nodes/rows required")
    if any(n < 1 or n * a.nodes > 32 for n in a.rows):
        raise ValueError("each pack needs 1..32 nodes total")
    return [dict(rows=n, nodes=a.nodes, context=a.context) for n in a.rows]


def tiny():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    from yunshu_engine.kernels import gdn_prefill, lane_linear, omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install, set_active

    omlx.apply(row_exact=False)
    mx.random.seed(23)
    lm = LanguageModel(
        TextConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=256,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            vocab_size=512,
            tie_word_embeddings=False,
        )
    )
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=4)
    lane_linear.convert(lm)
    lane_linear.set_stock_rows(lane_linear.PIECE)
    install(lm, packed=False)
    set_active(True)
    gdn_prefill.install()
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    mx.eval(lm.parameters())
    return lm, [3 + (i * 7) % 500 for i in range(10000)]


def state(cache):
    return [a for c in cache for a in c.state if a is not None]


def run(a, write):
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5 import language as q35
    from scripts.research.probe_wide_verify import make_shape, prefill, setup

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.dflash_fast import Verifier
    from yunshu_engine.round_driver.verify import Window, forward

    lm, ids = tiny() if a.tiny else setup(a.model)[1:]
    verifier = Verifier(q35._EXACT_SPECULATIVE_VERIFIER)
    capture = [0, len(lm.model.layers) - 1]
    # Reuse immutable prefix arrays; each copy owns the cache wrappers and any
    # mutable KV storage, matching the normal cache transaction's ownership.
    prefixes = [prefill(lm, ids[i:], a.context + i) for i in range(max(a.rows))]
    for n in a.rows:
        solo_caches = [copy.deepcopy(c) for c in prefixes[:n]]
        packed_caches = [copy.deepcopy(c) for c in prefixes[:n]]
        windows, solo = [], []
        for i, cache in enumerate(solo_caches):
            shape = make_shape(tv, "heap", a.nodes, True)
            tokens = mx.array(
                [ids[a.context + i : a.context + i + a.nodes]], dtype=mx.int32
            )
            ranks = getattr(shape, "original_ranks", None)
            if ranks is not None:
                tokens = mx.take(tokens, ranks, axis=1)
            windows.append(Window(tokens, shape, packed_caches[i]))
            res = tv.tree_forward(lm, tokens, shape, cache, capture, verifier=verifier)
            mx.eval(res.hidden, *res.captured)
            solo.append(res)
        packed = forward(lm, windows, capture, verifier=verifier)
        mx.eval(*[r.hidden for r in packed])
        for i, (s, p, w) in enumerate(zip(solo, packed, windows, strict=True)):

            def equal(x, y):
                return bool(mx.array_equal(x, y).item())

            def head(x):
                from yunshu_engine.round_driver.forward import logits

                return logits(lm, x)

            checks = dict(
                hidden=equal(s.hidden, p.hidden),
                logits=equal(head(s.hidden), head(p.hidden)),
                captured=all(
                    equal(x, y) for x, y in zip(s.captured, p.captured, strict=True)
                ),
            )
            path = tv.TreeShape(w.shape.parents_array().tolist()).paths[-1]
            tv.tree_commit(lm, solo_caches[i], s, path)
            tv.tree_commit(lm, packed_caches[i], p, path)
            left, right = state(solo_caches[i]), state(packed_caches[i])
            checks["commit"] = len(left) == len(right) and all(
                equal(x, y) for x, y in zip(left, right, strict=True)
            )
            write(
                dict(rows=n, row=i, nodes=a.nodes, context=a.context + i, checks=checks)
            )
            if not all(checks.values()):
                raise RuntimeError(
                    f"packed parity failed: rows={n}, row={i}, checks={checks}"
                )
    write(dict(complete=True, success=True, device="m5", tiny=a.tiny))


def main():
    a = parser().parse_args()
    cells = plan(a)
    if a.dry_run:
        print(json.dumps(dict(plan=cells, complete=True, dry_run=True)))
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:

        def write(row):
            line = json.dumps(row)
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)

        run(a, write)


if __name__ == "__main__":
    main()
