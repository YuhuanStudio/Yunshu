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
    p.add_argument(
        "--native-tiny",
        action="store_true",
        help="Flash-Next HC/QSA/PLE/MoE small-array pilot",
    )
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def plan(a):
    if not a.tiny and not a.native_tiny and not a.model:
        raise ValueError("model path or --tiny required")
    if a.tiny and a.native_tiny:
        raise ValueError("select one tiny architecture")
    if a.native_tiny:
        if (
            not 4 <= a.context <= 4000
            or not a.rows
            or any(n < 1 or n > 16 for n in a.rows)
        ):
            raise ValueError("native tiny context 4..4000 and 1..16 rows required")
        return [
            dict(rows=n, nodes="1/2 chain", context=a.context, native=True)
            for n in a.rows
        ]
    if a.context < 4 or a.nodes < 1 or not a.rows:
        raise ValueError("context >= 4 and positive nodes/rows required")
    if any(n < 1 or n * a.nodes > 32 for n in a.rows):
        raise ValueError("each pack needs 1..32 nodes total")
    return [dict(rows=n, nodes=a.nodes, context=a.context) for n in a.rows]


def tiny_config():
    from mlx_vlm.models.qwen3_5.config import TextConfig

    return TextConfig(
        model_type="qwen3_5",
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=512,
        tie_word_embeddings=False,
    )


def tiny():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    from yunshu_engine.kernels import gdn_prefill, lane_linear, omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install, set_active

    omlx.apply()
    mx.random.seed(23)
    lm = LanguageModel(tiny_config())
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
    if a.tiny:
        from yunshu_engine.round_driver.forward import (
            Segment,
        )
        from yunshu_engine.round_driver.forward import (
            forward as prefill_forward,
        )

        def prefix(i):
            cache = lm.make_cache()
            prefill_forward(
                lm,
                [Segment(cache, mx.array(ids[i : i + a.context + i], dtype=mx.int32))],
            )
            mx.eval(*state(cache))
            return cache

        prefixes = [prefix(i) for i in range(max(a.rows))]
    else:
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


def native_tiny():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from scripts.research.qwen4_mtp_probe import tiny_config

    mx.random.seed(17)
    config = ModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": tiny_config(),
            "vision_config": {
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [],
            },
        }
    )
    lm = LanguageModel(config.text_config, config)
    lm.set_dtype(mx.bfloat16)
    nn.quantize(
        lm,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: (
            hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0
        ),
    )
    mx.eval(lm.parameters())
    return lm


def run_native_tiny(a, write):
    import mlx.core as mx

    from yunshu_engine.round_driver.verify import native_chain_forward

    lm = native_tiny()
    for count in a.rows:
        for keep in (None, 1):
            prefixes, tokens = [], []
            for b in range(count):
                cache = lm.make_cache()
                lm(
                    mx.array([[1 + (i + b) % 60 for i in range(a.context + b)]]),
                    cache=cache,
                )
                mx.eval(*state(cache))
                prefixes.append(cache)
                tokens.append(mx.array([[8 + b, 9 + b]])[:, : 1 + b % 2])
            solo = copy.deepcopy(prefixes)
            packed = copy.deepcopy(prefixes)
            references = [
                lm.speculative_verify_hidden(t, c)
                for t, c in zip(tokens, solo, strict=True)
            ]
            results = native_chain_forward(lm, tokens, packed)
            for b, ((ref, _, st), (got, pt)) in enumerate(
                zip(references, results, strict=True)
            ):
                checks = dict(
                    hidden=bool(mx.array_equal(ref, got).item()),
                    logits=bool(
                        mx.array_equal(
                            lm.speculative_logits_from_hidden(ref),
                            lm.speculative_logits_from_hidden(got),
                        ).item()
                    ),
                )
                if keep is None:
                    st.abort()
                    pt.abort()
                else:
                    st.commit(keep)
                    pt.commit(keep)
                token = mx.array([[21 + b]])
                checks["next_logits"] = bool(
                    mx.array_equal(
                        lm(token, cache=solo[b]).logits,
                        lm(token, cache=packed[b]).logits,
                    ).item()
                )
                write(
                    dict(
                        rows=count,
                        row=b,
                        context=a.context + b,
                        keep=keep,
                        native=True,
                        checks=checks,
                    )
                )
                if not all(checks.values()):
                    raise RuntimeError(
                        f"native packed parity failed: row={b}, checks={checks}"
                    )
    write(dict(complete=True, success=True, device="m5", native=True, tiny=True))


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

        if a.native_tiny:
            run_native_tiny(a, write)
        else:
            run(a, write)


if __name__ == "__main__":
    main()
