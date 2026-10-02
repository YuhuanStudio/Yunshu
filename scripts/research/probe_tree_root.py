"""Distinguish a tree-root error from a single-row verify oracle mismatch."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from probe_wide_verify import chain_verify, prefill, setup, shape_parents


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    _, lm, ids = setup(args.model)
    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import lane_layers

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for fused_layers in (False, True):
            if fused_layers:
                lane_layers.install()
            cache = prefill(lm, ids, 600)
            decoded = lm(mx.array(ids[600:601])[None], cache=cache, return_hidden=True)
            reference = decoded.hidden_states[-1]
            mx.eval(reference)
            for kind, width in (("chain", 1), ("chain", 2), ("heap", 8), ("heap", 32)):
                cache = prefill(lm, ids, 600)
                tokens = mx.array(ids[600 : 600 + width])[None]
                if kind == "chain":
                    result = chain_verify(lm, tokens, cache)
                    hidden = result.hidden[:, :1]
                    mx.eval(hidden)
                    result.abort()
                else:
                    result = tv.tree_forward(
                        lm, tokens, tv.TreeShape(shape_parents(kind, width)), cache
                    )
                    hidden = result.hidden[:, :1]
                    mx.eval(hidden)
                    tv.tree_abort(cache, result)
                row = {
                    "kind": kind,
                    "width": width,
                    "fused_layers": fused_layers,
                    "equal_to_plain_root": bool(
                        mx.array_equal(hidden, reference).item()
                    ),
                    "different_values": int(mx.sum(hidden != reference).item()),
                    "max_abs": float(
                        mx.max(
                            mx.abs(
                                hidden.astype(mx.float32) - reference.astype(mx.float32)
                            )
                        ).item()
                    ),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)
        out.write(
            json.dumps({"complete": True, "measurement": "correctness diagnostic"})
            + "\n"
        )


if __name__ == "__main__":
    main()
