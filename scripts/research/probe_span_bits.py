"""Do prefill span plans give the same bits? Hidden states of one prompt run as
spans of 512 / 1024 / 2048 / whole through ``round_driver.forward``, compared
bitwise per layer prefix (first layer count where plans differ).

    python scripts/research/probe_span_bits.py <ckpt> --length 4096
"""

import argparse
import random
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def run(lm, ids, span, layers):
    from yunshu_engine.round_driver.forward import Segment, forward

    inner = lm.model
    full = inner.layers
    inner.layers = full[:layers]
    try:
        cache = lm.make_cache()
        outs = []
        for s in range(0, len(ids), span):
            h = forward(
                lm, [Segment(cache, mx.array(ids[s : s + span], dtype=mx.int32))]
            )
            mx.eval(h)
            outs.append(h)
        state = [a for c in cache for a in (c.state if hasattr(c, "state") else [])]
        mx.eval(*[a for a in state if a is not None])
        return mx.concatenate(outs, axis=0), state
    finally:
        inner.layers = full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--length", type=int, default=4133)
    ap.add_argument(
        "--spans", type=int, nargs="*", default=[512, 1024, 2048, 4096, 1000, 100]
    )
    a = ap.parse_args()
    from mlx_vlm import load as vlm_load

    from yunshu_engine.kernels import lane_linear

    model, _ = vlm_load(a.ckpt)
    lm = model.language_model
    lane_linear.convert(lm)
    rng = random.Random(0)
    ids = [rng.randrange(1000, 20000) for _ in range(a.length)]
    n = len(lm.model.layers)
    for layers in sorted({1, 4, n}):
        if layers > n:
            continue
        ref, rstate = run(lm, ids, a.spans[0], layers)
        for sp in a.spans[1:]:
            o, ostate = run(lm, ids, sp, layers)
            st = all(
                x is None or bool(mx.array_equal(x, y).item())
                for x, y in zip(rstate, ostate, strict=True)
            )
            print("   cache state equal:", st)
            same = bool(mx.array_equal(ref, o).item())
            bad = int((ref != o).any(axis=-1).sum().item())
            print(
                f"layers={layers} span {a.spans[0]} vs {sp}: equal={same} rows_differ={bad}/{len(ids)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
