"""Bit identity and timing: fast-tree attention with fused glue kernels vs the op chain.

Run only through gpuq. For several window start positions n0 (no shared chunk,
one, two, many; empty and long prefix tails) the two paths run on the same random
queries, keys, values and random trees; every output must be bit-equal. Times a
dependent chain of 16 layers (each output feeds the next layer's queries).
Writes JSON lines and a final ``complete`` record; exits nonzero on a mismatch.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--n0",
        type=int,
        nargs="+",
        default=[100, 511, 512, 700, 1024, 1300, 1535, 8229, 8700],
    )
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 3, 8, 9, 16])
    ap.add_argument("--trees", type=int, default=6)
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()

    import mlx.core as mx

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.dflash_plan import FastShape, reorder

    rng = random.Random(11)
    h, hkv, d, w, layers = 24, 4, 256, 16, 16
    scale = d**-0.5
    key = mx.random.key(2)

    def rnd(shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape, key=sub).astype(mx.bfloat16)

    rows, ok = [], True
    for w in a.widths:
        for n0 in a.n0:
            row = one(
                mx, tv, FastShape, reorder, rng, rnd, h, hkv, d, w, layers, scale, n0, a
            )
            row["width"] = w
            ok &= row["bit_equal"]
            rows.append(row)
            print(row, flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"complete": True, "bit_equal": ok}) + "\n")
    sys.exit(0 if ok else 1)


def one(mx, tv, FastShape, reorder, rng, rnd, h, hkv, d, w, layers, scale, n0, a):
    cap = -(-(n0 + w + 64) // 64) * 64
    cache = SimpleNamespace(keys=rnd((1, hkv, cap, d)), values=rnd((1, hkv, cap, d)))
    qs = [rnd((1, h, w, d)) for _ in range(layers)]
    shapes = []
    for _ in range(a.trees):
        par = mx.array(
            [-1] + [rng.randrange(0, t) for t in range(1, w)], dtype=mx.int32
        )
        if w > 1:
            _, par, _ = reorder(mx.arange(w - 1, dtype=mx.int32), par)
        shapes.append(FastShape(par, min(15, w - 1) if w > 1 else 0))
    mx.eval(cache.keys, cache.values, qs, [s.parents_array() for s in shapes])

    def run(shape, fused, queries):
        shape.fast_glue = fused
        rc = tv.RoundContext(shape, n0)
        return tv.tree_attention(queries, cache, scale, shape, n0, rc)

    equal = True
    for shape in shapes:
        ref, new = run(shape, False, qs[0]), run(shape, True, qs[0])
        mx.eval(ref, new)
        equal &= bool(mx.array_equal(ref, new))
    row = dict(n0=n0, bit_equal=equal)
    for arm, fused in (("old", False), ("new", True), ("old", False), ("new", True)):
        times = []
        for rep in range(a.reps + 3):
            t0 = time.perf_counter()
            shape = shapes[rep % a.trees]
            shape.fast_glue = fused
            rc = tv.RoundContext(shape, n0)
            x = qs[0]
            for _layer in range(layers):
                x = tv.tree_attention(x, cache, scale, shape, n0, rc)
            mx.eval(x)
            if rep >= 3:
                times.append((time.perf_counter() - t0) * 1000)
        row[arm + "_ms_16_layers"] = round(statistics.median(times), 3)
    return row


if __name__ == "__main__":
    main()
