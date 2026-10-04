"""Bit identity and timing: tree GDN prework with the conv window gathered in-kernel.

Old: concatenate + take + gdn_prework_fused (three dispatches). New:
dflash_fast.prework_gather (one). Run only through gpuq; writes JSON lines and a
final ``complete`` record. Exits nonzero on any bit mismatch.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()

    import mlx.core as mx

    from yunshu_engine.dflash_plan import FastShape, reorder
    from yunshu_engine.kernels.omlx import qwen35_gdn_prework as gp

    rng = random.Random(3)
    hk, hv, dk, dv, w = 16, 48, 128, 128, 16
    c_dim = 2 * hk * dk + hv * dv
    key = mx.random.key(5)

    def rnd(shape, scale=1.0):
        nonlocal key
        key, sub = mx.random.split(key)
        return (mx.random.normal(shape, key=sub) * scale).astype(mx.bfloat16)

    class Layer:
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = hk, hv, dk, dv

        def __init__(self):
            self.conv1d = type("C", (), {"weight": rnd((c_dim, 4, 1), 0.5)})()

    layers = [Layer() for _ in range(6)]
    mix = [rnd((1, w, c_dim)) for _ in range(6)]
    prev = [rnd((1, 3, c_dim)) for _ in range(6)]
    shapes = []
    for _ in range(a.trees):
        par = [-1] + [rng.randrange(0, t) for t in range(1, w)]
        par = mx.array(par, dtype=mx.int32)
        _, par, _ = reorder(mx.arange(w - 1, dtype=mx.int32), par)
        shapes.append(FastShape(par, 15))
    mx.eval([layers[0].conv1d.weight, mix, prev, [s.parents_array() for s in shapes]])

    def old(shape, i):
        ly = layers[i % 6]
        seq = mx.concatenate([prev[i % 6], mix[i % 6]], axis=1)[0]
        windows = mx.take(seq, shape.conv_index(), axis=0).reshape(w, 3, c_dim)
        inv = dk**-0.5
        q, k, v, _ = gp.gdn_prework_fused(
            mix[i % 6].reshape(w, 1, c_dim),
            windows,
            ly.conv1d.weight,
            mx.array(inv * inv, dtype=mx.bfloat16),
            mx.array(inv, dtype=mx.bfloat16),
            hk,
            hv,
            dk,
            dv,
        )
        return q, k, v

    def new(shape, i):
        return shape.gdn_prework(mix[i % 6], prev[i % 6], layers[i % 6])

    equal = True
    for i, s in enumerate(shapes):
        o, n = old(s, i), new(s, i)
        mx.eval(o, n)
        equal &= all(bool(mx.array_equal(x, y)) for x, y in zip(o, n, strict=True))
    rows = []
    for name, fn in (("old", old), ("new", new), ("old", old), ("new", new)):
        times = []
        for rep in range(a.reps + 3):
            t0 = time.perf_counter()
            ys = [fn(shapes[i % a.trees], i) for i in range(a.layers)]
            mx.eval(ys)
            if rep >= 3:
                times.append((time.perf_counter() - t0) * 1000)
        rows.append(
            dict(
                arm=name,
                ms_per_forward_median=round(statistics.median(times), 3),
                bit_equal=equal,
            )
        )
        print(rows[-1], flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"complete": True, "bit_equal": equal}) + "\n")
    sys.exit(0 if equal else 1)


if __name__ == "__main__":
    main()
