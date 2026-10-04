"""Bit identity and timing: residual add + RMSNorm with the lane group sums fused in.

Old: omlx add_rms_norm then the lane xsum launch. New: tree_glue.add_rms_lane.
Run only through gpuq. Compares the sum and normed outputs with the original
kernel and the group sums with the xsum kernel's, for several row counts.
Writes JSON lines and a final ``complete`` record; exits nonzero on a mismatch.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 5, 8, 15, 16, 17, 32])
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--layers", type=int, default=128)
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()

    import mlx.core as mx

    from yunshu_engine import tree_glue
    from yunshu_engine.kernels.omlx import qwen35_verify_qmm as vq
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    key = mx.random.key(9)

    def rnd(shape, scale=1.0):
        nonlocal key
        key, sub = mx.random.split(key)
        return (mx.random.normal(shape, key=sub) * scale).astype(mx.bfloat16)

    norm = SimpleNamespace(weight=rnd((a.d,), 0.3) + 1, eps=1e-6)
    rows_out, ok = [], True
    for m in a.rows:
        mp = 16 * ((m + 15) // 16)
        xs_in = [(rnd((1, m, a.d), 2.0), rnd((1, m, a.d), 2.0)) for _ in range(8)]
        mx.eval(xs_in, norm.weight)

        def old(i):
            x, y = xs_in[i % 8]
            s, n, _ = vq.add_rms_norm(x, y, norm)
            xs = q._kernel("xsum")(
                inputs=[n.reshape(m, a.d), q._mdims(m, mp)],
                template=[("K", a.d), ("GS", 64)],
                grid=(a.d // 64, mp, 1),
                threadgroup=(min(a.d // 64, 256), 1, 1),
                output_shapes=[(a.d // 64, mp)],
                output_dtypes=[mx.float32],
            )[0]
            return s, n, xs

        def new(i):
            x, y = xs_in[i % 8]
            return tree_glue.add_rms_lane(x, y, norm)

        equal = True
        for i in range(8):
            ref, got = old(i), new(i)
            mx.eval(ref, got)
            equal &= all(
                bool(mx.array_equal(r, g)) for r, g in zip(ref, got, strict=True)
            )
        ok &= equal
        row = dict(rows=m, bit_equal=equal)
        for arm, fn in (("old", old), ("new", new), ("old", old), ("new", new)):
            times = []
            for rep in range(a.reps + 3):
                t0 = time.perf_counter()
                outs = [fn(i) for i in range(a.layers)]
                mx.eval(outs)
                if rep >= 3:
                    times.append((time.perf_counter() - t0) * 1000)
            row[arm + "_ms_per_%d" % a.layers] = round(statistics.median(times), 3)
        rows_out.append(row)
        print(row, flush=True)
    with open(a.out, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"complete": True, "bit_equal": ok}) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
