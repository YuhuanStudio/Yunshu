"""Bit identity and timing: GDN norm-gate with the lane group sums fused in.

Old: omlx norm-gate kernel then the lane xsum launch. New:
tree_glue.norm_gate_lane. Run only through gpuq; exits nonzero on a mismatch.
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
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()

    import mlx.core as mx

    from yunshu_engine import tree_glue
    from yunshu_engine.kernels.omlx import qwen35_gdn_verify_fused as gv
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    hv, dv = 48, 128
    key = mx.random.key(4)

    def rnd(shape, scale=1.0):
        nonlocal key
        key, sub = mx.random.split(key)
        return (mx.random.normal(shape, key=sub) * scale).astype(mx.bfloat16)

    norm = SimpleNamespace(weight=rnd((dv,), 0.3) + 1, eps=1e-6)
    width = hv * dv
    rows_out, ok = [], True
    for m in a.rows:
        mp = 16 * ((m + 15) // 16)
        data = [(rnd((1, m, hv, dv), 2.0), rnd((1, m, hv, dv), 2.0)) for _ in range(8)]
        mx.eval(data, norm.weight)

        def old(i):
            y, z = data[i % 8]
            out, _ = gv._norm_gate_kernel(norm.eps)(
                inputs=[y, z, norm.weight],
                template=[("InT", mx.bfloat16)],
                grid=(32, m * hv, 1),
                threadgroup=(32, 8, 1),
                output_shapes=[(1, m, width), (m, width // 64)],
                output_dtypes=[mx.bfloat16, mx.float32],
            )
            xs = q._kernel("xsum")(
                inputs=[out.reshape(m, width), q._mdims(m, mp)],
                template=[("K", width), ("GS", 64)],
                grid=(width // 64, mp, 1),
                threadgroup=(min(width // 64, 256), 1, 1),
                output_shapes=[(width // 64, mp)],
                output_dtypes=[mx.float32],
            )[0]
            return out, xs

        def new(i):
            y, z = data[i % 8]
            out = tree_glue.norm_gate_lane(y, z, norm, hv)
            return out, q._xs_cache[id(out)][1]

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
