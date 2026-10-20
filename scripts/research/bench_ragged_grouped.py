"""Bit identity and timing: ragged_decode_attention for 9-32 token windows, one grouped tile
launch vs a launch per 8-token group (the dense lane's copy / chain verify windows).

Run only through gpuq. Every output must be bit-equal; timing chains 16 layers over 16
distinct caches so the prefix K/V is read cold, as in a real forward. JSON lines, then a
``complete`` record; exits nonzero on a mismatch.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[600, 1535, 8229, 32768])
    ap.add_argument(
        "--widths", type=int, nargs="+", default=[9, 12, 15, 16, 17, 24, 32]
    )
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()

    import mlx.core as mx

    from yunshu_engine.kernels import ragged_attention as ra

    h, hkv, d, layers = 24, 4, 256, 16
    scale = d**-0.5
    key = mx.random.key(5)

    def rnd(shape):
        nonlocal key
        key, sub = mx.random.split(key)
        return mx.random.normal(shape, key=sub).astype(mx.bfloat16)

    rows, ok = [], True
    for n in a.n:
        cap = -(-(n + 64) // 64) * 64
        caches = [(rnd((1, hkv, cap, d)), rnd((1, hkv, cap, d))) for _ in range(layers)]
        mx.eval(caches)
        for w in a.widths:
            q = rnd((1, h, w, d))
            length = mx.array([n], dtype=mx.int32)

            def run(grouped, x, k, v):
                ra.GROUPED_LAUNCH = grouped
                return ra.ragged_decode_attention(
                    x, k, v, length, scale, max_length=n, row_lengths=(n,), impl="tile"
                )

            k0, v0 = caches[0]
            ref, new = run(False, q, k0, v0), run(True, q, k0, v0)
            mx.eval(ref, new)
            equal = bool(mx.array_equal(ref, new).item())
            ok &= equal
            row = {"n": n, "width": w, "bit_equal": equal}
            for arm, grouped in (
                ("old", False),
                ("new", True),
                ("old", False),
                ("new", True),
            ):
                times = []
                for rep in range(a.reps + 2):
                    t0 = time.perf_counter()
                    x = q
                    for k, v in caches:
                        x = run(grouped, x, k, v)
                    mx.eval(x)
                    if rep >= 2:
                        times.append((time.perf_counter() - t0) * 1000)
                row[arm + "_ms_16_layers"] = round(statistics.median(times), 3)
            print(json.dumps(row), flush=True)
            rows.append(row)
        del caches
        mx.clear_cache()
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"complete": True, "bit_equal": ok}) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
