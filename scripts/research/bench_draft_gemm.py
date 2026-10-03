"""Where does the DFlash2 drafter's 6.5 ms hidden pass go?

Times the drafter's own matmul shapes at the row counts a draft step uses
(M = 4..12) for bf16 / q8 / q4, one eval per matmul (launch + execution) and a
batch of 40 matmuls with distinct, pre-evaluated inputs under one eval.
This amortizes host submission/synchronization, but does not isolate kernel
execution: weights can be cache-hot and graph encoding is still included.
Identical inputs would allow common-subexpression elimination to collapse the
batch, invalidating the per-matmul time. Output: one JSON line per case, final ``complete`` record.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

SHAPES = {  # name: (out, in) of the drafter's Linear layers
    "mlp_up": (17408, 5120),
    "mlp_down": (5120, 17408),
    "q_proj": (4096, 5120),
    "fc": (5120, 25600),
}


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 12])
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    mx.random.seed(0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        for name, (o, i) in SHAPES.items():
            if a.tiny:
                o, i = o // 16, i // 16
            w = (mx.random.normal((o, i)) * 0.02).astype(mx.bfloat16)
            for bits in (0, 8, 4):
                if bits:
                    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
                    nbytes = wq.nbytes + s.nbytes + b.nbytes
                    mm = lambda x: mx.quantized_matmul(  # noqa: E731
                        x, wq, s, b, transpose=True, group_size=64, bits=bits
                    )
                    mx.eval(wq, s, b)
                else:
                    nbytes = w.nbytes
                    mm = lambda x: x @ w.T  # noqa: E731
                    mx.eval(w)
                for m in a.rows:
                    x = mx.random.normal((1, m, i)).astype(mx.bfloat16)
                    mx.eval(x)
                    single = timed(lambda: mx.eval(mm(x)))

                    xs = [mx.random.normal(x.shape).astype(x.dtype) for _ in range(40)]
                    mx.eval(xs)

                    def chain():
                        ys = [mm(value) for value in xs]
                        mx.eval(ys)

                    chained = timed(chain, 5) / 40
                    row = dict(
                        shape=name,
                        bits=bits or 16,
                        rows=m,
                        bytes=nbytes,
                        single_ms=round(single, 4),
                        chained_ms=round(chained, 4),
                        gbps_chained=round(nbytes / chained / 1e6, 1),
                    )
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps(row), flush=True)
        out.write(json.dumps({"complete": True, "success": True}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
