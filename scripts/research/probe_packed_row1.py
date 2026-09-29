"""Is one row on the packed tensor-unit kernel the same bits as row 0 of a padded 2-row call?

The batch-invariant lane pads a single row with a zero row (Full + Concatenate +
Slice launches per projection) because "one row takes a different matvec
kernel". If the tensor-unit kernel accepts M=1 and returns the bits of the padded
call, the pad's three launches per projection can go. Random 4-bit weights at the
27B projection shapes; the answer does not depend on weight values, only shapes.

    python scripts/research/probe_packed_row1.py --output runs/packed-row1.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl  # noqa: E402

SHAPES = [  # (K, N) of Qwen3.8-27B's 4-bit projections
    (5120, 10240 + 6144),  # in_proj_qkv + z (one store)
    (5120, 17408 * 2),  # gate + up
    (17408, 5120),  # down
    (6144, 5120),  # out / o
    (5120, 12288 + 2048),  # q + k + v
    (5120, 17408),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    mx.random.seed(0)
    rows = []
    for k, n in SHAPES:
        lin = nn.Linear(k, n, bias=False)
        q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
        mx.eval(q.parameters())
        packed = pl._pack([q])[0]
        store = packed._store
        bad = 0
        for trial in range(8):
            x = mx.random.normal((1, k)).astype(mx.bfloat16)
            pad = mx.concatenate([x, mx.zeros_like(x)])
            ref = pl.packed_matmul(
                pad,
                packed.packed_weight,
                packed.packed_scales,
                packed.packed_biases,
                k,
                n,
            )[:1]
            # one row straight through the tensor-unit kernel
            ksplit, parts = pl._small_geometry(k, n)
            sg = pl.TILE_N // 16
            (y,) = pl._kernels()["small"](
                inputs=[
                    x,
                    packed.packed_weight,
                    packed.packed_scales,
                    packed.packed_biases,
                ],
                template=[
                    ("K", k),
                    ("N", n),
                    ("TILE_N", pl.TILE_N),
                    ("SG", sg),
                    ("PARTS", parts),
                    ("KSPLIT", ksplit),
                ],
                grid=(32 * sg * parts * (n // pl.TILE_N), ksplit, 1),
                threadgroup=(32 * sg * parts, 1, 1),
                output_shapes=[(1, n) if ksplit == 1 else (ksplit * 1 * n,)],
                output_dtypes=[mx.bfloat16 if ksplit == 1 else mx.float32],
            )
            got = pl._reduce(y, ksplit, 1, n)
            mx.eval(ref, got)
            bad += int(not mx.array_equal(ref, got).item())

        def bench(fn):
            mx.eval(fn())
            t = time.perf_counter()
            for _ in range(50):
                out = fn()
            mx.eval(out)
            return (time.perf_counter() - t) / 50 * 1e6

        x = mx.random.normal((1, k)).astype(mx.bfloat16)
        mx.eval(x)
        us_pad = bench(
            lambda: pl.packed_matmul(
                mx.concatenate([x, mx.zeros_like(x)]),
                packed.packed_weight,
                packed.packed_scales,
                packed.packed_biases,
                k,
                n,
            )[:1]
        )
        row = {
            "K": k,
            "N": n,
            "mismatched_trials": bad,
            "ksplit": ksplit,
            "padded_us": round(us_pad, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del store
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
