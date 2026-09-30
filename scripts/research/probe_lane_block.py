"""Lane matmul at 8..128 rows: one threadgroup row block of 32 (default) vs 16
rows per block: time and bit equality (27B projection shapes, 4-bit)."""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from yunshu_engine.kernels.tensorfold import lane_qmm  # noqa: E402

SHAPES = {
    "mlp_up": (5120, 17408),
    "mlp_down": (17408, 5120),
    "gdn_qkv": (5120, 10240),
    "attn_o": (6144, 5120),
    "lm_head": (5120, 248320),
}


def amort(fn, n=20):
    mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval([fn() for _ in range(n)])
    mx.synchronize()
    return round((time.perf_counter() - t0) / n * 1e3, 3)


def main():
    mx.random.seed(0)
    for name, (k, n) in SHAPES.items():
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        wt = lane_qmm.tile_weight(wq, 32, 64, bits=4)
        sbt = lane_qmm.pack_scales(sc, bi)
        row = {"shape": name}
        for m in (8, 16, 17, 24, 32, 40, 48, 64, 96, 128):
            x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            mx.eval(x, wt, sbt)

            def f(rb, x=x):
                return lane_qmm.lane_matmul(
                    x, wt, sbt, tiled=True, group=64, row_block=rb
                )

            a, b = f(None), f(16)
            mx.eval(a, b)
            row[f"M{m}"] = {
                "b32": amort(lambda: f(None)),
                "b16": amort(lambda: f(16)),
                "same": bool(mx.array_equal(a, b).item()),
            }
        print(json.dumps(row), flush=True)


main()
