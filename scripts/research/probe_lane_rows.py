"""Lane matmul vs stock quantized matmul cost by row count (27B projection
shapes, 4-bit g64): is the lane kernel flat in M up to 128 rows, and what does
a stock matmul cost at the same M. Also the untile copy (lane layout -> stock
layout) a prefill segment would pay per weight.

    python scripts/research/probe_lane_rows.py [--bits 4]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from yunshu_engine.kernels.tensorfold import lane_qmm  # noqa: E402

SHAPES = {  # name: (K in, N out)
    "mlp_up": (5120, 17408),
    "mlp_down": (17408, 5120),
    "gdn_qkv": (5120, 10240),
    "attn_o": (6144, 5120),
    "lm_head": (5120, 248320),
}
ROWS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 512, 2048]


def amort(fn, n=12):
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval([fn() for _ in range(n)])
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    mx.random.seed(0)
    for name, (k, n) in SHAPES.items():
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=a.bits)
        tiled = n % lane_qmm.NT == 0
        wt = lane_qmm.tile_weight(wq, lane_qmm.NT, 64, bits=a.bits) if tiled else wq
        sbt = lane_qmm.pack_scales(sc, bi)
        mx.eval(wt, sbt, wq, sc, bi)
        row = {"shape": name, "K": k, "N": n}
        if tiled:
            row["untile_ms"] = round(
                amort(lambda: lane_qmm.untile_weight(wt, lane_qmm.NT, 64, bits=a.bits)),
                2,
            )
        for m in ROWS:
            x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            stock = amort(
                lambda: mx.quantized_matmul(
                    x, wq, sc, bi, transpose=True, group_size=64, bits=a.bits
                )
            )
            if m <= lane_qmm.MAX_ROWS:
                lane = amort(
                    lambda: lane_qmm.lane_matmul(x, wt, sbt, tiled=tiled, group=64)
                )
            else:
                pieces = -(-m // 128)
                lane = amort(
                    lambda: mx.concatenate(
                        [
                            lane_qmm.lane_matmul(
                                x[i * 128 : (i + 1) * 128],
                                wt,
                                sbt,
                                tiled=tiled,
                                group=64,
                            )
                            for i in range(pieces)
                        ]
                    ),
                    n=4,
                )
            row[f"M{m}"] = {"lane_ms": round(lane, 3), "stock_ms": round(stock, 3)}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
