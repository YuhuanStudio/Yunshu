"""Lane matmul row block 32 vs 16 with weights streamed cold, as in a real forward.

probe_lane_block.py times one weight called repeatedly, so every row block after the first
reads it from the system cache. A verify forward touches each weight once per call, so a
second row block re-reads it from DRAM. Here each timed call cycles through COPIES distinct
weights of the shape (> 1 GB per shape), which is what a 64-layer forward does.

    python scripts/research/probe_lane_block_cold.py
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from yunshu_engine.kernels.tensorfold import lane_qmm  # noqa: E402

SHAPES = {  # name: (K in, N out, copies)
    "mlp_up": (5120, 17408, 24),
    "mlp_down": (17408, 5120, 24),
    "gdn_qkv": (5120, 10240, 40),
    "attn_o": (6144, 5120, 64),
}
ROWS = (8, 16, 17, 24, 32, 40, 48)


def timed(fn, reps=5):
    mx.eval(fn())
    mx.synchronize()
    best = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        best.append((time.perf_counter() - t0) * 1e3)
    best.sort()
    return round(best[len(best) // 2], 3)


def main():
    mx.random.seed(0)
    for name, (k, n, copies) in SHAPES.items():
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        base_wt = lane_qmm.tile_weight(wq, 32, 64, bits=4)
        base_sbt = lane_qmm.pack_scales(sc, bi)
        # distinct buffers (a copy each) so the cache cannot serve the next call
        weights = [(base_wt + 0, base_sbt + 0) for _ in range(copies)]
        mx.eval(weights)
        del w, wq, sc, bi
        gb = copies * (base_wt.nbytes + base_sbt.nbytes) / 1e9
        row = {"shape": name, "copies": copies, "gb": round(gb, 2)}
        for m in ROWS:
            x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)

            def f(rb, x=x):
                return [
                    lane_qmm.lane_matmul(x, wt, sbt, tiled=True, group=64, row_block=rb)
                    for wt, sbt in weights
                ]

            a, b = f(None)[0], f(16)[0]
            ms32, ms16 = timed(lambda: f(None)), timed(lambda: f(16))
            row[f"M{m}"] = {
                "b32_ms": ms32,
                "b16_ms": ms16,
                "b32_GBps": round(gb / ms32 * 1e3, 1),
                "same": bool(mx.array_equal(a, b).item()),
            }
        print(json.dumps(row), flush=True)
        weights = []
        mx.clear_cache()


main()
