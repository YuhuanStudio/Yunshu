"""5-bit projections on the lane matmul: K-slice count and weight tiling, by shape and rows.

The int-code lane kernel (``kernels/int_code_linear.py``) serves every 5-bit
projection of the speculative lane. Its K-slice count ``sk`` (``lane_qmm.split_k``)
and weight layout (tiled 32 columns wide or MLX's) fix each column's arithmetic
by shape alone, so any consistent choice keeps rows independent. This times each
(sk, tiled) at the 27B 5-bit shapes for 2 and 7 rows over distinct weight copies
(each call streams from DRAM) and reports GB/s.

    python scripts/research/bench_intcode_tune.py --output runs/intcode-tune.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from yunshu_engine.kernels.tensorfold import lane_qmm  # noqa: E402

SHAPES = {  # name: (K, N)
    "out_proj(6144->5120)": (6144, 5120),
    "down(17408->5120)": (17408, 5120),
    "z(5120->6144)": (5120, 6144),
    "o_proj(6144->5120)": (6144, 5120),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--copies", type=int, default=12)
    ap.add_argument("--bits", type=int, default=5)
    a = ap.parse_args()
    mx.random.seed(0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for name, (k, n) in SHAPES.items():
        mods = []
        for _ in range(a.copies):
            q = nn.QuantizedLinear.from_linear(
                nn.Linear(k, n, bias=False), group_size=64, bits=a.bits
            )
            mx.eval(q.parameters())
            w = q.weight
            wt = lane_qmm.tile_weight(w, bits=a.bits) if n % lane_qmm.NT == 0 else None
            sbt = lane_qmm.pack_scales(q.scales, q.biases)
            mx.eval(sbt, *(t for t in (wt,) if t is not None))
            mods.append((w, wt, sbt))
        nbytes = sum(t.nbytes for t in (mods[0][0], mods[0][2]))
        default = lane_qmm.split_k(n, k)
        for m in (2, 7):
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            mx.eval(x)
            rows = []
            for tiled in (False, True):
                if tiled and mods[0][1] is None:
                    continue
                for sk in (1, 2, 4, 8):

                    def run():
                        return [
                            lane_qmm.lane_matmul(
                                x, wt if tiled else w, sbt, tiled=tiled, sk=sk
                            )
                            for w, wt, sbt in mods
                        ]

                    mx.eval(run())
                    best = 1e9
                    for _ in range(4):
                        mx.synchronize()
                        t = time.perf_counter()
                        mx.eval(run())
                        best = min(best, (time.perf_counter() - t) / len(mods))
                    rows.append(
                        {
                            "tiled": tiled,
                            "sk": sk,
                            "us": round(best * 1e6, 1),
                            "GBps": round(nbytes / best / 1e9, 1),
                        }
                    )
            rows.sort(key=lambda r: r["us"])
            row = {
                "shape": name,
                "bits": a.bits,
                "rows": m,
                "MB": round(nbytes / 1e6, 1),
                "default_sk": default,
                "best": rows[:4],
                "worst_us": rows[-1]["us"],
            }
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        del mods
        mx.clear_cache()


if __name__ == "__main__":
    main()
