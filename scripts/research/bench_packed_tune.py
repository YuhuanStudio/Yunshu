"""Split-K / partition geometry of the packed small-row kernel, by shape and row count.

``qwen35_packed_linear._small_geometry(K, N)`` picks the K splits and in-threadgroup
partitions of the tensor-unit kernel that serves 1..8 rows (one row is padded to
two). Both depend only on the weight's shape, never on the row count, so any
choice keeps every row's bits independent of how many rows share the call (the
lane's invariance) — the choice is a free performance knob per shape. This times
each (ksplit, parts) at the 27B projection shapes, rows 2 / 4 / 7 / 8, over
distinct weight buffers so each call streams from DRAM, and reports GB/s.

    python scripts/research/bench_packed_tune.py --output runs/packed-tune.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from yunshu_engine.kernels import lane_linear  # noqa: E402
from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl  # noqa: E402

SHAPES = {  # name: (K, N)
    "gate_up(34816)": (5120, 34816),
    "gate/up(17408)": (5120, 17408),
    "down(5120)": (17408, 5120),
    "qkv+z(16384)": (5120, 16384),
    "out/o(5120)": (6144, 5120),
    "q+kv(14336)": (5120, 14336),
}


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def run(k, n, ksplit, parts, x, stores):
    sg = pl.TILE_N // 16
    outs = []
    for w, sc, bi in stores:
        (y,) = pl._kernels()["small"](
            inputs=[x, w, sc, bi],
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
            output_shapes=[
                (x.shape[0], n) if ksplit == 1 else (ksplit * x.shape[0] * n,)
            ],
            output_dtypes=[mx.bfloat16 if ksplit == 1 else mx.float32],
        )
        outs.append(pl._reduce(y, ksplit, x.shape[0], n))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 7, 8])
    ap.add_argument("--shapes", default=",".join(SHAPES))
    a = ap.parse_args()
    mx.random.seed(0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for name in a.shapes.split(","):
        k, n = SHAPES[name]
        stores, lanes, mods = [], [], []
        for _ in range(a.copies):
            q = nn.QuantizedLinear.from_linear(
                nn.Linear(k, n, bias=False), group_size=64, bits=4
            )
            mx.eval(q.parameters())
            lanes.append(lane_linear.LaneLinear.from_quantized(q))
            p = pl._pack([q])[0]
            mods.append(p)
            stores.append((p.packed_weight, p.packed_scales, p.packed_biases))
            mx.eval(lanes[-1].weight, lanes[-1].sbt)
            del q
        nbytes = sum(t.nbytes for t in stores[0])
        steps = k // 256
        tiles = n // pl.TILE_N
        default = pl._small_geometry(k, n)
        for m in a.rows:
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            mx.eval(x)
            results = []
            for ks in divisors(steps):
                if tiles * ks > 4200:
                    continue
                for parts in (1, 2):
                    if parts == 2 and (steps // ks) % 2:
                        continue
                    try:
                        mx.eval(run(k, n, ks, parts, x, stores[:2]))
                        best = 1e9
                        for _ in range(4):
                            mx.synchronize()
                            t = time.perf_counter()
                            mx.eval(run(k, n, ks, parts, x, stores))
                            best = min(best, (time.perf_counter() - t) / len(stores))
                        results.append((best, ks, parts))
                    except Exception as e:  # noqa: BLE001
                        results.append((1e9, ks, parts, str(e)[:60]))
            results.sort(key=lambda r: r[0])

            def bench_mods(fn):
                mx.eval(fn())
                best = 1e9
                for _ in range(4):
                    mx.synchronize()
                    t = time.perf_counter()
                    mx.eval(fn())
                    best = min(best, (time.perf_counter() - t) / len(stores))
                return best

            lane_t = bench_mods(lambda: [ll(x) for ll in lanes])
            if m == 1:  # as served: the lane pads one row to two
                pk_t = bench_mods(
                    lambda: [
                        mm(mx.concatenate([x, mx.zeros_like(x)]))[:1] for mm in mods
                    ]
                )
            else:
                pk_t = bench_mods(lambda: [mm(x) for mm in mods])
            row = {
                "lane_us": round(lane_t * 1e6, 1),
                "lane_GBps": round(nbytes / lane_t / 1e9, 1),
                "packed_module_us": round(pk_t * 1e6, 1),
                "packed_module_GBps": round(nbytes / pk_t / 1e9, 1),
                "shape": name,
                "K": k,
                "N": n,
                "rows": m,
                "default": list(default),
                "MB": round(nbytes / 1e6, 1),
                "best": [
                    {
                        "ksplit": r[1],
                        "parts": r[2],
                        "us": round(r[0] * 1e6, 1),
                        "GBps": round(nbytes / r[0] / 1e9, 1),
                    }
                    for r in results[:4]
                ],
            }
            dflt = next((r for r in results if (r[1], r[2]) == tuple(default)), None)
            if dflt:
                row["default_us"] = round(dflt[0] * 1e6, 1)
                row["default_GBps"] = round(nbytes / dflt[0] / 1e9, 1)
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        del stores
        mx.clear_cache()


if __name__ == "__main__":
    main()
