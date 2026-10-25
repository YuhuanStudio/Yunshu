"""Small-M lane matmul vs MLX's quantized_matmul on the 27B projection shapes (run only via gpuq).

For each (name, N, K, bits) and M in --rows: GB/s of the lane kernel (default K split, other K splits,
64-wide coop tiles when the weight allows) and of ``mx.quantized_matmul``, weights streamed (--copies
distinct copies, cycled) so the system cache does not serve them. Reports the K split's effect on the
lane kernel's bits too (a different split sums the K groups in another order).

    python scripts/research/lane_small_m_sweep.py --output out.jsonl
"""

import argparse
import json
import time
from pathlib import Path

SHAPES = [
    ("mlp.gate_up", 17408, 5120, 4),
    ("mlp.down4", 5120, 17408, 4),
    ("mlp.down5", 5120, 17408, 5),
    ("gdn.qkv", 10240, 5120, 4),
    ("gdn.z", 6144, 5120, 5),
    ("gdn.out5", 5120, 6144, 5),
]


def gbps(nbytes: float, seconds: float) -> float:
    return nbytes / seconds / 1e9


def summarize(times: list[float]) -> float:
    """Median of positive timings."""
    ts = sorted(t for t in times if t > 0)
    return ts[len(ts) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--copies", type=int, default=12)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    import sys

    import mlx.core as mx
    import mlx.nn as nn

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")
    for name, n, k, bits in SHAPES:
        lins, stock = [], []
        for c in range(a.copies):
            w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
            q = nn.QuantizedLinear.from_linear(
                nn.Linear(k, n, bias=False), group_size=64, bits=bits
            )
            q.weight, q.scales, q.biases = mx.quantize(w, group_size=64, bits=bits)
            mx.eval(q.parameters())
            lins.append(lane_linear.LaneLinear.from_quantized(q))
            stock.append(q)
        mx.eval([m.parameters() for m in lins])
        w64 = (
            [lane_qmm.tile_weight(q.weight, 64, 64, bits=bits) for q in stock]
            if bits == 4 and n % 64 == 0
            else []
        )
        mx.eval(w64)
        nbytes = lins[0].weight.nbytes + lins[0].sbt.nbytes
        sk0 = lane_qmm.split_k(n, k)
        for m in a.rows:
            x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            arms = {"stock": lambda i: stock[i](x)}
            for sk in sorted({1, 2, 4, 8}):
                arms[f"lane_sk{sk}{'*' if sk == sk0 else ''}"] = lambda i, sk=sk: (
                    lane_qmm.lane_matmul(
                        x, lins[i].weight, lins[i].sbt, tiled=lins[i].tiled, sk=sk
                    )
                )

            def fresh(i):
                lane_qmm._xs_cache.clear()  # a new input per call, as in a forward: its group sums are recomputed
                return lane_qmm.lane_matmul(
                    x, lins[i].weight, lins[i].sbt, tiled=lins[i].tiled
                )

            arms["lane_default_fresh_x"] = fresh
            if bits == 4 and n % 64 == 0:
                for sk in (1, 2, 4, 8):
                    arms[f"coop64_sk{sk}"] = lambda i, sk=sk: lane_qmm.lane_matmul(
                        x, w64[i], lins[i].sbt, tiled=True, nt=64, sk=sk
                    )
            ref = lane_qmm.lane_matmul(
                x, lins[0].weight, lins[0].sbt, tiled=lins[0].tiled
            )
            row = {
                "shape": name,
                "n": n,
                "k": k,
                "bits": bits,
                "m": m,
                "MB": nbytes / 1e6,
                "arms": {},
            }
            for arm, fn in arms.items():
                try:
                    for i in range(a.copies):
                        mx.eval(fn(i))
                    times = []
                    for _ in range(a.iters):
                        ys = [fn(i) for i in range(a.copies)]
                        mx.synchronize()
                        t0 = time.perf_counter()
                        mx.eval(ys)
                        times.append((time.perf_counter() - t0) / a.copies)
                    t = summarize(times)
                    same = None
                    if arm.startswith("lane"):
                        same = bool(mx.array_equal(fn(0), ref).item())
                    row["arms"][arm] = {
                        "us": t * 1e6,
                        "GBps": gbps(nbytes, t),
                        "same_bits_as_default": same,
                    }
                except Exception as exc:  # a split a shape cannot take
                    row["arms"][arm] = {"error": str(exc)[:120]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                name,
                m,
                {k: round(v.get("GBps", 0)) for k, v in row["arms"].items()},
                flush=True,
            )
    print("complete: true")


if __name__ == "__main__":
    main()
