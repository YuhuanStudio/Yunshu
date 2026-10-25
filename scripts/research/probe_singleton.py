"""Flash-Next's multi-row dense projections: per-row singleton qmv (gather_qmm) vs one lane matmul (gpuq only).

The Qwen4 verify forward runs every dense quantized projection as ``singleton_quantized_linear``: one
``gather_qmm`` whose rhs index is zero for every row, i.e. the weight is read once per row so that each row gets
the 1-row decode bits. This times that call for T rows against the lane matmul (row-invariant, one weight pass)
on the Flash-Next oQ3e dense shapes, weights streamed from a chain of distinct copies. Reports us per call and
the ratio to the 1-row call: a flat ratio means the rows share the weight pass, a ratio near T means they do not.
"""

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

SHAPES = [
    # name, n, k, bits, group, layers per round
    ("gdn.qkv", 10240, 2560, 8, 64, 36),
    ("gdn.z", 6144, 2560, 8, 64, 36),
    ("attn.q", 12288, 2560, 8, 64, 12),
    ("attn.o", 2560, 6144, 8, 64, 12),
    ("gdn.out", 2560, 6144, 5, 128, 36),
    ("shared.gate", 640, 2560, 8, 128, 48),
]


def per_row_ratio(us_by_rows):
    """Time at T rows over the time at 1 row (1.0 = free extra rows, T = no sharing)."""
    base = us_by_rows[1]
    return {t: v / base for t, v in us_by_rows.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--target-mb", type=int, default=768)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.quantized_verifier import singleton_quantized_linear

    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    lane_qmm.ready()
    mx.random.seed(0)
    for name, n, k, bits, group, layers in SHAPES:
        lin = nn.Linear(k, n, bias=False)
        lin.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        base = nn.QuantizedLinear.from_linear(lin, group_size=group, bits=bits)
        base.scales, base.biases = (
            base.scales.astype(mx.bfloat16),
            base.biases.astype(mx.bfloat16),
        )
        nbytes = base.weight.nbytes + base.scales.nbytes + base.biases.nbytes
        copies = max(2, min(64, a.target_mb * 2**20 // nbytes))
        qs = []
        for c in range(copies):
            q = copy.copy(base)
            q.weight = mx.contiguous(
                base.weight ^ mx.array(c * 2654435761 % 2**32, dtype=mx.uint32)
            )
            q.scales, q.biases = base.scales, base.biases
            q.bits, q.group_size = bits, group
            qs.append(q)
        mx.eval([q.weight for q in qs])
        lanes = []
        if lane_qmm.readable(bits, group):
            for q in qs:
                m = lane_linear.LaneLinear.from_quantized(q)
                lanes.append(m)
            mx.eval([m.weight for m in lanes])
        row = {
            "name": name,
            "n": n,
            "k": k,
            "bits": bits,
            "group": group,
            "layers": layers,
            "copies": copies,
        }
        for arm, mods, call in (
            ("singleton", qs, singleton_quantized_linear),
            ("lane", lanes, lambda m, x: m(x)),
        ):
            if not mods:
                continue
            us = {}
            for t in a.rows:
                x = (mx.random.normal((1, t, k)) * 0.5).astype(mx.bfloat16)
                mx.eval(x)

                def graph(mods=mods, call=call, x=x):
                    outs, prev = [], None
                    for m in mods:
                        xi = x if prev is None else mx.depends(x, prev)
                        prev = call(m, xi)
                        outs.append(prev)
                    return outs

                for _ in range(2):
                    mx.eval(graph())
                ts = []
                for _ in range(7):
                    g = graph()
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval(g)
                    ts.append((time.perf_counter() - t0) / copies)
                us[t] = statistics.median(ts) * 1e6
            row[arm] = {
                "us": us,
                "ratio_to_1row": per_row_ratio(us),
                "GBps_1row": nbytes / us[1] / 1e3,
            }
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
