"""Grouped lane projections equal the same projections one by one, bit for bit (gpuq only).

Random quantized projections at the 27B and Flash-Next verify shapes, every served row
count, every bit mix a layer has. ``--bench`` also times a layer's projections grouped vs
separate in one chained graph (what the round runs).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

CASES = {
    # name: [(n, k, bits, group)...] members in call order
    "gdn27_555": [(6144, 5120, 5, 64), (48, 5120, 5, 64), (48, 5120, 5, 64)],
    "gdn27_444": [(6144, 5120, 4, 64), (48, 5120, 4, 64), (48, 5120, 4, 64)],
    "gdn27_455": [(6144, 5120, 4, 64), (48, 5120, 5, 64), (48, 5120, 5, 64)],
    "gdn27_445": [(6144, 5120, 4, 64), (48, 5120, 4, 64), (48, 5120, 5, 64)],
    "gdn27_full": [
        (10240, 5120, 4, 64),
        (6144, 5120, 5, 64),
        (48, 5120, 5, 64),
        (48, 5120, 5, 64),
    ],
    "attn27_kv": [(1024, 5120, 4, 64), (1024, 5120, 4, 64)],
    "attn27_qkv": [(12288, 5120, 4, 64), (1024, 5120, 4, 64), (1024, 5120, 4, 64)],
    "attn27_kv5": [(1024, 5120, 5, 64), (1024, 5120, 4, 64)],
    "mlp27_gateup": [(17408, 5120, 4, 64), (17408, 5120, 4, 64)],
    "gdnfn_444": [(6144, 2560, 4, 32), (48, 2560, 4, 32), (48, 2560, 4, 32)],
    "gdnfn_full": [
        (10240, 2560, 4, 32),
        (6144, 2560, 4, 32),
        (48, 2560, 4, 32),
        (48, 2560, 4, 32),
    ],
    "shared_fn": [(640, 2560, 4, 32), (640, 2560, 4, 32)],
}
ROWS = [1, 2, 5, 8, 15, 16, 17, 32, 33, 48, 64, 128]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="*", default=sorted(CASES))
    p.add_argument("--bench", action="store_true")
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear, lane_virtual
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    if not q.ready():
        sys.exit("lane kernel not ready")
    mx.random.seed(3)

    def make(n, k, bits, group, salt=0):
        lin = nn.Linear(k, n, bias=False)
        lin.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        qa = nn.QuantizedLinear.from_linear(lin, group_size=group, bits=bits)
        qa.scales, qa.biases = (
            qa.scales.astype(mx.bfloat16),
            qa.biases.astype(mx.bfloat16),
        )
        m = lane_linear.LaneLinear.from_quantized(qa)
        mx.eval(m.parameters())
        return m

    failed = 0
    for name in a.cases:
        spec = CASES[name]
        members = [make(*s) for s in spec]
        k = spec[0][1]
        bad, grouped_any = [], False
        for rows in ROWS:
            x = (mx.random.normal((1, rows, k)) * 0.5).astype(mx.bfloat16)
            q.PIPE[0] = False
            ref = [m(x) for m in members]
            for pipe in (False, True):
                q.PIPE[0] = pipe
                got = lane_virtual.grouped_linears(members, x)
                if got is None:
                    continue
                grouped_any = True
                mx.eval(*got, *ref)
                for i, (g, r) in enumerate(zip(got, ref, strict=True)):
                    if g.shape != r.shape or not bool(mx.array_equal(g, r).item()):
                        bad.append((rows, i, "pipe" if pipe else "plain"))
            q.PIPE[0] = False
        row = {"case": name, "grouped": grouped_any, "mismatch": bad}
        if a.bench and grouped_any:
            x = (mx.random.normal((1, 16, k)) * 0.5).astype(mx.bfloat16)
            layers = [[make(*s) for s in spec] for _ in range(a.layers)]
            mx.eval(x)

            def chain(use_group):
                h, outs = x, []
                for ms in layers:
                    ys = (
                        lane_virtual.grouped_linears(ms, h)
                        if use_group
                        else tuple(m(h) for m in ms)
                    )
                    outs.append(ys)
                    h = mx.depends(x, ys[-1])
                return outs

            for flag in (True, False):
                mx.eval(chain(flag))
            res = {}
            for rep in range(a.reps):
                for flag in (False, True) if rep % 2 else (True, False):
                    g = chain(flag)
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval(g)
                    res.setdefault(flag, []).append(
                        (time.perf_counter() - t0) / a.layers * 1e6
                    )
            row["us_per_layer_grouped"] = statistics.median(res[True])
            row["us_per_layer_separate"] = statistics.median(res[False])
        a.out.open("a").write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        failed += bool(bad)
    if failed:
        sys.exit(f"{failed} case(s) differ from the separate projections")


if __name__ == "__main__":
    main()
