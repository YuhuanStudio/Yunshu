"""Cost of the fast tree's shared-prefix attention launch by 8-token group count.

The fused tree attention serves a window's 8-token groups with one tile launch over
the shared prefix (chunk-major work list). This times that launch alone at long
contexts for 1, 2 and 4 groups, chunk-major and group-major item orders, and one
launch per group, to tell whether a group's extra cost is a second K/V read or
the tile kernel's compute. Each sample evaluates ``--layers`` independent launches
in one graph (a verify's attention layers), so per-call submit latency is amortized.

    python scripts/research/probe_tile_groups.py --contexts 32768 131072 --groups 1 2 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

H, HKV, D = 24, 4, 256


def plan_arrays(tv, tail_start: int, groups: int, order: str):
    lengths, nc, items, starts, total = tv.shared_plan_lists(tail_start, groups)
    if order == "group":
        items = sorted(items, key=lambda it: (it // nc, it % nc))
    return (
        mx.array(lengths, dtype=mx.int32),
        mx.zeros((groups,), dtype=mx.int32),
        (mx.array(items, dtype=mx.int32), mx.array(starts, dtype=mx.int32), total),
        nc,
    )


def sample(fn, layers: int, reps: int) -> float:
    outs = [fn() for _ in range(layers)]
    mx.eval(outs)
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        outs = [fn() for _ in range(layers)]
        mx.eval(outs)
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072])
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    from yunshu_engine import tree_verify as tv

    g = H // HKV
    scale = tv._scale_array(D**-0.5)
    for ctx in a.contexts:
        cap = ctx + 512
        keys = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
        values = mx.random.normal((1, HKV, cap, D)).astype(mx.bfloat16)
        mx.eval(keys, values)
        tail_start = (ctx // tv.CK) * tv.CK
        kv_gb = 2 * HKV * tail_start * D * 2 / 1e9
        for groups in a.groups:
            q = mx.random.normal((groups, HKV, 8 * g, D)).astype(mx.bfloat16)
            mx.eval(q)
            row = {"context": ctx, "groups": groups, "kv_gb_per_layer": round(kv_gb, 3)}
            for order in ("chunk", "group"):
                lengths, slots, plan, nc = plan_arrays(tv, tail_start, groups, order)

                def fn(lengths=lengths, slots=slots, plan=plan, nc=nc):
                    return tv._tile_partials(
                        q, keys, values, lengths, slots, scale, plan, 8, nc
                    )[:3]

                ms = sample(fn, a.layers, a.reps)
                row[f"{order}_ms"] = round(ms, 3)
                row[f"{order}_gbps"] = round(kv_gb * a.layers / (ms / 1000.0), 1)
            if groups > 1:
                one = plan_arrays(tv, tail_start, 1, "chunk")

                def fn_sep(one=one):
                    return [
                        tv._tile_partials(
                            q[i : i + 1],
                            keys,
                            values,
                            one[0],
                            one[1],
                            scale,
                            one[2],
                            8,
                            one[3],
                        )[:3]
                        for i in range(groups)
                    ]

                row["separate_ms"] = round(sample(fn_sep, a.layers, a.reps), 3)
            print(json.dumps(row), flush=True)
    print(json.dumps({"complete": True}), flush=True)


if __name__ == "__main__":
    main()
