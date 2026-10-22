"""Tile attention kernel variants against the served one: bits and cost by 8-token group count.

``alias``: the bf16 probabilities overwrite the fp32 scores' threadgroup memory
(a barrier between the score reads and the probability writes), 18 KB -> 12 KB
of threadgroup memory per threadgroup, same arithmetic (bit-equal expected).
``n32``: 32 keys per online-softmax step (other bits; cost direction only).

    python scripts/research/probe_tile_variants.py --contexts 32768 131072 --groups 1 2 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

H, HKV, D = 24, 4, 256

SCORES_READ = """      for (int j = 0; j < KPL; ++j) {
        const int key = ts + col + j;
        const bool ok = key >= t0 && key - kbeg < nf;
        sv[j] = ok ? scores[f * N + col + j] * s : -INFINITY;
        lmax = max(lmax, sv[j]);
      }
"""


def variant_source(base: str, name: str) -> str:
    src = base
    if name == "alias":
        old = "  threadgroup bfloat16_t probs[M * N];\n"
        assert old in src
        src = src.replace(
            old, "  threadgroup bfloat16_t* probs = (threadgroup bfloat16_t*)scores;\n"
        )
        # every thread loads its scores, then a barrier, then the probability writes
        old_if = (
            "    if (f < M) {\n      float sv[KPL];\n      float lmax = -INFINITY;\n"
            + SCORES_READ
        )
        assert old_if in src, "softmax head changed"
        new_if = (
            "    float sv[KPL];\n    float lmax = -INFINITY;\n    if (f < M) {\n"
            + SCORES_READ
            + "    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (f < M) {\n"
        )
        src = src.replace(old_if, new_if)
    elif name == "n128a":
        # 128 keys per step fit only with the aliased probabilities (24 KB)
        src = variant_source(base, "alias")
        old = "  constexpr int N = 64;               // keys per step\n"
        assert old in src
        src = src.replace(
            old, "  constexpr int N = 128;              // keys per step\n"
        )
    elif name == "n32":
        old = "  constexpr int N = 64;               // keys per step\n"
        assert old in src
        src = src.replace(
            old, "  constexpr int N = 32;               // keys per step\n"
        )
    else:
        raise ValueError(name)
    return src


def kernel(ra, name: str):
    src = ra._TILE if name == "base" else variant_source(ra._TILE, name)
    return mx.fast.metal_kernel(
        name=f"yunshu_tile_probe_{name}",
        input_names=[
            "queries",
            "keys",
            "values",
            "lengths",
            "slots",
            "scale",
            "work",
            "wstart",
        ],
        output_names=["PO", "PM", "PL"],
        source=src,
        header=ra._TILE_HEADER,
        ensure_row_contiguous=False,
    )


def launch(k, tv, q, keys, values, lengths, slots, scale, plan, nc):
    hkv, cap, d = int(keys.shape[1]), int(keys.shape[2]), int(keys.shape[3])
    h = int(q.shape[2]) // 8 * hkv
    work, wstart, nslot = plan
    return k(
        inputs=[q, keys, values, lengths, slots, scale, work, wstart],
        template=[
            ("D", d),
            ("G", h // hkv),
            ("T", 8),
            ("H", h),
            ("HKV", hkv),
            ("CAP", cap),
            ("CK", tv.CK),
            ("NC", nc),
            ("WL", True),
        ],
        grid=(256, hkv * int(work.size), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(nslot * h * d,), (nslot * h,), (nslot * h,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def timed(fn, layers: int, reps: int) -> float:
    mx.eval([fn() for _ in range(layers)])
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        mx.eval([fn() for _ in range(layers)])
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072])
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--variants", nargs="+", default=["alias", "n32"])
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import ragged_attention as ra

    kernels = {name: kernel(ra, name) for name in ["base", *a.variants]}
    scale = tv._scale_array(D**-0.5)
    ok_all = True
    for ctx in a.contexts:
        cap = ctx + 512
        keys = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
        values = mx.random.normal((1, HKV, cap, D)).astype(mx.bfloat16)
        tail_start = (ctx // tv.CK) * tv.CK
        for groups in a.groups:
            q = mx.random.normal((groups, HKV, 8 * (H // HKV), D)).astype(mx.bfloat16)
            lengths, slots, plan, nc = tv._shared_plan(tail_start, groups)
            mx.eval(keys, values, q)
            row = {"context": ctx, "groups": groups}
            ref = launch(
                kernels["base"], tv, q, keys, values, lengths, slots, scale, plan, nc
            )
            mx.eval(ref)
            for name, k in kernels.items():

                def fn(k=k):
                    return launch(
                        k, tv, q, keys, values, lengths, slots, scale, plan, nc
                    )

                row[f"{name}_ms"] = round(timed(fn, a.layers, a.reps), 3)
                if name != "base":
                    out = fn()
                    mx.eval(out)
                    equal = all(
                        mx.array_equal(x, y).item()
                        for x, y in zip(out, ref, strict=True)
                    )
                    row[f"{name}_bit_equal"] = bool(equal)
                    if name == "alias":
                        ok_all &= equal
            print(json.dumps(row), flush=True)
    print(json.dumps({"complete": True, "alias_bit_equal": ok_all}), flush=True)


if __name__ == "__main__":
    main()
