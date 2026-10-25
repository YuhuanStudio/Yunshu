"""GPU/ANE FFN split microbenchmark on real layer weights: per-layer FFN time GPU-only vs split.

Runs in /Volumes/P5Plus/venvs/anesplit-mlx (mlx + coremltools). Under gpuq only.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import coremltools as ct
import mlx.core as mx
import numpy as np

sys.path.insert(0, "python")
from yunshu_engine.ane import convert as C  # noqa: E402, N812


def qmm(x, p, lo=None, hi=None, axis=0):
    """Quantized matmul with the projection's rows [lo:hi] (axis 0) or input columns [lo:hi] (axis 1)."""
    w, s, b = p["w"], p["s"], p["b"]
    bits, gs = p["bits"], p["gs"]
    if lo is not None and axis == 0:
        w, s, b = w[lo:hi], s[lo:hi], b[lo:hi]
    if lo is not None and axis == 1:
        w, s, b = (
            w[:, lo * bits // 32 : hi * bits // 32],
            s[:, lo // gs : hi // gs],
            b[:, lo // gs : hi // gs],
        )
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)


def gpu_ffn(x, L, n_gpu=None):
    g = L["gate_proj"]
    u = L["up_proj"]
    d = L["down_proj"]
    hi = n_gpu
    a = qmm(x, g, 0, hi) if hi else qmm(x, g)
    b = qmm(x, u, 0, hi) if hi else qmm(x, u)
    h = (a * mx.sigmoid(a)) * b
    return qmm(h, d, 0, hi, axis=1) if hi else qmm(h, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    ap.add_argument("--layers", default="40,41,42,43")
    ap.add_argument("--shares", default="0.125,0.1875,0.25,0.3125")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--mode", default="lut")
    ap.add_argument(
        "--out", default="/Volumes/P5Plus/yunshu-build/ane-models/split/micro"
    )
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layers = [int(v) for v in a.layers.split(",")]
    Ls, raw = {}, {}
    for l in layers:
        r = C.load_mlp_layer(a.model, l)
        raw[l] = r
        Ls[l] = {
            k: dict(
                w=mx.array(v["weight"]),
                s=mx.array(v["scales"]).astype(mx.bfloat16),
                b=mx.array(v["biases"]).astype(mx.bfloat16),
                bits=v["bits"],
                gs=v["group_size"],
            )
            for k, v in r.items()
        }
    # my numpy dequant must equal mx.dequantize
    p = Ls[layers[0]]["gate_proj"]
    ref = np.array(
        mx.dequantize(
            p["w"], p["s"], p["b"], group_size=p["gs"], bits=p["bits"]
        ).astype(mx.float32)
    )
    mine = C.dequantize_affine(
        raw[layers[0]]["gate_proj"]["weight"],
        raw[layers[0]]["gate_proj"]["scales"],
        raw[layers[0]]["gate_proj"]["biases"],
        p["bits"],
        p["gs"],
    )
    print(
        "dequant max abs diff vs mx (bf16 scale math)",
        float(np.abs(ref - mine).max()),
        "max|w|",
        float(np.abs(ref).max()),
        flush=True,
    )
    d = Ls[layers[0]]["down_proj"]
    refd = np.array(
        mx.dequantize(
            d["w"], d["s"], d["b"], group_size=d["gs"], bits=d["bits"]
        ).astype(mx.float32)
    )
    mined = C.dequantize_affine(
        raw[layers[0]]["down_proj"]["weight"],
        raw[layers[0]]["down_proj"]["scales"],
        raw[layers[0]]["down_proj"]["biases"],
        d["bits"],
        d["gs"],
    )
    print("down bits", d["bits"], "diff", float(np.abs(refd - mined).max()), flush=True)

    rng = np.random.default_rng(0)
    x = mx.array((rng.standard_normal((a.rows, 5120)) * 0.7).astype(np.float32)).astype(
        mx.bfloat16
    )
    mx.eval(x)
    inter = Ls[layers[0]]["gate_proj"]["w"].shape[0]
    pool = ThreadPoolExecutor(1)
    res = []

    # GPU only
    def run_gpu():
        t = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            for l in layers:
                y = gpu_ffn(x, Ls[l])
                mx.eval(y)
            t.append((time.perf_counter() - t0) / len(layers) * 1e3)
        return min(t), float(np.median(t))

    run_gpu()
    gmin, gmed = run_gpu()
    print(
        json.dumps(dict(kind="gpu_only", rows=a.rows, ms_min=gmin, ms_med=gmed)),
        flush=True,
    )
    full = {l: np.array(gpu_ffn(x, Ls[l]).astype(mx.float32)) for l in layers}

    for share in [float(v) for v in a.shares.split(",")]:
        n = C.split_channels(inter, share)
        n_gpu = inter - n
        models = {}
        for l in layers:
            path = f"{a.out}/l{l}_n{n}_r{a.rows}_{a.mode}.mlpackage"
            if not os.path.exists(path):
                C.build_mlp_model(
                    *C.mlp_slices(raw[l], n_gpu, a.mode),
                    a.rows,
                    compress=a.mode,
                    out_path=path,
                )
            m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            models[l] = (
                m,
                m.get_spec().description.input[0].name,
                m.get_spec().description.output[0].name,
            )

        def ane_call(l, xs):
            m, i, o = models[l]
            return m.predict({i: xs})[o]

        def split_ffn(l):
            xs = np.asarray(x.astype(mx.float16))[None]
            fut = pool.submit(ane_call, l, xs)
            yg = gpu_ffn(x, Ls[l], n_gpu)
            mx.eval(yg)
            ya = fut.result()
            return yg + mx.array(np.asarray(ya)[0]).astype(mx.bfloat16)

        for l in layers:
            split_ffn(l)  # warm
        t = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            for l in layers:
                y = split_ffn(l)
                mx.eval(y)
            t.append((time.perf_counter() - t0) / len(layers) * 1e3)
        ys = np.array(split_ffn(layers[0]).astype(mx.float32))
        rel = float(
            np.linalg.norm(ys - full[layers[0]]) / np.linalg.norm(full[layers[0]])
        )
        # ANE alone and GPU part alone
        ta = []
        xs = np.asarray(x.astype(mx.float16))[None]
        for _ in range(a.reps):
            t0 = time.perf_counter()
            ane_call(layers[0], xs)
            ta.append((time.perf_counter() - t0) * 1e3)
        tg = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            mx.eval(gpu_ffn(x, Ls[layers[0]], n_gpu))
            tg.append((time.perf_counter() - t0) * 1e3)
        r = dict(
            kind="split",
            share=share,
            n_ane=n,
            rows=a.rows,
            mode=a.mode,
            ms_min=min(t),
            ms_med=float(np.median(t)),
            gpu_part_ms=min(tg),
            ane_alone_ms=min(ta),
            rel_l2_vs_gpu=rel,
            speedup_vs_gpu=gmin / min(t),
        )
        print(json.dumps(r), flush=True)
        res.append(r)
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
