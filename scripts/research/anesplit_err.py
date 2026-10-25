"""Where does the ANE slice error come from? Four numbers for one FFN slice (layer, trailing share):
(a) fp32/fp64 reference on dequantized weights, (b) numpy dequant vs mx.dequantize, (c) ANE fp16 vs the
reference (CPU_AND_NE and CPU_ONLY, input scales, max |intermediate|), (d) the GPU quantized path vs the
reference. Run in /Volumes/P5Plus/venvs/anesplit-mlx under gpuq."""

import argparse
import json
import os
import sys

import coremltools as ct
import mlx.core as mx
import numpy as np

sys.path.insert(0, "python")
from yunshu_engine.ane import convert as C  # noqa: E402, N812


def l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument(
        "--out", default="/Volumes/P5Plus/yunshu-build/ane-models/split/err"
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = C.load_mlp_layer(a.model, a.layer)
    P = {
        k: dict(
            w=mx.array(v["weight"]),
            s=mx.array(v["scales"]).astype(mx.bfloat16),
            b=mx.array(v["biases"]).astype(mx.bfloat16),
            bits=v["bits"],
            gs=v["group_size"],
        )
        for k, v in L.items()
    }
    out = {}
    # (b) dequant tensors vs mx.dequantize
    W = {}
    for k, v in L.items():
        mine = C.dequantize_affine(
            v["weight"], v["scales"], v["biases"], v["bits"], v["group_size"]
        )
        p = P[k]
        ref = np.array(
            mx.dequantize(
                p["w"], p["s"], p["b"], group_size=p["gs"], bits=p["bits"]
            ).astype(mx.float32)
        )
        out[f"dequant_maxdiff_{k}"] = float(np.abs(ref - mine).max())
        out[f"dequant_maxabs_{k}"] = float(np.abs(ref).max())
        W[k] = ref.astype(np.float64)  # the reference uses mx's own dequantized values
    inter = W["gate_proj"].shape[0]
    n = C.split_channels(inter, a.share)
    lo = inter - n
    rng = np.random.default_rng(0)
    for scale in (0.05, 0.5, 2.0):
        x16 = (rng.standard_normal((a.rows, 5120)) * scale).astype(np.float16)
        x64 = x16.astype(np.float64)
        ag = x64 @ W["gate_proj"][lo:].T
        au = x64 @ W["up_proj"][lo:].T
        h = ag / (1 + np.exp(-ag)) * au
        ref = h @ W["down_proj"][:, lo:].T
        rec = dict(
            scale=scale,
            max_abs_h=float(np.abs(h).max()),
            ref_rms=float(np.sqrt((ref**2).mean())),
        )
        # (d) GPU quantized path on the same channel slice, bf16 activations
        xb = mx.array(x16.astype(np.float32)).astype(mx.bfloat16)

        def q(x, p, rows=None, cols=None):
            w_, s_, b_ = p["w"], p["s"], p["b"]
            if rows:
                w_, s_, b_ = (
                    w_[rows[0] : rows[1]],
                    s_[rows[0] : rows[1]],
                    b_[rows[0] : rows[1]],
                )
            if cols:
                g = p["gs"]
                w_ = w_[:, cols[0] * p["bits"] // 32 : cols[1] * p["bits"] // 32]
                s_, b_ = (
                    s_[:, cols[0] // g : cols[1] // g],
                    b_[:, cols[0] // g : cols[1] // g],
                )
            return mx.quantized_matmul(
                x, w_, s_, b_, transpose=True, group_size=p["gs"], bits=p["bits"]
            )

        g_ = q(xb, P["gate_proj"], rows=(lo, inter))
        u_ = q(xb, P["up_proj"], rows=(lo, inter))
        hh = (g_ * mx.sigmoid(g_)) * u_
        yg = np.array(
            q(hh, P["down_proj"], cols=(lo, inter)).astype(mx.float32)
        ).astype(np.float64)
        rec["gpu_vs_ref_l2"] = l2(yg, ref)
        # (c) ANE
        args = C.mlp_slices(L, lo, "fp16")
        path = f"{a.out}/l{a.layer}_n{n}_r{a.rows}.mlpackage"
        C.build_mlp_model(*args, a.rows, compress="fp16", out_path=path)
        for name, cu in (
            ("ne", ct.ComputeUnit.CPU_AND_NE),
            ("cpu", ct.ComputeUnit.CPU_ONLY),
        ):
            m = ct.models.MLModel(path, compute_units=cu)
            i = m.get_spec().description.input[0].name
            o = m.get_spec().description.output[0].name
            y = np.asarray(m.predict({i: x16[None]})[o], dtype=np.float64)[0]
            rec[f"{name}_vs_ref_l2"] = l2(y, ref)
        # fp16-weight reference in fp64 (weights rounded to fp16 only)
        g16 = W["gate_proj"][lo:].astype(np.float16).astype(np.float64)
        u16 = W["up_proj"][lo:].astype(np.float16).astype(np.float64)
        d16 = W["down_proj"][:, lo:].astype(np.float16).astype(np.float64)
        ag = x64 @ g16.T
        h2 = ag / (1 + np.exp(-ag)) * (x64 @ u16.T)
        rec["fp16_weights_only_l2"] = l2(h2 @ d16.T, ref)
        out.setdefault("cases", []).append(rec)
        print(json.dumps(rec), flush=True)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
