"""ANE slice probe: build MLP slice packages (layer L, share f, rows) in several weight encodings, check
MLComputePlan device placement, time predict, and compare against the fp32 reference.

Run with the coremltools venv: /Volumes/P5Plus/venvs/anesplit-venv/bin/python (ANE => under gpuq).
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "python")
from yunshu_engine.ane import convert as C  # noqa: E402, N812


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--modes", default="fp16,lut,int8")
    ap.add_argument(
        "--out", default="/Volumes/P5Plus/yunshu-build/ane-models/split/probe"
    )
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    import os

    import coremltools as ct

    os.makedirs(a.out, exist_ok=True)
    L = C.load_mlp_layer(a.model, a.layer)
    print({k: (v["bits"], v["group_size"]) for k, v in L.items()}, flush=True)
    W = {
        k: C.dequantize_affine(
            v["weight"], v["scales"], v["biases"], v["bits"], v["group_size"]
        )
        for k, v in L.items()
    }
    inter = W["gate_proj"].shape[0]
    n = C.split_channels(inter, a.share)
    lo, hi = inter - n, inter  # ANE takes the trailing channels
    g, u, d = C.mlp_slice(W["gate_proj"], W["up_proj"], W["down_proj"], lo, hi)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, a.rows, g.shape[1])) * 0.5).astype(np.float16)
    xf = x[0].astype(np.float32)
    ref = ((xf @ g.T) / (1 + np.exp(-(xf @ g.T)))) * (xf @ u.T)
    ref = ref @ d.T
    rows = []
    for mode in a.modes.split(","):
        path = f"{a.out}/l{a.layer}_n{n}_r{a.rows}_{mode}.mlpackage"
        t0 = time.time()
        args = C.mlp_slices(L, lo, mode)
        m = C.build_mlp_model(*args, a.rows, compress=mode, out_path=path)
        tconv = time.time() - t0
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        name = m.get_spec().description.input[0].name
        oname = m.get_spec().description.output[0].name
        t0 = time.time()
        y = m.predict({name: x})[oname]
        tfirst = time.time() - t0
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            y = m.predict({name: x})[oname]
            ts.append((time.perf_counter() - t0) * 1e3)
        y = np.asarray(y, dtype=np.float32)[0]
        err = float(np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9))
        rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
        # compute plan: where each op is placed
        dev = {}
        try:
            cp = ct.models.MLModel(
                path, compute_units=ct.ComputeUnit.CPU_AND_NE
            ).get_compiled_model_path()
            plan = ct.models.compute_plan.MLComputePlan.load_from_path(
                cp, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            fn = plan.model_structure.program.functions["main"]
            for op in fn.block.operations:
                du = plan.get_compute_device_usage_for_mlprogram_operation(op)
                k = type(du.preferred_compute_device).__name__ if du else "const"
                dev[op.operator_name + ":" + k] = (
                    dev.get(op.operator_name + ":" + k, 0) + 1
                )
        except Exception as e:  # noqa: BLE001
            dev = {"plan_error": str(e)[:160]}
        r = dict(
            mode=mode,
            n=n,
            rows=a.rows,
            convert_s=round(tconv, 1),
            first_ms=round(tfirst * 1e3, 1),
            ms_min=round(min(ts), 2),
            ms_med=round(float(np.median(ts)), 2),
            maxrel=err,
            l2rel=rel,
            size_mb=round(
                sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(path)
                    for f in fs
                )
                / 1e6,
                1,
            ),
            devices=dev,
        )
        print(json.dumps(r), flush=True)
        rows.append(r)
    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
