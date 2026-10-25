"""W8A8 (Splash-style, lossy) ANE slice probe: int8 weights + int8 activations via Core ML activation
quantization. Reports predict time and L2 error vs an fp64 reference. coremltools venv, under gpuq."""

import argparse
import json
import sys
import time

import coremltools as ct
import numpy as np

sys.path.insert(0, "python")
from yunshu_engine.ane import convert as C  # noqa: E402, N812


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--share", type=float, default=0.1875)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument(
        "--out", default="/Volumes/P5Plus/yunshu-build/ane-models/split/w8a8"
    )
    a = ap.parse_args()
    import os

    os.makedirs(a.out, exist_ok=True)
    L = C.load_mlp_layer(a.model, a.layer)
    inter = L["gate_proj"]["weight"].shape[0]
    n = C.split_channels(inter, a.share)
    lo = inter - n
    g, u, d = C.mlp_slices(L, lo, "int8")
    base = C.build_mlp_model(g, u, d, a.rows, compress="int8")
    rng = np.random.default_rng(0)
    name = base.get_spec().description.input[0].name
    oname = base.get_spec().description.output[0].name
    samples = [
        {name: (rng.standard_normal((1, a.rows, 5120))).astype(np.float16)}
        for _ in range(4)
    ]
    cfg = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.experimental.OpActivationLinearQuantizerConfig(
            mode="linear_symmetric"
        )
    )
    t0 = time.time()
    q = ct.optimize.coreml.experimental.linear_quantize_activations(base, cfg, samples)
    path = f"{a.out}/l{a.layer}_n{n}_r{a.rows}.mlpackage"
    q.save(path)
    print("calibrate+convert s", round(time.time() - t0, 1), flush=True)
    W = {
        k: C.dequantize_affine(
            v["weight"], v["scales"], v["biases"], v["bits"], v["group_size"]
        ).astype(np.float64)
        for k, v in L.items()
    }
    x = samples[0][name]
    x64 = x[0].astype(np.float64)
    ag = x64 @ W["gate_proj"][lo:].T
    ref = (ag / (1 + np.exp(-ag)) * (x64 @ W["up_proj"][lo:].T)) @ W["down_proj"][
        :, lo:
    ].T
    res = {}
    for label, p in (("int8w_fp16a", None), ("w8a8", path)):
        if p is None:
            m = (
                ct.models.MLModel(base._spec if False else f"{a.out}/base.mlpackage")
                if False
                else None
            )
            base.save(f"{a.out}/base_l{a.layer}_n{n}_r{a.rows}.mlpackage")
            p = f"{a.out}/base_l{a.layer}_n{n}_r{a.rows}.mlpackage"
        m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        y = m.predict({name: x})[oname]
        ts = []
        for _ in range(15):
            t0 = time.perf_counter()
            y = m.predict({name: x})[oname]
            ts.append((time.perf_counter() - t0) * 1e3)
        y = np.asarray(y, dtype=np.float64)[0]
        res[label] = dict(
            ms_min=round(min(ts), 2),
            l2=float(np.linalg.norm(y - ref) / np.linalg.norm(ref)),
        )
    print(json.dumps(dict(n=n, rows=a.rows, **res)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
