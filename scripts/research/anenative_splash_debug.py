"""Stage-by-stage check of the ANE function against numpy on the staged int8 surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yunshu_engine.ane import ffn_program as P
from yunshu_engine.ane.ffn_split import AneFfn
from anenative_splash_micro import load_layers  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--units", type=int, default=8)
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    import mlx.core as mx

    layers = load_layers(a.model, 40, 2)
    split = AneFfn(layers, a.units, a.cache, function_rows=[a.rows])
    shape, mem = split.shape, split.memory
    rows = a.rows
    rng = np.random.default_rng(3)
    x = mx.array(rng.normal(0, 1.0, (rows, shape.hidden)).astype(np.float32)).astype(
        mx.bfloat16
    )
    mx.eval(x)
    split.begin()
    out = split.add(0, x, None, rows)
    out1 = split.add(1, AneFfn.after(x, out), None, rows)
    mx.eval(out, out1)
    ok = split.finish()
    print("finish", ok, split.reason, flush=True)
    for li, o in ((0, out), (1, out1)):
        g_ = split.gpu_alone(li, x, None)
        mx.eval(g_)
        ga, oa = np.array(g_.astype(mx.float32)), np.array(o.astype(mx.float32))
        print(
            "layer",
            li,
            "rel err split vs gpu alone:",
            np.linalg.norm(oa - ga) / np.linalg.norm(ga),
            flush=True,
        )

    def surf(s):
        return np.array(split.alias(s))

    # inputs
    xq = np.concatenate([surf(s)[:, :rows] for s in mem.inputs], axis=0).astype(
        np.float32
    )  # [hidden, rows]
    tx = surf(mem.token_scale)[0, :rows].astype(np.float32)
    w = mem.sets[0]
    wg = np.concatenate([surf(s) for s in w["gate"]], axis=1).astype(np.float32)
    wu = np.concatenate([surf(s) for s in w["up"]], axis=1).astype(np.float32)
    sgv = surf(w["gate_scale"])[:, 0].astype(np.float32)
    suv = surf(w["up_scale"])[:, 0].astype(np.float32)
    wd = np.concatenate([surf(s) for s in w["down"]], axis=1).astype(np.float32)
    sdv = surf(w["down_scale"])[:, 0].astype(np.float32)
    partial = surf(mem.partial)[:, :rows].astype(np.float32)
    print("shapes", xq.shape, wg.shape, wd.shape, partial.shape, flush=True)
    print("tx[:4]", tx[:4], "sg[:4]", sgv[:4], "sd[:4]", sdv[:4], flush=True)
    print(
        "abs max: xq",
        np.abs(xq).max(),
        "wg",
        np.abs(wg).max(),
        "wd",
        np.abs(wd).max(),
        "partial",
        np.abs(partial).max(),
        flush=True,
    )

    # numpy model of the ANE function
    g = (wg @ xq) / 16384.0 * sgv[:, None] * tx[None, :]
    u = (wu @ xq) / 16384.0 * suv[:, None]
    gh = 0.5 * g
    silu = gh * (1 + np.tanh(gh))
    h = silu * u
    # intermediate rotation (blocks of 512) + int8 quantization
    r512 = P.rotation_matrix(512, P.rotation_signs())
    hr = (
        (h.reshape(-1, 512, rows).transpose(0, 2, 1) @ r512.T)
        .transpose(0, 2, 1)
        .reshape(-1, rows)
    )
    peak = np.maximum(np.abs(hr).max(axis=0), P.INTERMEDIATE_FLOOR)
    hq = np.clip(np.rint(hr * (127.0 / peak)), -127, 127)
    ds = (wd @ hq) / 16384.0 * sdv[:, None]
    hscale = peak * (128.0 / 127.0)
    print(
        "rel err ds vs partial[:hidden]:",
        np.linalg.norm(ds - partial[:-1]) / np.linalg.norm(ds),
        flush=True,
    )
    print(
        "rel err hscale vs partial[-1]:",
        np.linalg.norm(hscale - partial[-1]) / np.linalg.norm(hscale),
        flush=True,
    )
    for name, ref in (("g", g), ("h", h)):
        print(name, "absmax", np.abs(ref).max(), flush=True)
    # end result vs float reference
    gpu = split.gpu_alone(0, x, None)
    mx.eval(gpu)
    gpu_np = np.array(gpu.astype(mx.float32))
    out_np = np.array(out.astype(mx.float32))
    print(
        "rel err split vs gpu alone (whole FFN):",
        np.linalg.norm(out_np - gpu_np) / np.linalg.norm(gpu_np),
        flush=True,
    )
    print(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
