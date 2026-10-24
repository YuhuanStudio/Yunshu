"""Resident-input GPU/ANE FFN overlap eligibility probe (gpuq only).

Inputs are preloaded on both devices. This measures resource contention and host
orchestration, excluding producer/consumer handoff and join. It is not an engine
speedup or a completed zero-copy MLX integration.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from yunshu_engine.ane import convert
from yunshu_engine.ane.native import NativeSession
from yunshu_engine.ane.native.mil import mlp_program


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--channels", default="1088,2176,3264")
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--layer", type=int, default=40)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def validate_channels(text, inter):
    values = [int(v) for v in text.split(",")]
    if not values or any(v <= 0 or v >= inter or v % 64 for v in values):
        raise ValueError("channels must be nonempty proper 64-aligned slices")
    return values


def main():
    a = parser().parse_args()
    if min(a.rows, a.reps, a.inner) < 1:
        raise ValueError("rows/reps/inner must be positive")
    raw = convert.load_mlp_layer(a.model, a.layer)
    inter = raw["gate_proj"]["weight"].shape[0]
    channels = validate_channels(a.channels, inter)
    import mlx.core as mx

    quant = {
        k: {
            "w": mx.array(v["weight"]),
            "s": mx.array(v["scales"]).astype(mx.bfloat16),
            "b": mx.array(v["biases"]).astype(mx.bfloat16),
            "bits": v["bits"],
            "gs": v["group_size"],
        }
        for k, v in raw.items()
    }
    hidden = raw["gate_proj"]["weight"].shape[1] * 32 // raw["gate_proj"]["bits"]
    x = mx.array(
        np.random.default_rng(0).normal(0, 0.7, (a.rows, hidden)).astype(np.float32)
    ).astype(mx.bfloat16)
    mx.eval(x)
    # ANE gets the exact same numerical input as GPU bf16, represented as fp16.
    xs = np.asarray(x.astype(mx.float16)).copy()

    def qmm(values, proj, n=None, axis=0):
        p = quant[proj]
        w, s, b = p["w"], p["s"], p["b"]
        if n is not None:
            if axis == 0:
                w, s, b = w[:n], s[:n], b[:n]
            else:
                w, s, b = (
                    w[:, : n * p["bits"] // 32],
                    s[:, : n // p["gs"]],
                    b[:, : n // p["gs"]],
                )
        return mx.quantized_matmul(
            values, w, s, b, transpose=True, group_size=p["gs"], bits=p["bits"]
        )

    def gpu(n=None):
        gate = qmm(x, "gate_proj", n)
        up = qmm(x, "up_proj", n)
        h = (gate * mx.sigmoid(gate)) * up
        return qmm(h, "down_proj", n, axis=1)

    for _ in range(25):
        mx.eval(gpu())
    ref = np.array(gpu().astype(mx.float32))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as out, ThreadPoolExecutor(max_workers=1) as pool:

        def emit(record):
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps(record), flush=True)

        emit(
            {
                "kind": "meta",
                "device": "m5",
                "rows": a.rows,
                "mode": a.mode,
                "layer": a.layer,
                "resident_input": True,
                "handoff_included": False,
                "join_included": False,
                "serving_engaged": False,
                "cached": False,
            }
        )
        for count in channels:
            n_gpu = inter - count
            weights = convert.mlp_slices(raw, n_gpu, a.mode)
            mil, blob = mlp_program(*weights, a.rows, a.mode, 6)
            session, reason = NativeSession.try_create(
                mil, blob, [(hidden, a.rows)], (hidden, a.rows), a.cache
            )
            if session is None:
                raise RuntimeError(reason)
            with session:
                session.copy_input(xs.T)
                if session.run() is None:
                    raise RuntimeError(session.reason)
                part = np.array(gpu(n_gpu).astype(mx.float32))
                joined = part + session.copy_output().T.astype(np.float32)
                rel = float(np.linalg.norm(joined - ref) / np.linalg.norm(ref))
                if not np.isfinite(joined).all() or rel > 0.10:
                    raise ValueError(f"invalid overlap slice output relL2={rel}")
                for _ in range(25):
                    mx.eval(gpu(n_gpu))
                for arm in ["base", "cand", "cand-r1", "base-r1", "cand-r2", "base-r2"]:
                    times = []
                    ane = []
                    gpu_times = []
                    for _ in range(a.reps):
                        start = time.perf_counter()
                        for _ in range(a.inner):
                            if arm.startswith("cand"):
                                future = pool.submit(session.run)
                                gs = time.perf_counter()
                                mx.eval(gpu(n_gpu))
                                gpu_times.append((time.perf_counter() - gs) * 1000)
                                ms = future.result()
                                if ms is None:
                                    raise RuntimeError(session.reason)
                                ane.append(ms)
                            else:
                                mx.eval(gpu())
                        times.append((time.perf_counter() - start) * 1000 / a.inner)
                    emit(
                        {
                            "kind": "overlap",
                            "arm": arm,
                            "rows": a.rows,
                            "mode": a.mode,
                            "channels": count,
                            "share": count / inter,
                            "wall_ms": times,
                            "median_ms": float(np.median(times)),
                            "gpu_part_ms": gpu_times,
                            "ane_ms": ane,
                            "rel_l2_vs_gpu": rel,
                            "cached": False,
                        }
                    )
        emit({"complete": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
