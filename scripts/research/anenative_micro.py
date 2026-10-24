"""Paired real-checkpoint native/Core ML ANE slice timing (gpuq only).

Each cell interleaves Core ML/native/native/Core ML; compilation and copies are
separate from native evaluate time. A first failed/nonfinite cell stops the run.
No inference engine or serving arithmetic is changed. Outputs are JSONL receipts.
"""

from __future__ import annotations

import argparse
import json
import time
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
    p.add_argument("--layer", type=int, default=40)
    p.add_argument("--rows", default="128,512,1024,2048")
    p.add_argument("--channels", default="64,1088,2176,4352")
    p.add_argument("--modes", default="fp16,int8")
    p.add_argument("--h-exp", type=int, default=6)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--native-only", action="store_true")
    return p


def validate(y, reference):
    if y.shape != reference.shape or not np.isfinite(y).all():
        raise ValueError("invalid/nonfinite ANE output")
    rel = float(
        np.linalg.norm(y.astype(np.float32) - reference)
        / max(float(np.linalg.norm(reference)), 1e-30)
    )
    if rel > 0.10:
        raise ValueError(f"ANE slice differs from fp32 reference: relative L2={rel}")
    return rel


def economics(gpu_ms, native_floor_ms, per_channel_ms, total_channels=17408):
    """Optimistic linear overlap model; excludes sync, join and shared-memory contention."""
    if gpu_ms <= 0 or native_floor_ms < 0 or per_channel_ms < 0:
        raise ValueError("invalid cost")
    share = max(
        0.0,
        min(
            1.0, (gpu_ms - native_floor_ms) / (gpu_ms + per_channel_ms * total_channels)
        ),
    )
    split = max(
        gpu_ms * (1 - share), native_floor_ms + per_channel_ms * total_channels * share
    )
    if share == 0:
        split = gpu_ms
    return {
        "best_share": share,
        "optimistic_split_ms": split,
        "saving_ms": max(0.0, gpu_ms - split),
    }


def main():
    a = parser().parse_args()
    rows_list = [int(v) for v in a.rows.split(",")]
    channels_list = [int(v) for v in a.channels.split(",")]
    modes = a.modes.split(",")
    if (
        a.reps < 1
        or min(rows_list + channels_list) < 1
        or set(modes) - {"fp16", "int8"}
    ):
        raise ValueError("invalid sweep")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    raw = convert.load_mlp_layer(a.model, a.layer)
    dense = {
        k: convert.dequantize_affine(
            v["weight"], v["scales"], v["biases"], v["bits"], v["group_size"]
        )
        for k, v in raw.items()
    }
    inter, hidden = dense["gate_proj"].shape
    if max(channels_list) > inter:
        raise ValueError("slice too wide")
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

    def qmm(x, proj):
        v = quant[proj]
        return mx.quantized_matmul(
            x,
            v["w"],
            v["s"],
            v["b"],
            transpose=True,
            group_size=v["gs"],
            bits=v["bits"],
        )

    with a.out.open("w") as out:

        def emit(record):
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps(record), flush=True)

        emit(
            {
                "kind": "meta",
                "device": "M5",
                "model": a.model,
                "layer": a.layer,
                "h_exp": a.h_exp,
                "cached": False,
                "serving_engaged": False,
                "native_path": "_ANEClient + IOSurface + shared event completion",
                "coreml_path": "CPU_AND_NE (placement not guaranteed by API)",
            }
        )
        for rows in rows_list:
            rng = np.random.default_rng(0)
            x = rng.normal(0, 0.7, (rows, hidden)).astype(np.float16)
            xm = mx.array(x).astype(mx.bfloat16)

            def gpu():
                g, u = qmm(xm, "gate_proj"), qmm(xm, "up_proj")
                return qmm((g * mx.sigmoid(g)) * u, "down_proj")

            mx.eval(gpu())
            gpu_samples = []
            for _ in range(a.reps):
                start = time.perf_counter()
                mx.eval(gpu())
                gpu_samples.append((time.perf_counter() - start) * 1000)
            emit(
                {
                    "kind": "gpu_ffn",
                    "rows": rows,
                    "ms": gpu_samples,
                    "median_ms": float(np.median(gpu_samples)),
                }
            )
            for channels in channels_list:
                g, u, d = convert.mlp_slice(
                    dense["gate_proj"],
                    dense["up_proj"],
                    dense["down_proj"],
                    inter - channels,
                    inter,
                )
                # Reference is the same dequantized fp16 weights, not bf16 GPU arithmetic.
                gg = x.astype(np.float32) @ g.astype(np.float32).T
                uu = x.astype(np.float32) @ u.astype(np.float32).T
                ref = ((gg / (1 + np.exp(-gg))) * uu) @ d.astype(np.float32).T
                for mode in modes:
                    start = time.perf_counter()
                    mil, blob = mlp_program(g, u, d, rows, mode, a.h_exp)
                    session, reason = NativeSession.try_create(
                        mil, blob, [(hidden, rows)], (hidden, rows), a.cache / "native"
                    )
                    del blob
                    compile_ms = (time.perf_counter() - start) * 1000
                    if session is None:
                        raise RuntimeError(reason)
                    with session:
                        session.copy_input(x.T)
                        if session.run() is None:
                            raise RuntimeError(session.reason)
                        native_rel = validate(session.copy_output().T, ref)
                        model = None
                        if not a.native_only:
                            import coremltools as ct

                            path = (
                                a.cache
                                / f"l{a.layer}-c{channels}-r{rows}-{mode}-h{a.h_exp}.mlpackage"
                            )
                            if not path.exists():
                                convert.build_mlp_model(
                                    g, u, d, rows, mode, str(path), a.h_exp
                                )
                            model = ct.models.MLModel(
                                str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
                            )
                            inp = model.get_spec().description.input[0].name
                            output = model.get_spec().description.output[0].name
                            model.predict({inp: x[None]})
                        for arm in (
                            ["cand", "cand-r1"]
                            if a.native_only
                            else ["base", "cand", "cand-r1", "base-r1"]
                        ):
                            times, copies = [], []
                            for _ in range(a.reps):
                                if arm.startswith("cand"):
                                    start = time.perf_counter()
                                    session.copy_input(x.T)
                                    copies.append((time.perf_counter() - start) * 1000)
                                    ms = session.run()
                                    if ms is None:
                                        raise RuntimeError(session.reason)
                                    times.append(ms)
                                    start = time.perf_counter()
                                    y = session.copy_output().T
                                    copies[-1] += (time.perf_counter() - start) * 1000
                                else:
                                    start = time.perf_counter()
                                    y = model.predict({inp: x[None]})[output][0]
                                    times.append((time.perf_counter() - start) * 1000)
                            rel = validate(y, ref)
                            emit(
                                {
                                    "kind": "slice",
                                    "arm": arm,
                                    "rows": rows,
                                    "channels": channels,
                                    "mode": mode,
                                    "ms": times,
                                    "median_ms": float(np.median(times)),
                                    "copy_ms": copies,
                                    "rel_l2_fp32": rel,
                                    "native_first_rel_l2": native_rel,
                                    "compile_load_ms": compile_ms,
                                    "cached": False,
                                }
                            )
                        del model
        emit({"complete": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
