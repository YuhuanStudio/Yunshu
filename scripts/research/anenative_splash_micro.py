"""Splash-style FFN split on real 27B layers: error vs the GPU alone and per-layer timing.

Run under gpuq. Chain of N consecutive layers (the split stages layer l+1 during layer l).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from yunshu_engine.ane import convert
from yunshu_engine.ane import ffn_program as P
from yunshu_engine.ane.ffn_split import AneFfn, Layer, QProj


def load_layers(model: str, first: int, count: int):
    import mlx.core as mx

    out = []
    for i in range(first, first + count):
        raw = convert.load_mlp_layer(model, i)

        def q(name):
            v = raw[name]
            w = mx.array(np.ascontiguousarray(v["weight"]))
            s = mx.array(v["scales"]).astype(mx.bfloat16)
            b = mx.array(v["biases"]).astype(mx.bfloat16)
            bits, gs = v["bits"], v["group_size"]
            rows, packed = w.shape
            return QProj(w, s, b, bits, gs, rows, packed * 32 // bits)

        out.append(Layer(q("gate_proj"), q("up_proj"), q("down_proj")))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    p.add_argument("--first", type=int, default=40)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--rows", default="2048")
    p.add_argument("--units", default="8")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    import mlx.core as mx

    layers = load_layers(a.model, a.first, a.layers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as out:

        def emit(rec):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps(rec), flush=True)

        rows_list = [int(v) for v in a.rows.split(",")]
        for units in (int(v) for v in a.units.split(",")):
            t0 = time.perf_counter()
            split = AneFfn(layers, units, a.cache, function_rows=rows_list)
            emit({"kind": "setup", "units": units, "share": split.share, "seconds": time.perf_counter() - t0})
            if not a.no_verify:
                err = split.verify()
                emit({"kind": "verify", "units": units, "error": err})
            for rows in rows_list:
                rng = np.random.default_rng(1)
                x = mx.array(rng.normal(0, 1.0, (rows, split.shape.hidden)).astype(np.float32)).astype(
                    mx.bfloat16
                )
                zero = mx.zeros((rows, split.shape.hidden), dtype=mx.bfloat16)
                mx.eval(x, zero)

                def alone():
                    h = zero
                    for layer in range(len(layers)):
                        h = split.gpu_alone(layer, x, h)
                    mx.eval(h)

                def split_run():
                    h = zero
                    if not split.begin():
                        raise RuntimeError(split.reason)
                    for layer in range(len(layers)):
                        h = split.add(layer, x, h, rows)
                    mx.eval(h)
                    if not split.finish():
                        raise RuntimeError(split.reason)

                for _ in range(3):
                    alone()
                    split_run()
                ta, ts = [], []
                for _ in range(a.reps):
                    t = time.perf_counter()
                    alone()
                    ta.append((time.perf_counter() - t) * 1000 / len(layers))
                    t = time.perf_counter()
                    split_run()
                    ts.append((time.perf_counter() - t) * 1000 / len(layers))
                emit(
                    {
                        "kind": "timing",
                        "units": units,
                        "share": split.share,
                        "rows": rows,
                        "alone_ms_per_layer": ta,
                        "split_ms_per_layer": ts,
                        "alone_median": statistics.median(ta),
                        "split_median": statistics.median(ts),
                        "ane_ms_per_eval": split.served_ms / max(split.served_evaluations, 1),
                    }
                )
            split.close()
        emit({"complete": True})


if __name__ == "__main__":
    main()
