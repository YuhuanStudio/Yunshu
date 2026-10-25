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


def load_layers(model: str, first: int, count: int, tiled: bool = False):
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
            if tiled:
                from yunshu_engine.kernels.tensorfold import lane_qmm

                tw = lane_qmm.tile_weight(w, 32, gs, bits=bits)
                sbt = lane_qmm.pack_scales(s, b)
                mx.eval(tw, sbt)
                return QProj(
                    tw, None, None, bits, gs, rows, packed * 32 // bits, True, sbt
                )
            return QProj(w, s, b, bits, gs, rows, packed * 32 // bits)

        out.append(Layer(q("gate_proj"), q("up_proj"), q("down_proj")))
    return out


def _use_stock(split):
    """GPU part through mx.quantized_matmul on untiled leading slices (memory-heavy, a probe)."""
    import mlx.core as mx

    from yunshu_engine.ane.ffn_split import _swiglu
    from yunshu_engine.kernels.tensorfold import lane_qmm

    gpu = split.shape.gpu
    stock = []
    for lay in split.layers:
        entry = {}
        for name, proj, lead_in in (
            ("gate", lay.gate, False),
            ("up", lay.up, False),
            ("down", lay.down, True),
        ):
            w = lane_qmm.untile_weight(proj.w, 32, proj.gs, bits=proj.bits)
            sc = mx.contiguous(proj.sbt[..., 0].T)
            bi = mx.contiguous(proj.sbt[..., 1].T)
            if lead_in:
                w = mx.contiguous(w[:, : gpu * proj.bits // 32])
                sc, bi = mx.contiguous(sc[:, : gpu // proj.gs]), mx.contiguous(bi[:, : gpu // proj.gs])
            else:
                w, sc, bi = w[:gpu], sc[:gpu], bi[:gpu]
            mx.eval(w, sc, bi)
            entry[name] = (w, sc, bi, proj.bits, proj.gs)
        stock.append(entry)

    def qmm(x, t):
        return mx.quantized_matmul(
            x, t[0], t[1], t[2], transpose=True, group_size=t[4], bits=t[3]
        )

    def gpu_part(layer, normalized, residual):
        e = stock[layer]
        y = qmm(_swiglu(qmm(normalized, e["gate"]), qmm(normalized, e["up"])), e["down"])
        return y if residual is None else residual + y

    split._gpu_part = gpu_part


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--first", type=int, default=40)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--rows", default="2048")
    p.add_argument("--units", default="8")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--tile", default="", help="bm,bn,wm,wn for the NAX GPU part")
    p.add_argument(
        "--gpu-kernel", choices=["nax", "stock"], default="nax",
        help="tiled layers only: GPU part through the NAX tiled kernel or MLX quantized_matmul on untiled copies",
    )
    a = p.parse_args()
    import mlx.core as mx

    if a.tile:
        from yunshu_engine.ane import ffn_split

        bm, bn, wm, wn = (int(v) for v in a.tile.split(','))
        ffn_split.NAX_TILE.update(bm=bm, bn=bn, wm=wm, wn=wn)
    layers = load_layers(a.model, a.first, a.layers, a.tiled)
    if a.tiled:
        from yunshu_engine.kernels import nax_prefill

        if not nax_prefill.enable():
            raise SystemExit('NAX prefill unavailable')
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
            if a.tiled and a.gpu_kernel == 'stock':
                _use_stock(split)
            emit(
                {
                    "kind": "setup",
                    "units": units,
                    "share": split.share,
                    "seconds": time.perf_counter() - t0,
                }
            )
            if not a.no_verify:
                err = split.verify()
                emit({"kind": "verify", "units": units, "error": err})
            for rows in rows_list:
                rng = np.random.default_rng(1)
                x = mx.array(
                    rng.normal(0, 1.0, (rows, split.shape.hidden)).astype(np.float32)
                ).astype(mx.bfloat16)
                zero = mx.zeros((rows, split.shape.hidden), dtype=mx.bfloat16)
                mx.eval(x, zero)

                cpu = []

                def alone():
                    h = zero
                    for layer in range(len(layers)):
                        h = split.gpu_alone(layer, x, h)
                    mx.eval(h)

                def split_run():
                    h = zero
                    if not split.begin():
                        raise RuntimeError(split.reason)
                    t_cpu = time.perf_counter()
                    for layer in range(len(layers)):
                        h = split.add(
                            layer, AneFfn.after(x, h) if layer else x, h, rows
                        )
                    cpu.append((time.perf_counter() - t_cpu) * 1000 / len(layers))
                    mx.eval(h)
                    if not split.finish():
                        raise RuntimeError(split.reason)

                def gpu_part():
                    h = zero
                    split.begin()
                    for layer in range(len(layers)):
                        h = split.add(
                            layer,
                            AneFfn.after(x, h) if layer else x,
                            h,
                            rows,
                            ane=False,
                        )
                    mx.eval(h)

                for _ in range(3):
                    alone()
                    split_run()
                    gpu_part()
                split.profile(True)
                ta, ts, tg = [], [], []
                cpu.clear()
                for _ in range(a.reps):
                    t = time.perf_counter()
                    alone()
                    ta.append((time.perf_counter() - t) * 1000 / len(layers))
                    t = time.perf_counter()
                    split_run()
                    ts.append((time.perf_counter() - t) * 1000 / len(layers))
                    t = time.perf_counter()
                    gpu_part()
                    tg.append((time.perf_counter() - t) * 1000 / len(layers))
                emit(
                    {
                        "kind": "timing",
                        "units": units,
                        "share": split.share,
                        "rows": rows,
                        "alone_ms_per_layer": ta,
                        "split_ms_per_layer": ts,
                        "gpu_part_ms_per_layer": tg,
                        "gpu_part_median": statistics.median(tg),
                        "cpu_encode_ms_per_layer": statistics.median(cpu),
                        "timeline_last": split.timeline[-2 * len(layers) : -len(layers)],
                        "ticket_times_last": split.ticket_times[-1],
                        "phase_ms_per_layer": {
                            k: v * 1000 / (a.reps * 2 * len(layers))
                            for k, v in split.prof.items()
                            if k != "t0"
                        },
                        "alone_median": statistics.median(ta),
                        "split_median": statistics.median(ts),
                        "ane_ms_per_eval": split.served_ms
                        / max(split.served_evaluations, 1),
                    }
                )
            split.close()
        emit({"complete": True})


if __name__ == "__main__":
    main()
