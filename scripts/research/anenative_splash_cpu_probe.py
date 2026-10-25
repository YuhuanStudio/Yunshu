"""CPU cost of one staging/pack launch with IOSurface aliases vs ordinary arrays (async_eval time)."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from anenative_splash_micro import load_layers

from yunshu_engine.ane.ffn_split import AneFfn


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    import mlx.core as mx

    layers = load_layers(a.model, 40, 3, True)
    split = AneFfn(layers, 8, a.cache, function_rows=[2048])

    def measure(label):
        build, ev, total = [], [], []
        for _ in range(8):
            t0 = time.perf_counter()
            m = split._stage(1, 1)
            t1 = time.perf_counter()
            mx.async_eval(*m)
            t2 = time.perf_counter()
            mx.eval(*m)
            t3 = time.perf_counter()
            build.append((t1 - t0) * 1000)
            ev.append((t2 - t1) * 1000)
            total.append((t3 - t0) * 1000)
        print(
            f"{label}: build {statistics.median(build):.2f} ms, async_eval {statistics.median(ev):.2f} ms "
            f"(6 launches), total incl GPU {statistics.median(total):.2f} ms",
            flush=True,
        )

    measure("aliases")
    real = split.alias
    cache = {}

    def plain(surf):
        if surf.index not in cache:
            dtype = mx.int8 if surf.element == "int8" else mx.float16
            cache[surf.index] = mx.zeros((surf.rows, surf.width), dtype=dtype)
            mx.eval(cache[surf.index])
        return cache[surf.index]

    split.alias = plain
    measure("plain arrays")
    split.alias = real
    # a trivial custom kernel for the per-launch floor
    k = mx.fast.metal_kernel(
        name="probe_noop",
        input_names=["x"],
        output_names=["y"],
        source="y[thread_position_in_grid.x] = x[thread_position_in_grid.x];",
    )
    x = mx.zeros((1024,), dtype=mx.float32)
    mx.eval(x)
    t = []
    for _ in range(8):
        t0 = time.perf_counter()
        outs = [
            k(
                inputs=[x],
                grid=(1024, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1024,)],
                output_dtypes=[mx.float32],
            )[0]
            for _ in range(10)
        ]
        mx.async_eval(*outs)
        t.append((time.perf_counter() - t0) * 1000 / 10)
        mx.eval(*outs)
    print(f"noop custom kernel: {statistics.median(t):.3f} ms per launch (CPU)", flush=True)
    print("complete")


if __name__ == "__main__":
    main()
