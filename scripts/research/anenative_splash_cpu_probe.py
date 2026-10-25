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

    from yunshu_engine.kernels import nax_prefill

    if not nax_prefill.enable():
        raise SystemExit("NAX prefill unavailable")
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

    rows = 2048
    x = mx.array(
        __import__("numpy").random.default_rng(1).normal(0, 1, (rows, split.shape.hidden)).astype("float32")
    ).astype(mx.bfloat16)
    mx.eval(x)

    def gpu_time(label, fn, reps=20):
        mx.eval(*fn())
        import time as _t

        t = _t.perf_counter()
        for _ in range(reps):
            mx.eval(*fn())
        ms = (_t.perf_counter() - t) * 1000 / reps
        print(f"{label}: {ms:.3f} ms (eval + sync each)", flush=True)

    gpu_time("rotate+pack(2)", lambda: split._pack(x, rows))
    gpu_time("noop-ish (x+0)", lambda: [x + 0])
    out_gpu = x
    gpu_time("join", lambda: list(split._join(out_gpu, rows)))
    gpu_time("stage layer (6 kernels)", lambda: split._stage(1, 1))
    gpu_time("gpu part", lambda: [split._gpu_part(1, x, None)])
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
