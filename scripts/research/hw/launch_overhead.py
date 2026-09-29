"""Dispatch / sync overhead of the MLX Metal backend.

* mx.eval round trip on a trivial op (host -> GPU -> host)
* async_eval submit cost vs completion
* per-kernel cost of a long dependent chain of tiny kernels in ONE eval
  (encode + GPU launch gap), under different MLX_MAX_OPS_PER_BUFFER values
* how much mx.compile fuses an elementwise chain (kernel count via graph size + time)

  PYTHONPATH=scripts/research/hw python scripts/research/hw/launch_overhead.py --tag default
  MLX_MAX_OPS_PER_BUFFER=8 ... --tag ops8
"""

import argparse
import os
import time

import mlx.core as mx
from _common import Out, timeit


def graph_nodes(*outs):
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
        mx.export_to_dot(f, *outs)
        f.seek(0)
        txt = f.read()
    return txt.count(" -> ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    out = Out("launch_overhead")
    env = {"max_ops": os.environ.get("MLX_MAX_OPS_PER_BUFFER"), "tag": a.tag}

    x = mx.ones((64,))
    mx.eval(x)
    t, tmin = timeit(lambda: mx.eval(x + 1), 300, 20)
    out(
        kind="eval_roundtrip_tiny",
        us_median=round(t * 1e6, 1),
        us_min=round(tmin * 1e6, 1),
        **env,
    )

    # async_eval: time to return vs time until ready
    subs, waits = [], []
    for _ in range(200):
        t0 = time.perf_counter()
        y = x + 1
        mx.async_eval(y)
        t1 = time.perf_counter()
        mx.eval(y)
        t2 = time.perf_counter()
        subs.append(t1 - t0)
        waits.append(t2 - t1)
    subs.sort()
    waits.sort()
    out(
        kind="async_eval_tiny",
        submit_us=round(subs[100] * 1e6, 1),
        wait_us=round(waits[100] * 1e6, 1),
        **env,
    )

    # dependent chain of N tiny kernels in one eval
    for n in (10, 100, 500, 2000):
        for size in (64, 5120 * 16):
            v = mx.ones((size,))
            mx.eval(v)

            def build(n=n, v=v):
                y = v
                for _ in range(n):
                    y = y + 1.0
                return y

            tb0 = time.perf_counter()
            y = build()
            tb = time.perf_counter() - tb0
            t, _ = timeit(lambda: mx.eval(build()), 10, 2)
            out(
                kind="chain_add",
                n=n,
                elems=size,
                total_ms=round(t * 1e3, 3),
                per_kernel_us=round(t / n * 1e6, 2),
                graph_build_ms=round(tb * 1e3, 3),
                **env,
            )

    # mx.compile fusion on an elementwise chain
    def f(x, y):
        z = x * y + 1.0
        z = mx.sigmoid(z) * z
        z = z * x - y
        return mx.tanh(z) + mx.exp(-z)

    fc = mx.compile(f)
    xx = mx.random.normal((16, 5120))
    yy = mx.random.normal((16, 5120))
    mx.eval(xx, yy)
    t0, _ = timeit(lambda: mx.eval(f(xx, yy)), 200)
    t1, _ = timeit(lambda: mx.eval(fc(xx, yy)), 200)
    reps = 64
    t0b, _ = timeit(lambda: mx.eval([f(xx, yy) for _ in range(reps)]), 10)
    t1b, _ = timeit(lambda: mx.eval([fc(xx, yy) for _ in range(reps)]), 10)
    out(
        kind="compile_elementwise",
        nodes_uncompiled=graph_nodes(f(xx, yy)),
        us_eager=round(t0 * 1e6, 1),
        us_compiled=round(t1 * 1e6, 1),
        us_eager_batched=round(t0b / reps * 1e6, 1),
        us_compiled_batched=round(t1b / reps * 1e6, 1),
        **env,
    )


if __name__ == "__main__":
    main()
