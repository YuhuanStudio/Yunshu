"""Sustained ANE load beside tfbench GPU decode; run under the same gpuq lock."""

from __future__ import annotations

import argparse
import json
import signal
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
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["native", "coreml"], required=True)
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--channels", type=int, default=2176)
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    return p


def main():
    a = parser().parse_args()
    if a.rows < 1 or a.channels < 1:
        raise ValueError("rows/channels must be positive")
    a.ready.unlink(missing_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    layer = convert.load_mlp_layer(a.model, 40)
    inter = layer["gate_proj"]["weight"].shape[0]
    if a.channels > inter:
        raise ValueError("slice too wide")
    weights = convert.mlp_slices(layer, inter - a.channels, a.mode)
    hidden = weights[0].shape[1]
    x = np.random.default_rng(0).normal(0, 0.7, (a.rows, hidden)).astype(np.float16)
    session = None
    if a.backend == "native":
        mil, blob = mlp_program(*weights, a.rows, a.mode, 6)
        session, reason = NativeSession.try_create(
            mil, blob, [(hidden, a.rows)], (hidden, a.rows), a.cache / "native"
        )
        if session is None:
            raise RuntimeError(reason)
        session.copy_input(x.T)

        def run():
            if session.run() is None:
                raise RuntimeError(session.reason)
    else:
        import coremltools as ct

        path = a.cache / f"c{a.channels}-r{a.rows}-{a.mode}.mlpackage"
        if not path.exists():
            convert.build_mlp_model(*weights, a.rows, a.mode, str(path), 6)
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        inp = model.get_spec().description.input[0].name

        def run():
            model.predict({inp: x[None]})

    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    count = 0
    start = time.perf_counter()
    try:
        run()
        a.ready.write_text(
            json.dumps(
                {
                    "ready": True,
                    "backend": a.backend,
                    "rows": a.rows,
                    "channels": a.channels,
                }
            )
        )
        start = time.perf_counter()
        while not stop:
            run()
            count += 1
        seconds = time.perf_counter() - start
        a.out.write_text(
            json.dumps(
                {
                    "complete": True,
                    "backend": a.backend,
                    "calls": count,
                    "seconds": seconds,
                    "rows": a.rows,
                    "channels": a.channels,
                    "rows_per_second": count * a.rows / seconds,
                }
            )
            + "\n"
        )
    finally:
        if session is not None:
            session.close()
        a.ready.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
