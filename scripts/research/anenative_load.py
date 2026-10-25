"""Sustained ANE load beside tfbench GPU decode; run under the same gpuq lock."""

from __future__ import annotations

import argparse
import ctypes as c
import gc
import hashlib
import json
import signal
import sys
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
    p.add_argument("--layers", default="40,41,42,43")
    p.add_argument("--duration", type=float, default=0.0, help="0 runs until SIGTERM")
    p.add_argument("--start-paused", action="store_true")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["native", "coreml"], required=True)
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--channels", type=int, default=2176)
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    return p


def release_staging():
    """Free unreachable converter graphs/allocator staging, never live model storage."""
    gc.collect()
    if sys.platform != "darwin":
        return 0
    try:
        trim = c.CDLL(None).malloc_zone_pressure_relief
        trim.argtypes, trim.restype = [c.c_void_p, c.c_size_t], c.c_size_t
        return int(trim(None, 0))
    except (AttributeError, OSError):
        return 0


def main():
    a = parser().parse_args()
    if a.rows < 1 or a.channels < 1:
        raise ValueError("rows/channels must be positive")
    generator_sha = hashlib.sha256(
        Path(mlp_program.__code__.co_filename).read_bytes()
    ).hexdigest()
    a.ready.unlink(missing_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    layers = [int(value) for value in a.layers.split(",")]
    sessions, runners, whole_mlps = [], [], []
    rng = np.random.default_rng(0)

    def prepare(layer_index):
        layer = convert.load_mlp_layer(a.model, layer_index)
        inter = layer["gate_proj"]["weight"].shape[0]
        whole_mlps.append(a.channels == inter)
        if a.channels > inter:
            raise ValueError("slice too wide")
        weights = convert.mlp_slices(layer, inter - a.channels, a.mode)
        hidden = weights[0].shape[1]
        x = rng.normal(0, 0.7, (a.rows, hidden)).astype(np.float16)
        if a.backend == "native":
            mil, blob = mlp_program(*weights, a.rows, a.mode, 6)
            session, reason = NativeSession.try_create(
                mil, blob, [(hidden, a.rows)], (hidden, a.rows), a.cache / "native"
            )
            if session is None:
                raise RuntimeError(reason)
            sessions.append(session)
            session.copy_input(x.T)

            def call():
                if session.run() is None:
                    raise RuntimeError(session.reason)

            call()
            if not np.isfinite(session.copy_output()).all():
                raise ValueError("nonfinite native background output")
            return call
        import coremltools as ct

        path = a.cache / f"l{layer_index}-c{a.channels}-r{a.rows}-{a.mode}.mlpackage"
        if not path.exists():
            convert.build_mlp_model(*weights, a.rows, a.mode, str(path), 6)
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        inp = model.get_spec().description.input[0].name
        output = model.get_spec().description.output[0].name

        def call():
            return model.predict({inp: x[None]})[output]

        if not np.isfinite(call()).all():
            raise ValueError("nonfinite Core ML background output")
        return call

    try:
        for layer_index in layers:
            runners.append(prepare(layer_index))
    except BaseException:
        for session in sessions:
            session.close()
        raise
    index = 0

    def run():
        nonlocal index
        runners[index % len(runners)]()
        index += 1

    stop = False
    active = not a.start_paused
    count = 0
    samples, changes = [], []
    sample_start, sample_count = None, 0

    def flush_sample(now):
        nonlocal sample_start, sample_count
        if sample_start is not None and now > sample_start:
            samples.append(
                {
                    "start_s": sample_start,
                    "end_s": now,
                    "calls": count - sample_count,
                    "active": active,
                }
            )
        sample_start, sample_count = now, count

    def stopping(*_):
        nonlocal stop
        stop = True

    def toggle(signum, *_):
        nonlocal active
        now = time.perf_counter()
        flush_sample(now)
        active = signum == signal.SIGUSR1
        changes.append({"time_s": now, "active": active})

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    signal.signal(signal.SIGUSR1, toggle)
    signal.signal(signal.SIGUSR2, toggle)
    start = time.perf_counter()
    try:
        run()
        relief_bytes = release_staging()
        a.ready.write_text(
            json.dumps(
                {
                    "ready": True,
                    "backend": a.backend,
                    "generator_sha256": generator_sha,
                    "rows": a.rows,
                    "channels": a.channels,
                    "layers": layers,
                    "whole_mlp": all(whole_mlps),
                    "active": active,
                    "allocator_relief_bytes": relief_bytes,
                }
            )
        )
        start = time.perf_counter()
        sample_start, sample_count = start, 0
        while not stop and (not a.duration or time.perf_counter() - start < a.duration):
            if active:
                run()
                count += 1
            else:
                time.sleep(0.02)
            now = time.perf_counter()
            if now - sample_start >= 1.0:
                flush_sample(now)
        end = time.perf_counter()
        flush_sample(end)
        seconds = end - start
        active_seconds = sum(r["end_s"] - r["start_s"] for r in samples if r["active"])
        active_rate = count * a.rows / active_seconds if active_seconds else 0.0
        a.out.write_text(
            json.dumps(
                {
                    "complete": True,
                    "backend": a.backend,
                    "generator_sha256": generator_sha,
                    "calls": count,
                    "seconds": seconds,
                    "rows": a.rows,
                    "channels": a.channels,
                    "rows_per_second": count * a.rows / seconds,
                    "active_rows_per_second": active_rate,
                    "active_seconds": active_seconds,
                    "activity_changes": changes,
                    "allocator_relief_bytes": relief_bytes,
                    "layers": layers,
                    "whole_mlp": all(whole_mlps),
                    "mlp_only_model64_tps_upper_bound": active_rate / 64
                    if all(whole_mlps)
                    else None,
                    "rate_samples": samples,
                }
            )
            + "\n"
        )
    finally:
        for session in sessions:
            session.close()
        a.ready.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
