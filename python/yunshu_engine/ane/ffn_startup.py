# Derived from Splash 1.4.0 runtime/ops/AneFfnMeasurement.cpp and runtime/engine/AneFfnStartup.cpp (Apache-2.0).
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
# Modified for Yunshu: Python port; calibration kept as a file beside the compiled programs.
"""Calibrate and start the ANE FFN split: time the layers, choose the share and the least chunk.

Calibration times five sampled layers of full chunks at two shares (0.4, 0.8) and the GPU alone,
fits T(s) = max(A(s), G(s) + u*A(s)), picks the ANE's units, then times every program function
against the GPU alone to find the least chunk worth splitting. One calibration per device and
model is remembered; a split that later loses to the GPU alone forgets it.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yunshu_engine.ane import ffn_calibration as cal
from yunshu_engine.ane import ffn_program as P
from yunshu_engine.ane.ffn_split import AneFfn, AneRetired, AneUnavailable, Layer, planned_bytes, units

log = logging.getLogger(__name__)

TIMED_SHARES = (0.4, 0.8)
SAMPLED_LAYERS = 5
COMMAND_ROWS = 2 * P.MAXIMUM_ROWS
RUNS = 3


def _median_ms(run: Callable[[], float]) -> float:
    return statistics.median(run() for _ in range(RUNS))


class Measurement:
    def __init__(self, layers: Sequence[Layer]):
        import mlx.core as mx

        if len(layers) < 2:
            raise ValueError("ANE FFN split timed on fewer than two layers")
        self._mx = mx
        self.layers = list(layers)
        self.hidden = layers[0].gate.inp
        self._inputs: dict[int, tuple] = {}

    def _data(self, rows: int):
        mx = self._mx
        if rows not in self._inputs:
            rng = np.random.default_rng(20261001)
            x = mx.array(rng.uniform(-2.0, 2.0, (rows, self.hidden)).astype(np.float32)).astype(mx.bfloat16)
            zero = mx.zeros((rows, self.hidden), dtype=mx.bfloat16)
            mx.eval(x, zero)
            self._inputs[rows] = (x, zero)
        return self._inputs[rows]

    def layer(self, layers_idx: Sequence[int], count: int, rows: int, split: AneFfn | None, stand_in: float = 0.0):
        """Median milliseconds of a layer in commands of `count` layers; `split` runs the split.
        Each split layer follows the GPU alone's FFN of the same layer, which stands in for the
        mixer before it in a prefill; `stand_in` (the GPU alone timed alike) is subtracted."""
        mx = self._mx
        x, zero = self._data(rows)
        idx = list(layers_idx)
        if not count or count >= len(idx):
            raise ValueError("ANE FFN layers timed without one to stage")

        def command() -> float:
            if split is not None and not split.begin():
                raise AneRetired(split.reason)
            h0, h1 = zero, zero
            start = time.perf_counter()
            for i in range(count):
                if split is not None:
                    h1 = _gpu_alone(split, self.layers[idx[i]], x, h0)
                    h0 = split.add(i, AneFfn.after(x, h0) if i else x, h1, rows)
                else:
                    h0 = _gpu_alone(None, self.layers[idx[i]], x, h0)
            mx.eval(h0)
            if split is not None and not split.finish():
                raise AneRetired(split.reason)
            return (time.perf_counter() - start) * 1000 / count

        return cal.usable(_median_ms(command) - stand_in)

    def gpu_part(self, split: AneFfn) -> float:
        mx = self._mx
        x, zero = self._data(P.MAXIMUM_ROWS)
        n = len(split.layers)

        def cmd(count: int) -> float:
            def run() -> float:
                split.begin()
                h = zero
                start = time.perf_counter()
                for i in range(count):
                    h = split.add(i, AneFfn.after(x, h) if i else x, h, P.MAXIMUM_ROWS, ane=False)
                mx.eval(h)
                return (time.perf_counter() - start) * 1000

            return _median_ms(run)

        return cal.usable((cmd(n - 1) - cmd(1)) / (n - 2))

    def time(self, cache: Path) -> cal.Timings:
        n_units = units(self.layers)
        count = SAMPLED_LAYERS - 1
        sampled_idx = [i * (len(self.layers) - 1) // (SAMPLED_LAYERS - 1) for i in range(SAMPLED_LAYERS)]
        sampled = [self.layers[i] for i in sampled_idx]
        sub = Measurement(sampled)
        sub._inputs = self._inputs
        rows = P.MAXIMUM_ROWS
        gpu_alone = sub.layer(range(len(sampled)), count, rows, None)

        def measure(share: float) -> cal.Timing:
            ane_units = cal.nearest_units(share, n_units)
            split = AneFfn(sampled, ane_units, cache, function_rows=[rows])
            try:
                return cal.Timing(
                    ane_units / n_units,
                    sub.gpu_part(split),
                    sub.layer(range(len(sampled)), count, rows, split, gpu_alone),
                )
            finally:
                split.close()

        low = measure(TIMED_SHARES[0])
        high = measure(TIMED_SHARES[1])
        return cal.Timings(low, high, gpu_alone)

    def chunks(self, split: AneFfn) -> cal.ChunkTimings:
        rows_list = split.rows

        def count(chunk: int) -> int:
            return int(min(max(COMMAND_ROWS // chunk, 1), len(self.layers) - 1))

        def gpu(chunk: int) -> cal.GpuPoint:
            return cal.GpuPoint(chunk, self.layer(range(len(self.layers)), count(chunk), chunk, None))

        low, high = gpu(rows_list[0]), gpu(rows_list[-1])
        timings = cal.ChunkTimings((low, high), [])
        slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows)
        for f in range(len(rows_list) - 1, -1, -1):
            rows = rows_list[f]
            least = rows_list[f - 1] + 1 if f else rows_list[0]
            stand_in = low.milliseconds + slope * (rows - low.rows)
            ms = self.layer(range(len(self.layers)), count(rows), rows, split, stand_in)
            timings.functions.append(cal.FunctionTiming(least, rows, ms))
        return timings


def _gpu_alone(split: AneFfn | None, layer: Layer, x, residual):
    import mlx.core as mx

    from yunshu_engine.ane.ffn_split import _qmm

    gate, up = _qmm(x, layer.gate), _qmm(x, layer.up)
    y = _qmm((gate * mx.sigmoid(gate)) * up, layer.down)
    return residual + y


# ---- remembered calibration -------------------------------------------------------------
def calibration_key(model_id: str, layers: Sequence[Layer], max_units: int) -> str:
    h = hashlib.sha256(
        f"{model_id}|{platform.mac_ver()[0]}|{platform.machine()}|{len(layers)}|"
        f"{layers[0].gate.out}x{layers[0].gate.inp}|{max_units}".encode()
    ).hexdigest()[:16]
    return f"ane-ffn-{h}"


def recall(cache: Path, key: str) -> cal.Calibration | None:
    try:
        return cal.Calibration.parse((cache / key).read_text())
    except OSError:
        return None


def remember(cache: Path, key: str, calibration: cal.Calibration) -> None:
    try:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / key).write_text(calibration.text() + "\n")
    except OSError:
        pass


def forget(cache: Path, key: str) -> None:
    try:
        (cache / key).unlink()
    except OSError:
        pass


@dataclass
class Outcome:
    kind: str  # split | no_gain | unavailable | off
    reason: str
    split: AneFfn | None = None


def start(
    layers: Sequence[Layer],
    cache: Path,
    model_id: str,
    *,
    given_share: float | None = None,
    given_minimum_rows: int | None = None,
    budget_bytes: int | None = None,
) -> Outcome:
    """Decide and build the split (Splash decide()): remembered calibration, else calibrate."""
    try:
        n_units = units(layers)
    except ValueError as exc:
        return Outcome("unsupported", str(exc))
    max_units = n_units - 1
    if budget_bytes is not None:
        while max_units > 1 and planned_bytes(layers, max_units) > budget_bytes:
            max_units -= 1
    key = calibration_key(model_id, layers, max_units)
    started = time.perf_counter()
    try:
        if given_share is not None:
            ane_units = cal.nearest_units(given_share, n_units)
            split = AneFfn(layers, ane_units, cache)
            minimum = given_minimum_rows or P.MINIMUM_ROWS
            split.set_minimum_rows(minimum)
            err = split.verify()
            return Outcome(
                "split",
                f"at the given share {ane_units / n_units:.2f} for chunks of {minimum} rows or more, "
                f"{100 * err:.2f}% from the GPU alone on the Neural Engine's part",
                split,
            )
        calibration = recall(cache, key)
        if calibration and calibration.ane_units and calibration.ane_units > max_units:
            calibration = None
        if calibration:
            if not calibration.ane_units:
                return Outcome("no_gain", "no Neural Engine split beat it by enough when this Mac calibrated it")
            split = AneFfn(layers, calibration.ane_units, cache)
            _take(split, calibration, cache, key)
            err = split.verify()
            return Outcome(
                "split",
                f"at share {calibration.ane_units / n_units:.2f} for chunks of {calibration.minimum_rows} "
                f"rows or more, {100 * err:.2f}% from the GPU alone (set up as calibrated before)",
                split,
            )
        measurement = Measurement(layers)
        timings = measurement.time(cache)
        chosen = cal.choose(cal.fit(timings.low, timings.high), timings.gpu_alone, n_units, max_units)
        if chosen is None:
            remember(cache, key, cal.Calibration())
            return Outcome("no_gain", f"no Neural Engine split beats its {timings.gpu_alone:.1f} ms per 2048-row FFN layer by enough")
        split = AneFfn(layers, chosen, cache)
        chunks = measurement.chunks(split)
        minimum = cal.minimum_rows(chunks)
        if minimum is None:
            split.close()
            remember(cache, key, cal.Calibration())
            return Outcome("no_gain", "no chunk of the Neural Engine split beats the GPU alone by enough")
        made = cal.Calibration(chosen, minimum, chunks.gpu)
        _take(split, made, cache, key)
        err = split.verify()
        remember(cache, key, made)
        return Outcome(
            "split",
            f"at share {chosen / n_units:.2f} for chunks of {minimum} rows or more: "
            f"{chunks.functions[0].split:.1f} ms per 2048-row FFN layer against {chunks.gpu[1].milliseconds:.1f} "
            f"on the GPU alone, {100 * err:.2f}% from the GPU alone (calibrated in "
            f"{time.perf_counter() - started:.1f} s)",
            split,
        )
    except (AneUnavailable, AneRetired, RuntimeError, ValueError, OSError) as exc:
        return Outcome("unavailable", str(exc))


def _take(split: AneFfn, calibration: cal.Calibration, cache: Path, key: str) -> None:
    split.set_minimum_rows(calibration.minimum_rows)
    split.breaker = cal.Breaker(calibration.gpu)
    split.lost = lambda: forget(cache, key)
