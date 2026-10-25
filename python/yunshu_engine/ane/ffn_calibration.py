# Upstream (Splash 1.4.0, Apache-2.0): runtime/ops/AneFfnCalibration.{hpp,cpp} (Apache-2.0).
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
# Modified for Yunshu: Python port, same model, constants and decisions.
"""How the ANE FFN split runs on this device, from measured timings (no MLX, no ANE).

A full chunk's split FFN layer at share s takes T(s) = max(A(s), G(s) + u*A(s)):
the ANE's part A(s) (which the layer waits on where it is longer) or the GPU's part
G(s) slowed by the memory bandwidth the ANE's part takes beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# The split's predicted gain below which the GPU runs the FFN alone.
MINIMUM_GAIN = 0.05
# How much longer than the GPU alone a chunk's split layer may take and still split.
CHUNK_NOISE = 0.02
CALIBRATION_TAG = "ane-ffn-calibration-1"
WINDOW = 8


@dataclass(frozen=True)
class Timing:
    share: float
    gpu: float
    split: float


@dataclass(frozen=True)
class Timings:
    low: Timing
    high: Timing
    gpu_alone: float


def usable(milliseconds: float) -> float:
    if not (milliseconds > 0.0 and math.isfinite(milliseconds)):
        raise RuntimeError("calibration timings are not usable")
    return milliseconds


def nearest_units(share: float, units: int) -> int:
    if not (0.0 < share < 1.0) or units < 2:
        raise ValueError("ANE FFN share must lie in (0, 1) of two units or more")
    return min(max(int(math.floor(share * units + 0.5)), 1), units - 1)


@dataclass(frozen=True)
class Line:
    at0: float
    slope: float

    def __call__(self, x: float) -> float:
        return self.at0 + self.slope * x


@dataclass(frozen=True)
class Model:
    gpu: Line
    ane: Line
    u: float = 0.0

    def __call__(self, share: float) -> float:
        a = self.ane(share)
        g = self.gpu(share) + self.u * a
        return a if a > g else g


def fit(low: Timing, high: Timing) -> Model:
    if not low.share < high.share:
        raise ValueError("ANE FFN calibration needs a low and a high share")
    slope = (high.gpu - low.gpu) / (high.share - low.share)
    gpu_line = Line(low.gpu - slope * low.share, slope)
    ane_line = Line(0.0, high.split / high.share)
    gpu, ane = gpu_line(low.share), ane_line(low.share)
    u = 0.0
    if gpu > ane:
        u = min(max((low.split - gpu) / ane, 0.0), 1.0)
    return Model(gpu_line, ane_line, u)


def choose(
    model: Model, gpu_alone: float, units: int, max_ane_units: int
) -> int | None:
    if not max_ane_units or max_ane_units >= units:
        raise ValueError("ANE FFN choice of no feasible units")
    at = [0.0] * units
    for ane_units in range(1, units):
        at[ane_units] = model(ane_units / units)
        if not (at[ane_units] > 0.0 and math.isfinite(at[ane_units])):
            return None

    def longest(n: int) -> float:
        return max(at[n], at[max(n - 1, 1)], at[min(n + 1, units - 1)])

    chosen = 1
    for n in range(2, max_ane_units + 1):
        if longest(n) < longest(chosen):
            chosen = n
    if not at[chosen] <= (1.0 - MINIMUM_GAIN) * gpu_alone:
        return None
    return chosen


@dataclass(frozen=True)
class GpuPoint:
    rows: int
    milliseconds: float


@dataclass(frozen=True)
class FunctionTiming:
    least: int
    rows: int
    split: float


@dataclass
class ChunkTimings:
    gpu: tuple[GpuPoint, GpuPoint]
    functions: list[FunctionTiming] = field(default_factory=list)


def gpu_alone_line(gpu) -> Line:
    low, high = gpu
    slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows)
    return Line(low.milliseconds - slope * low.rows, slope)


def minimum_rows(timings: ChunkTimings) -> int | None:
    low, high = timings.gpu
    ordered = bool(timings.functions) and low.rows < high.rows
    for i, f in enumerate(timings.functions):
        if not ordered:
            break
        ordered = (
            f.least > 0
            and f.least <= f.rows
            and (i == 0 or f.rows + 1 == timings.functions[i - 1].least)
        )
    if not ordered:
        raise ValueError("ANE FFN chunks are not those of its functions")
    gpu = gpu_alone_line(timings.gpu)
    minimum = None
    for f in timings.functions:
        for rows in range(f.rows, f.least - 1, -1):
            if not f.split <= (1.0 + CHUNK_NOISE) * gpu(rows):
                return minimum
            minimum = rows
    return minimum


@dataclass(frozen=True)
class Calibration:
    ane_units: int = 0
    minimum_rows: int = 0
    gpu: tuple[GpuPoint, GpuPoint] | None = None

    def text(self) -> str:
        out = f"{CALIBRATION_TAG} {self.ane_units} {self.minimum_rows}"
        for p in self.gpu or (GpuPoint(0, 0.0), GpuPoint(0, 0.0)):
            out += f" {p.rows} {p.milliseconds:.9g}"
        return out

    @staticmethod
    def parse(text: str) -> Calibration | None:
        parts = text.split()
        try:
            if len(parts) != 7 or parts[0] != CALIBRATION_TAG:
                return None
            units, minimum = int(parts[1]), int(parts[2])
            pts = (
                GpuPoint(int(parts[3]), float(parts[4])),
                GpuPoint(int(parts[5]), float(parts[6])),
            )
        except ValueError:
            return None

        def ok(v: float) -> bool:
            return v > 0.0 and math.isfinite(v)

        if not units:
            return Calibration() if not minimum else None
        if not minimum or not pts[0].rows < pts[1].rows:
            return None
        if not (ok(pts[0].milliseconds) and ok(pts[1].milliseconds)):
            return None
        return Calibration(units, minimum, pts)


class Breaker:
    """Stops a split that serves but loses to the GPU alone (per WINDOW commands)."""

    def __init__(self, gpu=None):
        self._gpu: Line | None = None
        self._most = 0
        self._commands = 0
        self._ane = 0.0
        self._alone = 0.0
        if gpu is not None:
            low, high = gpu
            if not low.rows < high.rows:
                raise ValueError(
                    "ANE FFN breaker needs the GPU alone at two row counts"
                )
            usable(low.milliseconds)
            usable(high.milliseconds)
            self._gpu = gpu_alone_line(gpu)
            self._most = high.rows

    def add(self, rows: int, evaluations: int, milliseconds: float) -> str:
        if self._gpu is None:
            return ""
        self._ane += milliseconds
        self._alone += evaluations * self._gpu(rows)
        self._commands += 1
        if self._commands < WINDOW:
            return ""
        ratio = self._ane / self._alone
        most = self._gpu(self._most)
        self._commands, self._ane, self._alone = 0, 0.0, 0.0
        if not ratio >= 1.0:
            return ""
        return (
            f"losing to the GPU alone ({ratio * most:.1f} ms on the Neural Engine per "
            f"{self._most}-row layer against {most:.1f} ms on the GPU alone)"
        )
