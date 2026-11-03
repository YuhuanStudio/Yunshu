#!/usr/bin/env python3
"""Measured peaks on this machine: the denominators for MBU / MFU (FULLCMP_SPEC.md, "Derived").

Measures, with MLX, what this Mac actually sustains (not the spec sheet):

  1. memory read bandwidth   bf16 array >= 4 GiB, a reduction that reads all of it once (GB/s);
                             plus a copy (read + write) variant for reference
  2. matmul FLOP/s           bf16 and fp16, 8192^3 square and the 27B prefill shape
                             M=2048 K=5120 N=17408 (TFLOP/s)
  3. quantized GEMV          mx.quantized_matmul, 4-bit, group 64, M=1..8, K=5120, N=17408;
                             GB/s of packed weight bytes actually read (decode MBU is relative
                             to what quantized kernels reach, so this is the decode denominator).
                             Round-robin over M, 8 chained calls per block, 24 pinned weight
                             copies (~1 GB, larger than cache), 30 warm-up, 400 timed rounds.

Every result has median / min / max / spread_pct = (max - min) / median * 100 over the samples.
GEMV results also carry the stable statistic used as the audit denominator: stable =
top_half_median (median of the upper half of the samples), plus p10, p90 and
spread_p10_p90_pct. Output: one JSON file (label, mlx version, chip, date, results).

Timing rule: warm-up iterations first, then each iteration is timed around op() + mx.eval(out),
so lazy graphs are fully materialised inside the timer.

Run only through gpuq (no direct GPU use):
    GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq scripts/dev/gpuq run --priority 1 --quiet --timeout 10 \
        --label peak-probe -- <venv python> scripts/research/probe_peak.py --out PATH
    add --pilot for the one-rep small-size pilot.

The helpers (stats, time_loop, rate, make_report, validate_report) are pure and CPU-only; the MLX
imports live inside measure() so unit tests never touch the GPU.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

LABEL = "measured peak on this machine (M5 Max)"
GIB = 1024**3
GEMV_MS = (1, 2, 4, 8)  # decode-like batch rows
GEMV_CHAIN = 8  # GEMV calls per timed block (same pinned weights, distinct inputs)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested with fake ops and synthetic timings)
# ---------------------------------------------------------------------------
def stats(values: list[float]) -> dict:
    """median / min / max / spread over finite positive samples. Raises on empty or bad input."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("stats: no samples")
    if any((not math.isfinite(v)) or v <= 0 for v in vals):
        raise ValueError(f"stats: non-positive or non-finite sample in {vals}")
    med = statistics.median(vals)
    lo, hi = min(vals), max(vals)
    return {
        "n": len(vals),
        "median": med,
        "min": lo,
        "max": hi,
        "spread_pct": (hi - lo) / med * 100.0,
    }


def time_loop(
    op: Callable[[], object],
    eval_fn: Callable[[object], None],
    warmup: int,
    iters: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Run op() warmup times untimed, then iters times timed (op + eval_fn(out)). Seconds per iter."""
    if iters <= 0:
        raise ValueError("time_loop: iters must be positive")
    for _ in range(warmup):
        eval_fn(op())
    durs = []
    for _ in range(iters):
        t0 = clock()
        out = op()
        eval_fn(out)
        durs.append(clock() - t0)
    return durs


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]) of an already sorted, non-empty list."""
    if not sorted_vals:
        raise ValueError("percentile: no samples")
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def robust_stats(values: list[float]) -> dict:
    """Stable statistics for noisy GPU timings.

    stable = top_half_median: the median of the upper half of the samples. Interference (clock dips,
    a stray sync) only pulls samples DOWN, so the upper half is the sustained rate.
    spread_p10_p90_pct = (p90 - p10) / median * 100: spread one outlier cannot inflate.
    """
    base = stats(values)
    srt = sorted(float(v) for v in values)
    p10, p90 = percentile(srt, 0.10), percentile(srt, 0.90)
    return {
        "p10": p10,
        "p90": p90,
        "top_half_median": statistics.median(srt[len(srt) // 2 :]),
        "spread_p10_p90_pct": (p90 - p10) / base["median"] * 100.0,
    }


def interleaved_times(
    ops: dict[str, Callable[[], object]],
    eval_fn: Callable[[object], None],
    warm_rounds: int,
    rounds: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, list[float]]:
    """Time several ops round-robin: every round times each op once, in the same order, so clock or
    thermal drift hits all ops alike. Warm-up rounds run every op untimed first.
    Returns seconds per call for each op, one entry per round."""
    if rounds <= 0:
        raise ValueError("interleaved_times: rounds must be positive")
    for _ in range(warm_rounds):
        for op in ops.values():
            eval_fn(op())
    out: dict[str, list[float]] = {k: [] for k in ops}
    for _ in range(rounds):
        for name, op in ops.items():
            t0 = clock()
            res = op()
            eval_fn(res)
            out[name].append(clock() - t0)
    return out


def rate(work: float, seconds: list[float]) -> list[float]:
    """work per second for each timed iteration (work = bytes or FLOP per iteration)."""
    if any(s <= 0 for s in seconds):
        raise ValueError("rate: non-positive duration")
    return [work / s for s in seconds]


RESULT_KEYS = ("name", "unit", "n", "median", "min", "max", "spread_pct", "values")


def make_result(name: str, unit: str, values: list[float], **extra) -> dict:
    s = stats(values)
    return {
        "name": name,
        "unit": unit,
        **{k: s[k] for k in ("n", "median", "min", "max", "spread_pct")},
        "values": [float(v) for v in values],
        **extra,
    }


def make_robust_result(name: str, unit: str, values: list[float], **extra) -> dict:
    """make_result plus the stable statistic (top_half_median) that the audit uses as a denominator."""
    r = make_result(name, unit, values, **extra)
    rs = robust_stats(values)
    r.update(rs)
    r["stable"] = rs["top_half_median"]
    r["stable_stat"] = "top_half_median"
    return r


def make_report(results: list[dict], mlx_version: str, chip: str, date: str) -> dict:
    rep = {
        "label": LABEL,
        "mlx_version": mlx_version,
        "chip": chip,
        "date": date,
        "results": results,
    }
    validate_report(rep)
    return rep


def validate_report(rep: dict) -> None:
    """Schema check: raises ValueError on any missing or inconsistent field (fail closed)."""
    for k in ("label", "mlx_version", "chip", "date", "results"):
        if not rep.get(k):
            raise ValueError(f"report: missing {k}")
    if rep["label"] != LABEL:
        raise ValueError("report: label mismatch")
    if not isinstance(rep["results"], list) or not rep["results"]:
        raise ValueError("report: no results")
    names = set()
    for r in rep["results"]:
        for k in RESULT_KEYS:
            if k not in r:
                raise ValueError(f"result {r.get('name')}: missing {k}")
        if r["name"] in names:
            raise ValueError(f"duplicate result {r['name']}")
        names.add(r["name"])
        if len(r["values"]) != r["n"]:
            raise ValueError(f"result {r['name']}: n != len(values)")
        if not (r["min"] <= r["median"] <= r["max"]):
            raise ValueError(f"result {r['name']}: median outside [min, max]")
        if not (r["median"] > 0 and math.isfinite(r["median"])):
            raise ValueError(f"result {r['name']}: non-positive median")


def _chip() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    return out or "unknown"


# ---------------------------------------------------------------------------
# MLX measurements (GPU: only ever reached through gpuq)
# ---------------------------------------------------------------------------
def measure(pilot: bool) -> tuple[list[dict], str]:
    import mlx.core as mx  # noqa: PLC0415  (GPU import kept out of the pure helpers)

    if pilot:
        bw_shape = (2**14, 2**10)  # 32 MiB bf16
        sq, pf_shape, gemv_nk, warm, iters = (1024, (256, 512, 1024), (1024, 512), 1, 3)
        gemv_warm, gemv_rounds, gemv_copies = 2, 3, 2
    else:
        bw_shape = (2**21, 2**10)  # 2^31 bf16 elements = 4 GiB
        sq, pf_shape, gemv_nk, warm, iters = (
            8192,
            (2048, 5120, 17408),
            (17408, 5120),
            5,
            20,
        )
        # long warm-up; 400 interleaved rounds per M; 24 pinned copies = ~1 GB working set
        gemv_warm, gemv_rounds, gemv_copies = 30, 400, 24
    if not pilot and bw_shape[0] * bw_shape[1] * 2 < 4 * GIB:
        raise RuntimeError("bandwidth array must be >= 4 GiB")
    results: list[dict] = []
    ev = mx.eval

    # 1. bandwidth
    x = mx.ones(bw_shape, dtype=mx.bfloat16)
    ev(x)
    nbytes = x.nbytes
    secs = time_loop(lambda x=x: mx.sum(x, axis=0), ev, warm, iters)
    results.append(
        make_result(
            "bw_read_sum_bf16",
            "GB/s",
            [v / 1e9 for v in rate(nbytes, secs)],
            bytes=nbytes,
            shape=list(bw_shape),
        )
    )
    secs = time_loop(lambda x=x: x * 2, ev, warm, iters)  # read nbytes + write nbytes
    results.append(
        make_result(
            "bw_copy_rw_bf16",
            "GB/s",
            [2 * nbytes / s / 1e9 for s in secs],
            bytes=2 * nbytes,
            shape=list(bw_shape),
        )
    )
    del x

    # 2. matmul FLOP/s
    for dt, dname in ((mx.bfloat16, "bf16"), (mx.float16, "fp16")):
        for shape_name, (m, k, n) in (("sq", (sq, sq, sq)), ("prefill27b", pf_shape)):
            a = mx.random.normal((m, k), dtype=dt)
            b = mx.random.normal((k, n), dtype=dt)
            ev(a, b)
            flop = 2.0 * m * k * n
            secs = time_loop(lambda a=a, b=b: a @ b, ev, warm, iters)
            results.append(
                make_result(
                    f"matmul_{shape_name}_{dname}",
                    "TFLOP/s",
                    [v / 1e12 for v in rate(flop, secs)],
                    shape=[m, k, n],
                    flop_per_iter=flop,
                )
            )
            del a, b

    # 3. quantized GEMV (decode-like), 4-bit group 64
    n_out, k_in = gemv_nk
    # Packed weight copies, pinned for the whole run and cycled through by every call. One copy
    # (~45 MB) would be re-read from the system-level cache and report more than DRAM bandwidth;
    # the copies together exceed any cache, like a real decode step that streams distinct layers.
    wqs = []
    for _ in range(gemv_copies):
        w = mx.random.normal((n_out, k_in), dtype=mx.bfloat16)
        wqs.append(mx.quantize(w, group_size=64, bits=4))
        ev(*wqs[-1])
        del w
    wbytes = sum(a.nbytes for a in wqs[0])
    # Each timed block runs GEMV_CHAIN calls with distinct inputs, so one host sync is amortised
    # over several kernels. The M values are timed round-robin so drift hits all of them alike.
    # cursor advances on every call, so consecutive calls hit different pinned copies.
    xs = {
        m: [mx.random.normal((m, k_in), dtype=mx.bfloat16) for _ in range(GEMV_CHAIN)]
        for m in GEMV_MS
    }
    ev(*[x for xl in xs.values() for x in xl])
    cursor = [0]

    def block(m):
        def run():
            outs = []
            for x in xs[m]:
                wq, sc, bi = wqs[cursor[0] % gemv_copies]
                cursor[0] += 1
                outs.append(
                    mx.quantized_matmul(
                        x, wq, sc, bi, transpose=True, group_size=64, bits=4
                    )
                )
            return outs

        return run

    secs_by_m = interleaved_times(
        {m: block(m) for m in GEMV_MS},
        lambda outs: ev(*outs),
        gemv_warm,
        gemv_rounds,
    )
    for m in GEMV_MS:
        per_call = [s / GEMV_CHAIN for s in secs_by_m[m]]
        results.append(
            make_robust_result(
                f"qgemv_w4g64_M{m}",
                "GB/s",
                [v / 1e9 for v in rate(wbytes, per_call)],
                bytes=wbytes,
                shape=[m, k_in, n_out],
                chain=GEMV_CHAIN,
                weight_copies=gemv_copies,
                weight_working_set_bytes=wbytes * gemv_copies,
                interleaved=True,
            )
        )
    return results, mx.__version__


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Measure this machine's MLX peaks (bandwidth, FLOP/s, GEMV)."
    )
    p.add_argument("--out", required=True, help="output json path")
    p.add_argument(
        "--pilot",
        action="store_true",
        help="one-rep small-size pilot (validation only)",
    )
    a = p.parse_args(argv)
    try:
        results, mlx_version = measure(a.pilot)
        rep = make_report(
            results,
            mlx_version=mlx_version,
            chip=_chip(),
            date=datetime.date.today().isoformat(),
        )
    except (
        Exception
    ) as exc:  # fail closed: any error is a non-zero exit, never a partial file
        print(f"probe_peak: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    # first-rep validation: every result has the requested number of finite positive samples
    expect = 3 if a.pilot else None
    for r in rep["results"]:
        if expect is not None and r["n"] != expect:
            print(
                f"probe_peak: pilot sample count mismatch in {r['name']}",
                file=sys.stderr,
            )
            return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    for r in rep["results"]:
        print(
            f"{r['name']:24s} median {r['median']:10.3f} {r['unit']:9s} "
            f"min {r['min']:10.3f} max {r['max']:10.3f} spread {r['spread_pct']:5.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
