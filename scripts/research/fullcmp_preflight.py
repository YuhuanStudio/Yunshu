#!/usr/bin/env python3
"""Preflight gate: is swap stable over a window? (FULLCMP_SPEC.md, "Telemetry" / swap guard.)

Samples `sysctl vm.swapusage` every INTERVAL seconds for WINDOW seconds and fails when the used
swap grew by more than LIMIT bytes between the first and last sample. A swap that keeps growing
means the machine is paging under the benchmark, so no cell run now is trustworthy.

    python scripts/research/fullcmp_preflight.py --window-s 300 --interval-s 10 --limit-mb 64

Prints the full series as JSON and exits 0 when stable, 1 when swap grew past the limit, 2 when a
sample cannot be read (fail closed: an unreadable swap counter is not a stable one).

Pure core: swap_stable(sample=..., sleep=..., clock=...) takes injectable callables so the
decision logic is unit-tested on synthetic series without touching the machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullcmp_telemetry as tm  # noqa: E402

MB = 1024 * 1024


def read_swap_used_bytes() -> int:
    """Used swap in bytes from `sysctl vm.swapusage`. Raises tm.ProbeError when unreadable."""
    return tm.parse_swapusage(tm._run(["sysctl", "vm.swapusage"]))["used_bytes"]


def swap_stable(
    window_s: float = 300.0,
    interval_s: float = 10.0,
    limit_bytes: int = 64 * MB,
    sample: Callable[[], int] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    """Sample swap used over the window; stable when (last - first) <= limit_bytes.

    The series has window_s / interval_s + 1 points (t = 0, interval, ..., window). Any exception
    from `sample` propagates: a missing reading is never counted as stable.
    """
    if window_s <= 0 or interval_s <= 0:
        raise ValueError("window_s and interval_s must be positive")
    sample = sample or read_swap_used_bytes
    sleep = sleep or time.sleep
    clock = clock or time.time
    n_steps = int(round(window_s / interval_s))
    series = []
    t0 = clock()
    for i in range(n_steps + 1):
        if i:
            sleep(interval_s)
        series.append({"t": round(clock() - t0, 3), "used_bytes": int(sample())})
    start, end = series[0]["used_bytes"], series[-1]["used_bytes"]
    growth = end - start
    return {
        "check": "swap_stable",
        "ok": growth <= limit_bytes,
        "window_s": window_s,
        "interval_s": interval_s,
        "limit_bytes": limit_bytes,
        "start_used_bytes": start,
        "end_used_bytes": end,
        "growth_bytes": growth,
        "max_used_bytes": max(p["used_bytes"] for p in series),
        "series": series,
    }


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(description="Swap-stability preflight (vm.swapusage).")
    p.add_argument("--window-s", type=float, default=300.0)
    p.add_argument("--interval-s", type=float, default=10.0)
    p.add_argument("--limit-mb", type=float, default=64.0)
    p.add_argument("--out", default=None, help="also write the result json here")
    a = p.parse_args(argv)
    try:
        res = swap_stable(
            window_s=a.window_s,
            interval_s=a.interval_s,
            limit_bytes=int(a.limit_mb * MB),
        )
    except tm.ProbeError as exc:
        print(json.dumps({"check": "swap_stable", "ok": False, "error": str(exc)}))
        return 2
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(
        f"swap_stable: {'PASS' if res['ok'] else 'FAIL'} "
        f"(growth {res['growth_bytes'] / MB:+.1f} MB over {a.window_s:.0f} s, "
        f"limit {a.limit_mb:.0f} MB)",
        file=sys.stderr,
    )
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
