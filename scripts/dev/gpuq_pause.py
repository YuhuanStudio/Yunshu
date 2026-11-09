"""Pause awareness for jobs run under gpuq (stdlib only, python 3.9 safe).

When a production Yunshu server is configured, gpuq SIGSTOPs a job while user traffic is served and records each
interval to the file named by $GPUQ_PAUSE_FILE. A timed benchmark sample that overlaps such an interval measured
the pause, not the engine, so it must never reach a result table:

    from gpuq_pause import timed, PausedSampleError
    rec = timed(lambda: send(url, body))          # re-runs the sample if it overlapped a pause

Fail closed: if GPUQ_PAUSE_FILE is set but unreadable the answer is "paused"; outside gpuq (variable unset)
nothing is ever paused. Timestamps are wall-clock (time.time()) because the daemon is another process.

The pause file is parsed by the serving-side reader, ``python/yunshu_engine/serving/pauses.py`` (the single
parser). It is loaded by file path so this script stays stdlib-only and does not import the engine package.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_RETRY_WAIT_S = 1.0

_PAUSES_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "yunshu_engine"
    / "serving"
    / "pauses.py"
)


def _load_reader():
    spec = importlib.util.spec_from_file_location(
        "_yunshu_serving_pauses", _PAUSES_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the pause reader at {_PAUSES_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pauses = _load_reader()


class PausedSampleError(RuntimeError):
    """A timed sample kept overlapping gpuq pauses; the job's measurement is invalid."""


def pause_intervals(path: str | None = None) -> list[tuple[float, float]]:
    """The recorded pauses as (t0, t1); an open (still stopped) interval ends at now.

    Raises ValueError (``PauseDataError``) when the file is configured but cannot be read or parsed."""
    return _pauses.read_pause_file(path)


def paused_seconds(t0: float, t1: float, spans: list[tuple[float, float]]) -> float:
    """Seconds of [t0, t1] covered by ``spans`` (from ``pause_intervals()``); overlapping pauses are merged so none is counted twice."""
    return _pauses.paused_seconds(t0, t1, spans)


def was_paused(t0: float, t1: float, path: str | None = None) -> bool:
    """True if any pause interval overlaps [t0, t1] (wall-clock seconds). Fails closed on a bad file."""
    return _pauses.was_paused(t0, t1, path)


def timed(fn: Callable[[], Any], retries: int = 5, path: str | None = None) -> Any:
    """Run `fn` (one timed sample); discard and re-run it if it overlapped a pause.

    Raises PausedSampleError after `retries` consecutive overlapped attempts, so the job fails instead of
    reporting numbers that include paused time."""
    for _ in range(max(1, retries)):
        t0 = time.time()
        result = fn()
        if not was_paused(t0, time.time(), path):
            return result
        time.sleep(_RETRY_WAIT_S)
    raise PausedSampleError(
        f"sample overlapped a gpuq pause on {retries} attempts; measurement invalid"
    )


def total_paused(path: str | None = None) -> float:
    """Sum of pause seconds so far (for logging in result files)."""
    try:
        return sum(b - a for a, b in pause_intervals(path))
    except (OSError, ValueError, TypeError):
        return float("nan")
