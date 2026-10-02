"""The gpuq pause contract, serving side.

gpuq (``scripts/dev/gpuq.py``, preemption) SIGSTOPs a dev job while production traffic is served
and records every stopped interval. This module is the reader the serving / analysis code uses
to drop samples that overlap such an interval. It is written against the interface the gpuq
side defines (kept identical to ``scripts/dev/gpuq_pause.py::was_paused`` so the two reconcile
at merge without a behaviour change):

- the job JSON and the file named by ``$GPUQ_PAUSE_FILE`` hold ``{"pauses": [[t0, t1], ...]}``;
- timestamps are wall-clock seconds (``time.time()``), the daemon being another process;
- ``t1`` is ``null`` while the job is still stopped: the interval is open and ends at *now*;
- intervals are closed: touching at a single instant counts as an overlap;
- fail closed: a variable that is set but names an unreadable / malformed file means "paused";
  an unset variable (not running under gpuq) means "never paused".
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

ENV_VAR = "GPUQ_PAUSE_FILE"

Pause = tuple[float, float]


class PauseDataError(ValueError):
    """The pause data is present but cannot be trusted."""


def parse_pauses(data: Any, now: float | None = None) -> list[Pause]:
    """``[(t0, t1), ...]`` from a decoded pause document (``{"pauses": [...]}``) or a bare
    list. An open interval (``t1`` null) ends at ``now``. Raises :class:`PauseDataError` on any
    malformed entry: a partially understood pause list must not be used."""
    now = time.time() if now is None else now
    raw = data.get("pauses") if isinstance(data, dict) else data
    if raw is None:
        raw = []
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise PauseDataError("pauses is not a list")
    out: list[Pause] = []
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise PauseDataError(f"bad pause entry: {item!r}")
        a, b = item
        try:
            t0 = float(a)
            t1 = now if b is None else float(b)
        except (TypeError, ValueError):
            raise PauseDataError(f"bad pause entry: {item!r}") from None
        if not (math.isfinite(t0) and math.isfinite(t1)) or t1 < t0:
            raise PauseDataError(f"bad pause entry: {item!r}")
        out.append((t0, t1))
    return out


def read_pause_file(path: str | None = None, now: float | None = None) -> list[Pause]:
    """The pauses in ``path`` (default ``$GPUQ_PAUSE_FILE``); [] when no path is given.
    Raises :class:`PauseDataError` when the file cannot be read or parsed."""
    path = path or os.environ.get(ENV_VAR)
    if not path:
        return []
    try:
        with open(path) as f:
            return parse_pauses(json.load(f), now)
    except (OSError, ValueError) as exc:  # json errors are ValueErrors
        raise PauseDataError(f"cannot read pause file {path}: {exc}") from exc


def overlaps(t0: float, t1: float, pauses: Iterable[Sequence[float | None]]) -> bool:
    """True when [t0, t1] touches any pause interval (closed intervals; an open pause, with
    ``t1`` None, extends to infinity). ``pauses`` are ``(t0, t1)`` pairs."""
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 < t0:
        raise ValueError("sample interval must be finite with t0 <= t1")
    for a, b in pauses:
        if a is None:
            raise PauseDataError("pause without a start")
        end = math.inf if b is None else b
        if a <= t1 and end >= t0:
            return True
    return False


def was_paused(
    t0: float, t1: float, path: str | None = None, now: float | None = None
) -> bool:
    """Whether [t0, t1] overlapped a recorded pause. Fails closed: an unreadable pause file
    answers True; no file configured answers False."""
    path = path or os.environ.get(ENV_VAR)
    if not path:
        return False
    try:
        pauses = read_pause_file(path, now)
    except PauseDataError:
        return True
    return overlaps(t0, t1, pauses)


def paused_seconds(t0: float, t1: float, pauses: Iterable[Sequence[float]]) -> float:
    """Seconds of [t0, t1] covered by pauses (intervals merged, so overlaps are not double
    counted)."""
    spans = sorted((max(a, t0), min(b, t1)) for a, b in pauses if a <= t1 and b >= t0)
    total, cur_end = 0.0, -math.inf
    for a, b in spans:
        a = max(a, cur_end)
        if b > a:
            total += b - a
        cur_end = max(cur_end, b)
    return total


def clean_samples(
    samples: Iterable[tuple[float, float, Any]],
    pauses: Sequence[Sequence[float | None]],
) -> tuple[list[Any], int]:
    """Split ``(t0, t1, payload)`` samples into the payloads that never touched a pause and
    the count of dropped ones. The count is returned, not hidden: a report must say how many
    samples the pauses cost."""
    kept, dropped = [], 0
    for t0, t1, payload in samples:
        if overlaps(t0, t1, pauses):
            dropped += 1
        else:
            kept.append(payload)
    return kept, dropped
