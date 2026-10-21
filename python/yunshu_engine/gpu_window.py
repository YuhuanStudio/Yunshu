"""GPU busy/idle over a window of decode rounds, from the Metal command-buffer interposer.

Active only when the process was started with DYLD_INSERT_LIBRARIES pointing at
libmetalrt_trace.dylib (scripts/research/metalrt_trace.mm on branch metalrt); otherwise inert.
``tick()`` is called once per round; rounds START..END are measured and logged once.
``mark(name)`` notes a host point inside the window; with an interposer that exports
``metalrt_gaps`` each GPU idle interval is charged to the host segment it falls in (the segment
that starts at a mark is named after it), so the log says which part of a round the GPU waits on.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import time

logger = logging.getLogger(__name__)
START, END = 40, 80
_STATE: dict = {"lib": False, "n": 0, "t0": 0.0, "marks": None}


def _lib():
    if _STATE["lib"] is False:
        path = os.environ.get("DYLD_INSERT_LIBRARIES", "")
        _STATE["lib"] = None
        if "metalrt_trace" in path:
            try:
                lib = ctypes.CDLL(path)
                lib.metalrt_end.argtypes = [ctypes.c_char_p, ctypes.c_int]
                _STATE["lib"] = lib
            except OSError:
                pass
    return _STATE["lib"]


def tick() -> None:
    lib = _lib()
    if lib is None:
        return
    _STATE["n"] += 1
    n = _STATE["n"]
    if n == START:
        lib.metalrt_begin()
        _STATE["t0"] = time.perf_counter()
        _STATE["marks"] = [(time.monotonic(), "tick")]
    elif n == END:
        wall = (time.perf_counter() - _STATE["t0"]) * 1000
        buf = ctypes.create_string_buffer(4096)
        lib.metalrt_end(buf, 4096)
        stats = json.loads(buf.value.decode())
        rounds = END - START
        logger.info(
            "GPU window (%d rounds): wall %.2f ms/round, busy %.2f, idle %.2f, dispatches %d/round, commit cpu %.2f, stats %s",
            rounds,
            wall / rounds,
            stats["busy_union_ms"] / rounds,
            (wall - stats["busy_union_ms"]) / rounds,
            stats["dispatches"] // rounds,
            stats["commit_cpu_ms"] / rounds,
            stats,
        )
        marks, _STATE["marks"] = _STATE["marks"], None
        if marks and hasattr(lib, "metalrt_gaps"):
            lib.metalrt_gaps.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
            big = ctypes.create_string_buffer(1 << 21)
            lib.metalrt_gaps(big, 1 << 21, 0.02)
            charged = charge_gaps(json.loads(big.value.decode()), marks)
            logger.info(
                "GPU idle by host segment (ms/round, gaps > 20 us): %s",
                {
                    k: round(v / rounds, 3)
                    for k, v in sorted(charged.items(), key=lambda kv: -kv[1])
                },
            )
            spans = segment_times(marks)
            logger.info(
                "Host time by segment (ms/round): %s",
                {
                    k: round(v / rounds, 3)
                    for k, v in sorted(spans.items(), key=lambda kv: -kv[1])
                },
            )
    elif START < n < END and _STATE["marks"] is not None:
        _STATE["marks"].append((time.monotonic(), "tick"))


def mark(name: str) -> None:
    """A host point inside the measured window (no-op outside it)."""
    marks = _STATE["marks"]
    if marks is not None:
        marks.append((time.monotonic(), name))


def segment_times(marks) -> dict:
    """Host ms per segment (from one mark to the next, named after the opening mark)."""
    out: dict = {}
    for (t0, name), (t1, _) in zip(marks, marks[1:], strict=False):
        out[name] = out.get(name, 0.0) + (t1 - t0) * 1000
    return out


def charge_gaps(gaps, marks) -> dict:
    """Idle ms per host segment: a segment runs from one mark to the next and is named after
    the mark that opens it; idle after the last mark is charged to that mark."""
    out: dict = {}
    if not marks:
        return out
    bounds = [t for t, _ in marks[1:]] + [float("inf")]
    j = 0
    for start, end in gaps:
        while j + 1 < len(marks) and marks[j + 1][0] <= start:
            j += 1
        if start < marks[0][0]:
            start = marks[0][0]
        k = j
        while start < end and k < len(marks):
            stop = min(end, bounds[k])
            if stop > start:
                out[marks[k][1]] = out.get(marks[k][1], 0.0) + (stop - start) * 1000
            start = stop
            k += 1
    return out


def stage_begin() -> None:
    """Reset the dispatch counter (profile mode: the caller has just evaluated everything)."""
    lib = _lib()
    if lib is not None:
        lib.metalrt_begin()


def stage_end() -> int:
    """Dispatches since the last ``stage_begin``/``stage_end``, then reset; 0 without the interposer."""
    lib = _lib()
    if lib is None:
        return 0
    buf = ctypes.create_string_buffer(4096)
    lib.metalrt_end(buf, 4096)
    lib.metalrt_begin()
    return int(json.loads(buf.value.decode())["dispatches"])
