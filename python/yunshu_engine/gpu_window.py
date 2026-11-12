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
import statistics
import time

logger = logging.getLogger(__name__)
START, END = 40, 80
ROUNDS = END - START
# a round gap is a stall only when it exceeds STALL_X x the window's median round gap and STALL_MIN_S
STALL_X, STALL_MIN_S, STALL_MIN_HISTORY = 5.0, 0.05, 3
# marks is None outside a window; done once a steady window was reported.
_STATE: dict = {
    "lib": False,
    "n": 0,
    "t0": 0.0,
    "marks": None,
    "start": 0,
    "last": 0.0,
    "gaps": [],
    "done": False,
}


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


def tick(*, capture_kernels: bool = True) -> None:
    lib = _lib()
    if lib is None:
        return
    _STATE["n"] += 1
    n = _STATE["n"]
    capture = capture_kernels and hasattr(lib, "metalrt_profile")
    if capture and n == START - 1:
        lib.metalrt_cap_begin()
    elif capture and n == START:
        _log_kernel_profile(lib)
    now = time.monotonic()
    if _STATE["marks"] is not None:
        gap = now - _STATE["last"]
        gaps = _STATE["gaps"]
        if _is_stall(gap, gaps):
            median = statistics.median(gaps)
            _drop(
                lib,
                f"host gap of {gap * 1000:.0f} ms (median round {median * 1000:.0f} ms)",
            )
        elif n - _STATE["start"] < ROUNDS:
            gaps.append(gap)
            _STATE["marks"].append((now, "tick"))
            _STATE["last"] = now
            return
        else:
            _finish(lib)
            return
    if n >= START and not _STATE["done"]:
        lib.metalrt_begin()
        _STATE["start"] = n
        _STATE["t0"] = time.perf_counter()
        _STATE["last"] = now
        _STATE["gaps"] = []
        _STATE["marks"] = [(now, "tick")]


def _is_stall(gap: float, gaps: list) -> bool:
    """A gap far above the window's own median round time (a prefill, a stall), not a slow round."""
    if len(gaps) < STALL_MIN_HISTORY or gap <= STALL_MIN_S:
        return False
    return bool(gap > STALL_X * statistics.median(gaps))


def boundary(name: str) -> None:
    """A non-decode host event (prefill, request join or end): a window open now spans it, so it is
    contaminated. The window is discarded without tables and a fresh one opens at the next tick."""
    if _STATE["marks"] is not None:
        _drop(_STATE["lib"], f"boundary {name}")


def _drop(lib, reason: str) -> None:
    """Discard an open window: its counters are drained, no tables are printed, one line says why."""
    buf = ctypes.create_string_buffer(4096)
    lib.metalrt_end(buf, 4096)
    rounds = len(_STATE["marks"]) - 1
    _STATE["marks"] = None
    logger.info(
        "GPU window dropped (%s) after %d rounds: not steady decode", reason, rounds
    )


def _finish(lib) -> None:
    """Close a steady window: log its wall, busy/idle and the per-segment tables once."""
    wall = (time.perf_counter() - _STATE["t0"]) * 1000
    buf = ctypes.create_string_buffer(4096)
    lib.metalrt_end(buf, 4096)
    stats = json.loads(buf.value.decode())
    rounds = ROUNDS
    _STATE["done"] = True
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


def _log_kernel_profile(lib) -> None:
    """Per-kernel durations of the one round captured since the previous tick (interposer build with
    ``metalrt_profile``): the captured dispatches are replayed one encoder each with timestamps, by function
    name. The replay runs on the live buffers, so generation after this point is meaningless (profiling runs)."""
    lib.metalrt_cap_end.restype = ctypes.c_int
    lib.metalrt_profile.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.metalrt_profile.restype = ctypes.c_int
    count = lib.metalrt_cap_end()
    buf = ctypes.create_string_buffer(1 << 17)
    n = lib.metalrt_profile(80, buf, 1 << 17)
    logger.info(
        "GPU per-kernel (%d captured dispatches): %s", count, buf.raw[:n].decode()
    )


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
