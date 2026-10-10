"""GPU busy/idle over a window of decode rounds, from the Metal command-buffer interposer.

Active only when the process was started with DYLD_INSERT_LIBRARIES pointing at
libmetalrt_trace.dylib (scripts/research/metalrt_trace.mm on branch metalrt); otherwise inert.
``tick()`` is called once per round; rounds START..END are measured and logged once.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import time

logger = logging.getLogger(__name__)
START, END = 40, 80
_STATE: dict = {"lib": False, "n": 0, "t0": 0.0}


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
