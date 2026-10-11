"""Shorter GIL switch interval for the serving process.

CPython lets a thread that wants the GIL wait up to ``sys.getswitchinterval()`` (5 ms) when
another thread is running Python. During decode the generation thread is that thread, so the
event loop (SSE writes, new requests) saw p99 wake-up lag of ~5 ms. 0.5 ms cuts it to ~2 ms on
Qwen3.8-27B with tokens/s unchanged. Scheduling only: output is unchanged.
"""

from __future__ import annotations

import sys

from . import settings


def apply() -> float:
    """Set the interval from YUNSHU_GIL_SWITCH_MS (0 keeps CPython's 5 ms); returns seconds in effect."""
    ms = float(settings.get("YUNSHU_GIL_SWITCH_MS") or 0.0)
    if ms > 0:
        sys.setswitchinterval(ms / 1000.0)
    return sys.getswitchinterval()
