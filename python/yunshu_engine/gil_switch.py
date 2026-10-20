"""Shorter GIL switch interval for the serving process.

CPython lets a thread that wants the GIL wait up to ``sys.getswitchinterval()`` (5 ms) when
another thread is running Python. During decode the generation thread is that thread, so the
event loop (SSE writes, new requests) saw p99 wake-up lag of ~5 ms. 0.5 ms cuts it to ~2 ms on
Qwen3.8-27B with tokens/s unchanged. Scheduling only: output is unchanged.
"""

from __future__ import annotations

import sys

from . import settings

_original: float | None = None  # the interval before this module first changed it


def apply() -> float:
    """Set the interval from YUNSHU_GIL_SWITCH_MS (0 restores CPython's 5 ms if this
    module had changed it); returns seconds in effect."""
    global _original
    ms = float(settings.get("YUNSHU_GIL_SWITCH_MS") or 0.0)
    if ms > 0:
        if _original is None:
            _original = sys.getswitchinterval()
        sys.setswitchinterval(ms / 1000.0)
    elif _original is not None:
        sys.setswitchinterval(_original)
        _original = None
    return sys.getswitchinterval()
