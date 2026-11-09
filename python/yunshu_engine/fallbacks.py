"""Process-wide count of kernel/path fallbacks, logged once per kind with the shape that caused it.

A fused kernel that cannot take a shape falls back to the upstream path (correct, slower). That must
never be silent: ``record`` counts every fallback and logs the first one of each kind at WARNING.
``counts()`` feeds the ``/monitoring/kernel-fallbacks`` route.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}
_FIRST: dict[str, str] = {}


def record(kind: str, detail: str) -> bool:
    """Count one fallback of ``kind``; log it once per process. Returns True on the first one."""
    with _LOCK:
        _COUNTS[kind] = _COUNTS.get(kind, 0) + 1
        first = kind not in _FIRST
        if first:
            _FIRST[kind] = detail
    if first:
        logger.warning(
            "Kernel fallback (%s): %s; using the upstream path", kind, detail
        )
    return first


def counts() -> dict[str, dict[str, object]]:
    with _LOCK:
        return {k: {"count": n, "first": _FIRST.get(k, "")} for k, n in _COUNTS.items()}


def reset() -> None:
    with _LOCK:
        _COUNTS.clear()
        _FIRST.clear()
