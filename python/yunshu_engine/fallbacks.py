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


def _detail_text(detail) -> str:
    try:
        return str(detail()) if callable(detail) else str(detail)
    except Exception:  # a detail probe must never fail the inference call that hit the fallback
        return "<detail unavailable>"


def record(kind: str, detail) -> bool:
    """Count one fallback of ``kind``; log it once per process. Returns True on the first one.

    ``detail`` may be a zero-argument callable: it is evaluated only for the first event of a kind,
    so hot-path call sites pay one lock and one dict update per fallback.
    """
    with _LOCK:
        _COUNTS[kind] = _COUNTS.get(kind, 0) + 1
        first = kind not in _FIRST
        if first:
            text = _detail_text(detail)
            _FIRST[kind] = text
    if first:
        logger.warning("Kernel fallback (%s): %s; using the upstream path", kind, text)
    return first


def counts() -> dict[str, dict[str, object]]:
    with _LOCK:
        return {k: {"count": n, "first": _FIRST.get(k, "")} for k, n in _COUNTS.items()}


def reset() -> None:
    with _LOCK:
        _COUNTS.clear()
        _FIRST.clear()
