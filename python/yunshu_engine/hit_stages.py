"""One log line per request that splits the host time before its first token into stages.

perf_counter only (no mx.eval, no sync): the runner thread calls ``begin`` when it admits a
request, the stages below add their elapsed time, and ``finish`` logs one line when the first
token is out. The stage times let a pilot attribute TTFT (they sum to the admit-to-first-token
span within the unlisted remainder printed as ``other``).

Line: ``turn stages: ids=N hit=M | standing=a vocab=b add=c (lookup=d plan=e boundaries=f hash=g
window=h) wait_first=i emit=j other=k total=t ms`` where ``add`` is ``RoundDriver.add`` (lookup
and plan run inside it; boundaries and hash run inside those), ``wait_first`` is from the end
of ``add`` to the first token (the tail prefill and its first sampling), ``emit`` is the hand-off
of the first token to the consumer.
"""

from __future__ import annotations

import contextlib
import logging
import time

logger = logging.getLogger(__name__)

_REC: dict | None = None
_NAMES = ("standing", "vocab", "add", "lookup", "plan", "boundaries", "hash", "window")


def begin() -> None:
    global _REC
    _REC = {"t0": time.perf_counter()}


def active() -> bool:
    return _REC is not None


def add(name: str, seconds: float) -> None:
    rec = _REC
    if rec is not None:
        rec[name] = rec.get(name, 0.0) + seconds


@contextlib.contextmanager
def stage(name: str):
    if _REC is None:
        yield
        return
    t = time.perf_counter()
    try:
        yield
    finally:
        add(name, time.perf_counter() - t)


def mark(name: str) -> None:
    """Remember a point in time (``add_end``) for ``finish``."""
    rec = _REC
    if rec is not None:
        rec[name] = time.perf_counter()


def finish(ids: int, hit: int, emit_s: float) -> None:
    """Log and clear the record (first token handed out)."""
    global _REC
    rec, _REC = _REC, None
    if rec is None:
        return
    now = time.perf_counter()
    total = now - rec["t0"]
    wait = now - emit_s - rec["add_end"] if "add_end" in rec else 0.0
    listed = sum(rec.get(k, 0.0) for k in ("standing", "vocab", "add")) + wait + emit_s
    ms = lambda k: rec.get(k, 0.0) * 1e3  # noqa: E731
    logger.info(
        "turn stages: ids=%d hit=%d | standing=%.1f vocab=%.1f add=%.1f (lookup=%.1f "
        "plan=%.1f boundaries=%.1f hash=%.1f window=%.1f) wait_first=%.1f emit=%.1f "
        "other=%.1f total=%.1f ms",
        ids,
        hit,
        ms("standing"),
        ms("vocab"),
        ms("add"),
        ms("lookup"),
        ms("plan"),
        ms("boundaries"),
        ms("hash"),
        ms("window"),
        wait * 1e3,
        emit_s * 1e3,
        (total - listed) * 1e3,
        total * 1e3,
    )
