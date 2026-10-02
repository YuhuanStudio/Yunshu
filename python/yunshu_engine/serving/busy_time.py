"""GPU-busy accounting: how much wall time the serving threads spent doing work.

A :class:`BusyMeter` wraps each unit of GPU work (one executor slice of the VLM batch runner,
one step of the round driver) and keeps a cumulative busy-seconds counter. The idle fraction
over a window is ``1 - busy / wall`` from two snapshots of the counter, so a scrape of
``/metrics`` (``yunshu_gpu_busy_seconds_total``) is enough to compute it with ``rate()``.

Nested spans on one thread (a slice that contains a driver step) are counted once; spans from
different threads are merged as a union of intervals, not summed. The clock is injectable
(tests use a fake one).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager


class BusyMeter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._depth = 0  # spans currently open, across threads
        self._since = 0.0  # when the meter became busy (depth went 0 -> 1)
        self._busy = 0.0  # closed busy time
        self._t0 = clock()
        self.spans = 0

    def enter(self) -> None:
        with self._lock:
            if self._depth == 0:
                self._since = self._clock()
            self._depth += 1
            self.spans += 1

    def exit(self) -> None:
        with self._lock:
            if self._depth <= 0:
                return  # unbalanced exit: ignore rather than go negative
            self._depth -= 1
            if self._depth == 0:
                self._busy += max(self._clock() - self._since, 0.0)

    @contextmanager
    def span(self):
        self.enter()
        try:
            yield
        finally:
            self.exit()

    def busy_seconds(self) -> float:
        """Cumulative busy seconds, including a span that is open right now."""
        with self._lock:
            extra = max(self._clock() - self._since, 0.0) if self._depth else 0.0
            return self._busy + extra

    def uptime_seconds(self) -> float:
        return max(self._clock() - self._t0, 0.0)

    def snapshot(self) -> dict:
        busy, up = self.busy_seconds(), self.uptime_seconds()
        return {
            "busy_seconds": round(busy, 3),
            "uptime_seconds": round(up, 3),
            "idle_fraction": idle_fraction(busy, up),
            "spans": self.spans,
        }


def idle_fraction(busy_s: float, wall_s: float) -> float | None:
    """``1 - busy / wall`` clamped to [0, 1]; None when there is no wall time to speak of
    (no data is reported as missing, not as 0 or 1)."""
    if wall_s <= 0:
        return None
    return round(min(max(1.0 - busy_s / wall_s, 0.0), 1.0), 4)


def window_idle_fraction(
    busy_a: float, wall_a: float, busy_b: float, wall_b: float
) -> float | None:
    """Idle fraction between two snapshots (busy seconds, wall seconds) of the counter."""
    if busy_b < busy_a or wall_b <= wall_a:
        return None  # a counter reset or a non-advancing clock: no answer
    return idle_fraction(busy_b - busy_a, wall_b - wall_a)
