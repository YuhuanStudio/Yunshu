"""Executor-thread -> asyncio consumer bridge for streaming generation.

The old pattern checked ``q.full()`` on the producer thread and then scheduled
``q.put_nowait`` on the loop: the check and the enqueue were not atomic, so a
burst scheduled more callbacks than free slots and the surplus (including the
"overflow" error sentinel) died as ``QueueFull`` inside the loop callback.

Here the producer reserves a slot under a lock before scheduling, the loop
callback releases it, and terminal items (errors / the end sentinel) are
delivered out-of-band of the capacity so they can never be lost. The producer
never waits long: on a persistently full queue it fails the stream with a
visible error and tells the caller to stop generating.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# At most two terminal items reach a stream: an error, then the end sentinel.
_TERMINAL_SLOTS = 2


class _Queue(asyncio.Queue):
    """A queue with room kept free for terminal items: data may fill only
    ``data_capacity`` slots, so an error or the end sentinel always fits."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize=maxsize + _TERMINAL_SLOTS)
        self.data_capacity = maxsize


def make_stream_queue(maxsize: int) -> asyncio.Queue:
    return _Queue(maxsize)


class StreamBridge:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        q: asyncio.Queue,
        is_terminal: Callable[[Any], bool],
        *,
        on_overflow: Callable[[], None] | None = None,
        wait_s: float = 0.05,
        overflow_error: str = "Streaming queue overflow — output truncated",
    ) -> None:
        self._loop = loop
        self._q = q
        self._is_terminal = is_terminal
        self._on_overflow = on_overflow
        self._wait_s = wait_s
        self._overflow_error = overflow_error
        self._lock = threading.Lock()
        self._capacity = getattr(q, "data_capacity", q.maxsize)
        self._reserved = 0  # scheduled data items not yet in the queue
        self._closed = False  # a terminal has been scheduled; drop the rest

    def _schedule(self, fn: Callable[[], None]) -> bool:
        try:
            self._loop.call_soon_threadsafe(fn)
            return True
        except RuntimeError:  # loop closed: consumer is gone
            return False

    def _terminal(self, item: Any) -> None:
        def deliver() -> None:
            try:
                self._q.put_nowait(item)
            except asyncio.QueueFull:  # only a plain queue without terminal slots
                logger.warning("terminal item dropped: queue full")

        self._schedule(deliver)

    def put(self, item: Any) -> bool:
        """Deliver ``item``; False means the stream failed (stop producing)."""
        if self._is_terminal(item):
            with self._lock:
                already = self._closed
                self._closed = True
            # A terminal after overflow's error is redundant but harmless to
            # the consumer only if it is the end sentinel; keep it (exactly
            # one error, then the sentinel) unless it is a second error.
            if already and isinstance(item, BaseException):
                return False
            self._terminal(item)
            return True
        deadline = time.monotonic() + self._wait_s
        while True:
            with self._lock:
                if self._closed:
                    return False
                if self._q.qsize() + self._reserved < self._capacity:
                    self._reserved += 1
                    break
            if time.monotonic() >= deadline:
                return self._overflow()
            time.sleep(0.0005)

        def deliver() -> None:
            with self._lock:
                self._reserved -= 1
            try:
                self._q.put_nowait(item)
            except asyncio.QueueFull:  # unreachable by construction; fail loudly
                self._overflow()

        if not self._schedule(deliver):
            with self._lock:
                self._reserved -= 1
            return False
        return True

    def _overflow(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._closed = True
        logger.warning("%s (client not draining)", self._overflow_error)
        self._terminal(Exception(self._overflow_error))
        if self._on_overflow is not None:
            self._on_overflow()
        return False
