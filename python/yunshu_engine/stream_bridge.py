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


class _Queue(asyncio.Queue):
    def put_terminal_nowait(self, item: Any) -> None:
        """Enqueue regardless of maxsize (loop thread only)."""
        self._put(item)
        self._unfinished_tasks += 1
        self._finished.clear()
        self._wakeup_next(self._getters)


def make_stream_queue(maxsize: int) -> asyncio.Queue:
    return _Queue(maxsize=maxsize)


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
            put = getattr(self._q, "put_terminal_nowait", None)
            if put is not None:
                put(item)
            else:  # plain queue: best effort
                try:
                    self._q.put_nowait(item)
                except asyncio.QueueFull:
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
                if self._q.qsize() + self._reserved < self._q.maxsize:
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
