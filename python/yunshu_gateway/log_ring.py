"""In-memory ring of recent server log records for the console log viewer.

A ``logging.Handler`` keeps the last ``CAPACITY`` records (level, time, logger, message)
with a monotonically increasing id, so a client can poll incrementally with ``since_id``.
Messages are redacted (credentials, tokens, echoed request payload fragments) *when the
record is emitted*, never later, and cut to ``MAX_MESSAGE`` characters. Nothing is written
to disk. Memory is bounded by ``CAPACITY * MAX_MESSAGE`` characters.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

CAPACITY = 2000
MAX_MESSAGE = 2048

_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def level_no(name: str | None) -> int:
    """Numeric level for a name (``warning``, ``ERROR``...); ValueError for an unknown one."""
    if not name:
        return 0
    key = name.strip().upper()
    if key not in _LEVELS:
        raise ValueError(f"unknown log level {name!r}")
    return _LEVELS[key]


def _scrub(text: str) -> str:
    from yunshu_cli.bundle import scrub_line

    return scrub_line(text, MAX_MESSAGE)


class RingHandler(logging.Handler):
    def __init__(self, capacity: int = CAPACITY) -> None:
        super().__init__(level=logging.NOTSET)
        self._ring: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=capacity
        )
        self._ring_lock = threading.Lock()
        self._next_id = 1
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.exc_info and record.exc_info[0] is not None:
                msg += f" [{record.exc_info[0].__name__}]"
            row = {
                "t": record.created,
                "level": record.levelname,
                "levelno": record.levelno,
                "logger": record.name,
                "msg": _scrub(msg),
            }
        except Exception:  # noqa: BLE001 - logging must never raise into the caller
            return
        with self._ring_lock:
            if len(self._ring) == self._ring.maxlen:
                self.dropped += 1
            row["id"] = self._next_id
            self._next_id += 1
            self._ring.append(row)

    def query(
        self,
        *,
        level: str | None = None,
        since: float | None = None,
        since_id: int = 0,
        q: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Newest ``limit`` matching records, oldest first. ``next_id`` is the cursor for
        the next incremental poll (the last id in the ring, even when nothing matched)."""
        minimum = level_no(level)
        needle = q.lower() if q else None
        with self._ring_lock:
            rows = list(self._ring)
            next_id = self._next_id - 1
            dropped = self.dropped
        out = [
            r
            for r in rows
            if r["id"] > since_id
            and r["levelno"] >= minimum
            and (since is None or r["t"] >= since)
            and (
                needle is None
                or needle in r["msg"].lower()
                or needle in r["logger"].lower()
            )
        ]
        out = out[-max(1, int(limit)) :]
        return {
            "records": [{k: v for k, v in r.items() if k != "levelno"} for r in out],
            "next_id": next_id,
            "dropped": dropped,
            "capacity": self._ring.maxlen,
            "server_time": time.time(),
        }


_handler: RingHandler | None = None


def handler() -> RingHandler | None:
    return _handler


def install(capacity: int = CAPACITY) -> RingHandler:
    """Attach the ring to the root logger once; later calls return the same handler."""
    global _handler
    if _handler is None:
        _handler = RingHandler(capacity)
        logging.getLogger().addHandler(_handler)
    return _handler


def uninstall() -> None:
    global _handler
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler = None
