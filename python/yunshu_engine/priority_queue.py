"""Heap-based priority queue for scheduler waiting requests.

Replaces the previous deque + sort pattern with O(log n) push/pop via heapq.

Two modes:
- FCFS: requests are served in insertion order (auto-incrementing sequence number)
- PRIORITY: higher-priority requests are served first; same-priority ties broken
  by insertion order (FIFO)

Thread-safe via a simple threading.Lock (CPython GIL makes this cheap).
"""
from __future__ import annotations

import heapq
import threading
from enum import Enum, auto
from typing import Generic, TypeVar

T = TypeVar("T")


class _QueueMode(Enum):
    FCFS = auto()
    PRIORITY = auto()


class RequestPriorityQueue(Generic[T]):
    """Heap-based priority queue supporting FCFS and PRIORITY modes.

    For FCFS mode:
        The heap key is (sequence_number, item). Items come out in insertion order.

    For PRIORITY mode:
        The heap key is (-priority, sequence_number, item). Higher priority values
        are served first; ties broken by insertion order (FIFO).

    ``push_front()`` is used for preempted requests — they get the lowest
    sequence number seen so far minus one, so they appear before any other
    request at the same priority level.
    """

    def __init__(self, mode: _QueueMode = _QueueMode.FCFS) -> None:
        self._mode = mode
        self._heap: list[tuple] = []
        self._seq: int = 0
        self._front_seq: int = -(1 << 30)
        self._lock = threading.Lock()

    # -- internal helpers ------------------------------------------------

    def _make_entry(self, priority: int, seq: int, item: T) -> tuple:
        """Build a heap entry tuple according to the queue mode."""
        if self._mode == _QueueMode.FCFS:
            return (seq, item)
        else:  # PRIORITY
            return (-priority, seq, item)

    def _extract_item(self, entry: tuple) -> T:
        """Extract the item from a heap entry tuple."""
        return entry[-1]

    # -- public API ------------------------------------------------------

    def push(self, item: T, priority: int = 0) -> None:
        """Insert *item* into the queue.

        *priority* is only used under PRIORITY mode.  Under FCFS it is
        silently ignored.
        """
        with self._lock:
            seq = self._seq
            self._seq += 1
            entry = self._make_entry(priority, seq, item)
            heapq.heappush(self._heap, entry)

    def push_front(self, item: T, priority: int = 0) -> None:
        """Insert *item* ahead of all existing items at the same priority.

        Used for preempted requests that must be re-scheduled before any
        newly-arriving requests at the same priority level.
        """
        with self._lock:
            # Use a special "front" sequence number that sorts before all
            # normal entries (which start at seq=0).  We start at a large
            # negative value and increment, so push_front calls maintain
            # FIFO order among themselves.
            front_seq = self._front_seq
            self._front_seq += 1

            if self._mode == _QueueMode.FCFS:
                entry = (front_seq, item)
            else:  # PRIORITY
                entry = (-priority, front_seq, item)
            heapq.heappush(self._heap, entry)

    def pop(self) -> T:
        """Remove and return the highest-priority (or oldest) item.

        Raises ``IndexError`` if the queue is empty.
        """
        with self._lock:
            entry = heapq.heappop(self._heap)
            return self._extract_item(entry)

    def peek(self) -> T:
        """Return (without removing) the highest-priority item.

        Raises ``IndexError`` if the queue is empty.
        """
        with self._lock:
            return self._extract_item(self._heap[0])

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._heap.clear()
            self._seq = 0
            self._front_seq = -(1 << 30)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __contains__(self, item: object) -> bool:
        """Linear scan — only used for test assertions, not hot path."""
        return any(self._extract_item(e) is item for e in self._heap)

    def __getitem__(self, index: int) -> T:
        """Support ``queue[0]`` for peek-style access in tests.

        NOTE: This returns the item at heap position *index*, which is NOT
        guaranteed to be sorted order beyond index 0.  Only ``queue[0]`` is
        meaningful (same as ``peek()``).  Provided for backward-compat with
        test code that used ``deque[0]``.
        """
        return self._extract_item(self._heap[index])


def make_waiting_queue(policy) -> RequestPriorityQueue:
    """Factory: create a RequestPriorityQueue configured for the given policy.

    Accepts either a ``SchedulingPolicy`` enum value or a string.
    """
    # Import here to avoid circular imports at module level.
    from .scheduler import SchedulingPolicy

    if isinstance(policy, str):
        policy = SchedulingPolicy[policy]

    if policy == SchedulingPolicy.PRIORITY:
        return RequestPriorityQueue(mode=_QueueMode.PRIORITY)
    else:
        return RequestPriorityQueue(mode=_QueueMode.FCFS)
