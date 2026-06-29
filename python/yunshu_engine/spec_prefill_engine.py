from __future__ import annotations

"""Yunshu SpecPrefillEngine — speculative prefill during decode idle time.

While the model generates tokens for request A, uses idle GPU capacity to
prefill request B.  This hides prefill latency behind ongoing decode work,
reducing time-to-first-token (TTFT) for queued requests.

Architecture:
  EngineCore._engine_loop()
    → SpecPrefillEngine.try_prefill(available_budget)
      → selects highest-priority waiting request
      → preempts into prefill if budget allows
      → stores partially-prefilled KV state for later resume

Design:
  - Priority queue (higher priority = prefill first)
  - Budget-aware: respects remaining GPU memory / step budget
  - Cancellation support for aborted requests
  - Stats tracking for utilisation analysis
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PrefillEntry:
    """A pending or in-progress prefill request."""

    request_id: str
    tokens: list[int]
    priority: float = 0.0
    enqueued_at: float = field(default_factory=time.monotonic)
    tokens_prefilled: int = 0
    kv_state: Any | None = None  # partial KV state for resume
    status: str = "pending"  # pending | in_progress | completed | cancelled
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class SpecPrefillStats:
    """Accumulated statistics for SpecPrefillEngine."""

    prefills_completed: int = 0
    prefills_cancelled: int = 0
    tokens_prefilled: int = 0
    budget_requested: int = 0
    budget_consumed: int = 0
    total_prefill_time_ms: float = 0.0


class SpecPrefillEngine:
    """Speculative prefill engine — hides prefill latency behind decode.

    Usage:
        engine = SpecPrefillEngine(max_budget=4096)
        engine.enqueue("req-B", tokens=[1,2,3,...], priority=1.0)
        ...
        # inside engine loop, after decode step:
        engine.try_prefill(available_budget=2048)
    """

    def __init__(
        self,
        max_budget: int = 8192,
        max_queue_size: int = 64,
        prefill_chunk_size: int = 512,
    ) -> None:
        self._max_budget = max_budget
        self._max_queue_size = max_queue_size
        self._prefill_chunk_size = prefill_chunk_size

        # Priority queue: sorted by priority descending
        self._queue: list[PrefillEntry] = []
        # Fast lookup by request_id
        self._entries: dict[str, PrefillEntry] = {}
        # Currently in-progress entry (at most one at a time)
        self._current: PrefillEntry | None = None

        self._stats = SpecPrefillStats()
        self._start_time = time.monotonic()

    # ── Public API ──

    def enqueue(
        self,
        request_id: str,
        tokens: list[int],
        priority: float = 0.0,
    ) -> bool:
        """Add a request to the speculative prefill queue.

        Args:
            request_id: Unique identifier for the request.
            tokens: Token IDs to prefill.
            priority: Higher priority = prefilled first (default 0).

        Returns:
            True if enqueued, False if queue is full or duplicate.
        """
        if request_id in self._entries:
            return False
        if len(self._queue) >= self._max_queue_size:
            return False

        entry = PrefillEntry(
            request_id=request_id,
            tokens=tokens,
            priority=priority,
        )
        self._entries[request_id] = entry
        self._queue.append(entry)
        # Keep queue sorted by priority (descending), then by enqueue time (FIFO)
        self._queue.sort(key=lambda e: (-e.priority, e.enqueued_at))
        logger.debug(
            f"SpecPrefill: enqueued {request_id} "
            f"({len(tokens)} tokens, priority={priority}, queue_size={len(self._queue)})"
        )
        return True

    def try_prefill(self, available_budget: int) -> PrefillEntry | None:
        """Attempt to prefill the highest-priority waiting request.

        Called from the engine loop after a decode step. If there is
        spare budget (tokens that could be processed without delaying
        the decode batch), start or resume a prefill.

        Args:
            available_budget: Number of tokens of spare GPU budget.

        Returns:
            The PrefillEntry that was prefilled (completed), or None if
            no prefill was possible.
        """
        self._stats.budget_requested += available_budget
        if available_budget <= 0:
            return None

        # If currently in progress, try to continue it
        if self._current is not None and self._current.status == "in_progress":
            entry = self._current
            remaining = len(entry.tokens) - entry.tokens_prefilled
            chunk = min(remaining, available_budget, self._prefill_chunk_size)
            if chunk > 0:
                self._stats.budget_consumed += chunk
                entry.tokens_prefilled += chunk
                if entry.tokens_prefilled >= len(entry.tokens):
                    return self._complete_entry(entry)
            return None

        # Pick the highest-priority pending entry
        entry = self._pick_next_pending()
        if entry is None:
            return None

        # Budget check: can we make progress?
        chunk = min(len(entry.tokens), available_budget, self._prefill_chunk_size)
        if chunk <= 0:
            return None

        # Start prefill
        entry.status = "in_progress"
        entry.started_at = time.monotonic()
        self._current = entry
        self._stats.budget_consumed += chunk
        entry.tokens_prefilled = chunk

        if entry.tokens_prefilled >= len(entry.tokens):
            return self._complete_entry(entry)

        return None

    def cancel_prefill(self, request_id: str) -> bool:
        """Cancel a pending or in-progress prefill.

        Args:
            request_id: The request to cancel.

        Returns:
            True if the entry was found and cancelled, False otherwise.
        """
        entry = self._entries.get(request_id)
        if entry is None:
            return False
        if entry.status in ("completed", "cancelled"):
            return False

        entry.status = "cancelled"
        entry.completed_at = time.monotonic()
        self._stats.prefills_cancelled += 1

        # Remove from queue and entries (allows re-enqueue with same ID)
        self._queue = [e for e in self._queue if e.request_id != request_id]
        self._entries.pop(request_id, None)

        # Clear current if it's this entry
        if self._current is not None and self._current.request_id == request_id:
            self._current = None

        logger.debug(f"SpecPrefill: cancelled {request_id}")
        return True

    def get_entry(self, request_id: str) -> PrefillEntry | None:
        """Retrieve a prefill entry by request ID."""
        return self._entries.get(request_id)

    def remove_entry(self, request_id: str) -> PrefillEntry | None:
        """Remove and return an entry after it has been consumed."""
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            self._queue = [e for e in self._queue if e.request_id != request_id]
            if self._current is not None and self._current.request_id == request_id:
                self._current = None
        return entry

    def peek_next(self) -> PrefillEntry | None:
        """Return the next pending entry without starting it."""
        return self._pick_next_pending()

    def queue_size(self) -> int:
        """Number of entries still in the queue (pending + in_progress)."""
        return len([e for e in self._queue if e.status in ("pending", "in_progress")])

    def has_capacity(self) -> bool:
        """Whether the queue can accept more entries."""
        return len(self._queue) < self._max_queue_size

    @property
    def current(self) -> PrefillEntry | None:
        """The currently in-progress prefill entry, if any."""
        return self._current

    def get_stats(self) -> dict:
        """Return engine statistics."""
        elapsed = time.monotonic() - self._start_time
        budget_util = (
            self._stats.budget_consumed / self._stats.budget_requested
            if self._stats.budget_requested > 0
            else 0.0
        )
        avg_ttft_ms = (
            self._stats.total_prefill_time_ms / self._stats.prefills_completed
            if self._stats.prefills_completed > 0
            else 0.0
        )
        return {
            "prefills_completed": self._stats.prefills_completed,
            "prefills_cancelled": self._stats.prefills_cancelled,
            "tokens_prefilled": self._stats.tokens_prefilled,
            "budget_utilization": round(budget_util, 4),
            "budget_requested": self._stats.budget_requested,
            "budget_consumed": self._stats.budget_consumed,
            "queue_size": self.queue_size(),
            "max_queue_size": self._max_queue_size,
            "avg_prefill_time_ms": round(avg_ttft_ms, 2),
            "uptime_seconds": round(elapsed, 1),
            "current_request": (self._current.request_id if self._current else None),
        }

    # ── Internal ──

    def _pick_next_pending(self) -> PrefillEntry | None:
        """Select the highest-priority pending entry."""
        for entry in self._queue:
            if entry.status == "pending":
                return entry
        return None

    def _complete_entry(self, entry: PrefillEntry) -> PrefillEntry:
        """Mark entry as completed and update stats."""
        entry.status = "completed"
        entry.completed_at = time.monotonic()
        if entry.started_at is not None:
            duration_ms = (entry.completed_at - entry.started_at) * 1000
            self._stats.total_prefill_time_ms += duration_ms

        self._stats.prefills_completed += 1
        self._stats.tokens_prefilled += entry.tokens_prefilled

        # Remove from queue
        self._queue = [e for e in self._queue if e.request_id != entry.request_id]

        # Clear current if it's this entry
        if self._current is not None and self._current.request_id == entry.request_id:
            self._current = None

        logger.debug(
            f"SpecPrefill: completed {entry.request_id} "
            f"({entry.tokens_prefilled} tokens)"
        )
        return entry
