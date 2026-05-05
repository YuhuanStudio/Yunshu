"""Yunshu Control Plane — Request queue management and priority scheduling hooks.

Provides a control-plane-level view of the request queue with:
- Request priority overrides (boost/demote)
- Queue inspection API support (pending, active, completed stats)
- Backpressure signal (queue full detection)
- Integration hooks with yunshu_engine.scheduler.Scheduler

Phase 1: In-process queue observer + priority overrides.
Phase 2: Distributed queue across mesh nodes.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Priority levels for request scheduling."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


class QueueAction(Enum):
    """Actions that can be taken on a queued request."""
    ACCEPT = auto()
    REJECT_QUEUE_FULL = auto()
    REJECT_RATE_LIMITED = auto()
    REJECT_UNAUTHORIZED = auto()
    PRIORITY_BOOST = auto()
    PRIORITY_DEMOTE = auto()
    TIMEOUT = auto()
    CANCEL = auto()


@dataclass
class QueueEntry:
    """A tracked entry in the request queue."""
    request_id: str
    tenant_id: str = ""
    model_id: str = ""
    priority: int = QueuePriority.NORMAL.value
    slo_class: str = "standard"
    arrival_time: float = field(default_factory=time.monotonic)
    scheduled_time: Optional[float] = None
    completion_time: Optional[float] = None
    action: QueueAction = QueueAction.ACCEPT
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def wait_time_ms(self) -> float:
        """Time spent waiting in queue (ms)."""
        end = self.scheduled_time or time.monotonic()
        return (end - self.arrival_time) * 1000

    @property
    def total_time_ms(self) -> float:
        """Total time from arrival to completion (ms)."""
        if self.completion_time is None:
            return (time.monotonic() - self.arrival_time) * 1000
        return (self.completion_time - self.arrival_time) * 1000

    @property
    def status(self) -> str:
        if self.completion_time is not None:
            return "completed"
        if self.scheduled_time is not None:
            return "running"
        return "waiting"


@dataclass
class QueueStats:
    """Snapshot of queue state."""
    waiting: int = 0
    running: int = 0
    completed: int = 0
    rejected: int = 0
    cancelled: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


class RequestQueueManager:
    """Manages request queue at the control plane level.

    Provides:
    1. Queue inspection for admin API
    2. Priority overrides (boost/demote specific requests)
    3. Backpressure signaling
    4. Per-SLO-class fairness tracking
    5. Integration with yunshu_engine.scheduler.Scheduler

    Thread-safe for concurrent access from ASGI handlers and engine thread.
    """

    def __init__(self, max_queue_size: int = 1024) -> None:
        self._max_queue_size = max_queue_size
        self._entries: dict[str, QueueEntry] = {}
        self._priority_overrides: dict[str, int] = {}
        self._lock = threading.Lock()
        self._completed_count = 0
        self._rejected_count = 0
        self._cancelled_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._wait_times: list[float] = []

    def on_request_arrival(
        self,
        request_id: str,
        tenant_id: str = "",
        model_id: str = "",
        priority: int = QueuePriority.NORMAL.value,
        slo_class: str = "standard",
    ) -> QueueAction:
        """Called when a new request arrives. Returns the action to take.

        Checks queue capacity and applies priority overrides.
        """
        with self._lock:
            if len(self._entries) >= self._max_queue_size:
                self._rejected_count += 1
                return QueueAction.REJECT_QUEUE_FULL

            # Apply priority override if one was set
            effective_priority = self._priority_overrides.pop(request_id, priority)

            entry = QueueEntry(
                request_id=request_id,
                tenant_id=tenant_id,
                model_id=model_id,
                priority=effective_priority,
                slo_class=slo_class,
            )
            self._entries[request_id] = entry
            return QueueAction.ACCEPT

    def on_request_scheduled(self, request_id: str) -> None:
        """Called when a request is picked up by the scheduler."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                entry.scheduled_time = time.monotonic()
                self._wait_times.append(entry.wait_time_ms)
                # Keep only last 1000 wait times for rolling average
                if len(self._wait_times) > 1000:
                    self._wait_times = self._wait_times[-1000:]

    def on_request_completed(
        self,
        request_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Called when a request completes (success or failure)."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                entry.completion_time = time.monotonic()
                entry.prompt_tokens = prompt_tokens
                entry.completion_tokens = completion_tokens
                entry.error = error

                self._completed_count += 1
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += completion_tokens

    def on_request_cancelled(self, request_id: str) -> None:
        """Called when a request is cancelled."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry is not None:
                entry.action = QueueAction.CANCEL
                entry.completion_time = time.monotonic()
                self._cancelled_count += 1

    def set_priority_override(self, request_id: str, priority: int) -> None:
        """Set a priority override for a specific request.

        Applied on next on_request_arrival() call for this request_id.
        """
        with self._lock:
            self._priority_overrides[request_id] = priority

    def get_scheduling_priority(self, request_id: str, default: int = QueuePriority.NORMAL.value) -> int:
        """Get effective priority for a request, including any override.

        Returns the default priority if no override is set.
        The override is consumed (removed) after reading.
        """
        with self._lock:
            return self._priority_overrides.pop(request_id, default)

    def get_queue_stats(self) -> QueueStats:
        """Get a snapshot of queue state."""
        with self._lock:
            waiting = 0
            running = 0
            max_wait = 0.0

            for entry in self._entries.values():
                status = entry.status
                if status == "waiting":
                    waiting += 1
                    max_wait = max(max_wait, entry.wait_time_ms)
                elif status == "running":
                    running += 1

            avg_wait = 0.0
            if self._wait_times:
                avg_wait = sum(self._wait_times) / len(self._wait_times)
                max_wait = max(max_wait, max(self._wait_times))

            return QueueStats(
                waiting=waiting,
                running=running,
                completed=self._completed_count,
                rejected=self._rejected_count,
                cancelled=self._cancelled_count,
                avg_wait_time_ms=round(avg_wait, 2),
                max_wait_time_ms=round(max_wait, 2),
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
            )

    def get_pending_requests(self, limit: int = 100) -> list[dict]:
        """Get pending (waiting) requests for admin inspection."""
        with self._lock:
            pending = []
            for entry in self._entries.values():
                if entry.status == "waiting":
                    pending.append({
                        "request_id": entry.request_id,
                        "tenant_id": entry.tenant_id,
                        "model_id": entry.model_id,
                        "priority": entry.priority,
                        "slo_class": entry.slo_class,
                        "wait_time_ms": round(entry.wait_time_ms, 2),
                    })
            pending.sort(key=lambda x: x["priority"], reverse=True)
            return pending[:limit]

    def get_active_requests(self, limit: int = 100) -> list[dict]:
        """Get active (running) requests for admin inspection."""
        with self._lock:
            active = []
            for entry in self._entries.values():
                if entry.status == "running":
                    active.append({
                        "request_id": entry.request_id,
                        "tenant_id": entry.tenant_id,
                        "model_id": entry.model_id,
                        "priority": entry.priority,
                        "slo_class": entry.slo_class,
                        "total_time_ms": round(entry.total_time_ms, 2),
                    })
            return active[:limit]

    def is_queue_full(self) -> bool:
        """Check if the queue has reached capacity."""
        with self._lock:
            return len(self._entries) >= self._max_queue_size

    def clear_completed(self) -> int:
        """Remove completed entries to free memory. Returns count removed."""
        with self._lock:
            to_remove = [
                rid for rid, entry in self._entries.items()
                if entry.status == "completed"
            ]
            for rid in to_remove:
                del self._entries[rid]
            return len(to_remove)


# Module-level singleton
_queue_manager: RequestQueueManager | None = None


def get_request_queue_manager() -> RequestQueueManager:
    """Get the global RequestQueueManager singleton."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = RequestQueueManager()
    return _queue_manager


def set_request_queue_manager(manager: RequestQueueManager) -> None:
    """Set the global RequestQueueManager (for testing)."""
    global _queue_manager
    _queue_manager = manager
