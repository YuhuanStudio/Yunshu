from __future__ import annotations
"""Active generation request tracking and cancellation.

Maintains a registry of in-progress generations that can be cancelled
via the /v1/cancel endpoint. Tracks request_id → cancellation Event mapping.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ActiveGeneration:
    """Tracks a single in-progress generation."""
    request_id: str
    model: str
    created_at: float
    cancel_event: asyncio.Event

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.created_at


class RequestTracker:
    """Registry for tracking and cancelling active generations."""

    def __init__(self):
        self._active: dict[str, ActiveGeneration] = {}

    def register(self, request_id: str, model: str = "") -> ActiveGeneration:
        """Register a new generation request. Returns ActiveGeneration with cancel_event."""
        gen = ActiveGeneration(
            request_id=request_id,
            model=model,
            created_at=time.time(),
            cancel_event=asyncio.Event(),
        )
        self._active[request_id] = gen
        return gen

    def unregister(self, request_id: str) -> None:
        """Remove a completed generation from the registry."""
        self._active.pop(request_id, None)

    def cancel(self, request_id: str) -> bool:
        """Signal cancellation for a specific request. Returns True if found."""
        gen = self._active.get(request_id)
        if gen is None:
            return False
        gen.cancel_event.set()
        logger.info(f"Generation cancelled: {request_id}")
        return True

    def cancel_all(self) -> int:
        """Cancel all active generations. Returns count of cancelled requests."""
        count = 0
        for gen in self._active.values():
            gen.cancel_event.set()
            count += 1
        logger.info(f"Cancelled all generations: {count}")
        return count

    def is_cancelled(self, request_id: str) -> bool:
        """Check if a request has been signalled for cancellation."""
        gen = self._active.get(request_id)
        if gen is None:
            return False
        return gen.cancel_event.is_set()

    def list_active(self) -> list[dict]:
        """List all active generations with metadata."""
        return [
            {
                "request_id": gen.request_id,
                "model": gen.model,
                "elapsed_s": round(gen.elapsed_s, 2),
            }
            for gen in self._active.values()
        ]

    @property
    def active_count(self) -> int:
        return len(self._active)


_tracker: Optional[RequestTracker] = None
_tracker_lock = threading.Lock()


def get_request_tracker() -> RequestTracker:
    """Get the global RequestTracker singleton (thread-safe)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = RequestTracker()
    return _tracker
