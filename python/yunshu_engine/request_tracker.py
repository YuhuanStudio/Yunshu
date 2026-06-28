from __future__ import annotations

"""Active generation request tracking and cancellation.

Maintains a registry of in-progress generations that can be cancelled
via the /v1/cancel endpoint. Tracks request_id → cancellation Event mapping.
"""

import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# set by the auth middleware per request; register() uses it as the default
# owner so /v1/cancel can enforce per-request ownership (the previous guard was dead —
# get_owner didn't exist). Best-effort: None where context doesn't propagate → the
# cancel guard stays permissive there (admin can always cancel), never a regression.
current_actor: ContextVar[str | None] = ContextVar("yunshu_current_actor", default=None)


@dataclass
class ActiveGeneration:
    """Tracks a single in-progress generation."""
    request_id: str
    model: str
    created_at: float
    cancel_event: threading.Event
    owner: str | None = None  # actor/key that started it (for per-request cancel auth)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.created_at


class RequestTracker:
    """Registry for tracking and cancelling active generations."""

    def __init__(self):
        self._active: dict[str, ActiveGeneration] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str, model: str = "", owner: str | None = None) -> ActiveGeneration:
        """Register a new generation request. Returns ActiveGeneration with cancel_event."""
        gen = ActiveGeneration(
            request_id=request_id,
            model=model,
            created_at=time.monotonic(),
            cancel_event=threading.Event(),
            owner=owner if owner is not None else current_actor.get(),
        )
        with self._lock:
            self._active[request_id] = gen
        return gen

    def get_owner(self, request_id: str) -> str | None:
        """Return the actor/key that started a request (for per-request cancel auth).

        the /v1/cancel ownership guard probed `hasattr(tracker, "get_owner")`,
        which was always False (this method didn't exist) → any user could cancel any
        other user's request by id. Defining it (and threading `owner` through register)
        makes the guard live.
        """
        with self._lock:
            gen = self._active.get(request_id)
        return gen.owner if gen is not None else None

    def unregister(self, request_id: str) -> None:
        """Remove a completed generation from the registry."""
        with self._lock:
            self._active.pop(request_id, None)

    def cancel(self, request_id: str) -> bool:
        """Signal cancellation for a specific request. Returns True if found."""
        with self._lock:
            gen = self._active.get(request_id)
        if gen is None:
            return False
        gen.cancel_event.set()
        logger.info(f"Generation cancelled: {request_id}")
        return True

    def cancel_all(self) -> int:
        """Cancel all active generations. Returns count of cancelled requests."""
        with self._lock:
            gens = list(self._active.values())
        count = 0
        for gen in gens:
            gen.cancel_event.set()
            count += 1
        logger.info(f"Cancelled all generations: {count}")
        return count

    def is_cancelled(self, request_id: str) -> bool:
        """Check if a request has been signalled for cancellation."""
        with self._lock:
            gen = self._active.get(request_id)
        if gen is None:
            return False
        return gen.cancel_event.is_set()

    def list_active(self, include_internal: bool = False) -> list[dict]:
        """List active generations with metadata.

        By default, filters out internal engine-level registrations
        (request IDs prefixed "stream-") so the public
        /v1/active-generations endpoint shows only the user-facing
        request IDs (one per request). Set include_internal=True for
        debug / shutdown drain views that need every in-flight handle.
        """
        with self._lock:
            gens = list(self._active.values())
        return [
            {
                "request_id": gen.request_id,
                "model": gen.model,
                "elapsed_s": round(gen.elapsed_s, 2),
            }
            for gen in gens
            if include_internal or not gen.request_id.startswith("stream-")
        ]

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)


_tracker: RequestTracker | None = None
_tracker_lock = threading.Lock()


def get_request_tracker() -> RequestTracker:
    """Get the global RequestTracker singleton (thread-safe)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = RequestTracker()
    return _tracker
