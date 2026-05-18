"""KV Cache event system for distributed cache coherency.

Enables disaggregated prefill/decode nodes to maintain cache coherency
without full KV transfers. Events are published on block cache/evict/free
operations and can be relayed to peer nodes via the mesh layer.

Event types:
  - ``block_cached``: A block was added to the prefix cache.
  - ``block_evicted``: A block was removed from the prefix cache.
  - ``request_freed``: A request's blocks were released.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class CacheEvent:
    """A single cache coherency event.

    Attributes:
        event_type: Category of the event (e.g. ``"block_cached"``).
        block_hash: Hash of the affected block (prefix cache key).
        block_ids: Physical block IDs involved.
        node_id: Origin node or request identifier.
    """

    event_type: str
    block_hash: int | None = None
    block_ids: list[int] = field(default_factory=list)
    node_id: str | None = None


class CacheEventBus:
    """Thread-safe publish/subscribe event bus for cache coherency.

    Subscribers register callbacks for specific event types.  When a
    :class:`CacheEvent` is published, all matching callbacks are invoked
    synchronously.  Exceptions in callbacks are logged and swallowed so
    that a faulty subscriber never disrupts the publisher.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[CacheEvent], None]]] = {}
        self._lock = threading.Lock()

    # -- Subscribe / Unsubscribe ------------------------------------------

    def subscribe(self, event_type: str, callback: Callable[[CacheEvent], None]) -> None:
        """Register *callback* for events of *event_type*."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[CacheEvent], None]) -> None:
        """Remove *callback* from the subscriber list for *event_type*."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb is not callback
                ]

    # -- Publish -----------------------------------------------------------

    def publish(self, event: CacheEvent) -> None:
        """Deliver *event* to all subscribers registered for its type.

        The subscriber list is snapshot under the lock; callbacks run
        outside the lock to prevent deadlocks if a callback itself
        subscribes/unsubscribes.
        """
        with self._lock:
            callbacks = list(self._subscribers.get(event.event_type, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                logger.warning(
                    "CacheEventBus subscriber %r raised on %s",
                    cb,
                    event.event_type,
                    exc_info=True,
                )

    # -- Introspection -----------------------------------------------------

    def subscriber_count(self, event_type: str) -> int:
        """Return number of subscribers for *event_type*."""
        with self._lock:
            return len(self._subscribers.get(event_type, []))
