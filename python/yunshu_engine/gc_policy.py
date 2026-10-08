"""Keep Python's cyclic collector out of the time-to-first-token path.

A generation-2 collection walks every tracked container in the process. On a loaded
27B server that is ~110 ms, and it can land right after a 32K prefill while the copy
drafter builds its n-gram index (tens of thousands of fresh tuples trip the
collector's thresholds), delaying the first token. Three measures, all invisible to
the model (no tensor, sampler or cache value changes):

- ``freeze_startup_heap``: after the model is loaded and warmed, collect once and
  ``gc.freeze()`` so the long-lived startup heap (modules, tokenizer, model objects)
  is never walked again. vLLM (``freeze_gc_heap`` in the engine core) and SGLang
  (``freeze_gc``) do the same after init.
- a higher generation-2 multiplier, so full collections stop being triggered by
  transient allocation bursts. Idle servers still collect (``idle_memory``).
- ``TtftGcGuard``: from request arrival until its first token the collector is
  disabled. Memory stays bounded: the hold ends at the first token, at
  ``max_hold_s``, or when more than ``max_allocs`` net container allocations piled
  up, whichever is first, and then the ordinary thresholds apply again (the next
  collection is cheap: young generations only, generation 2 is rare and small).

Policies (``YUNSHU_GC_POLICY``): ``default`` changes nothing; ``freeze`` freezes only;
``defer`` is freeze + threshold + guard.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable, Hashable
from typing import Any

logger = logging.getLogger(__name__)

POLICIES = ("default", "freeze", "defer")
GEN2_MULTIPLIER = 1000
MAX_HOLD_S = 60.0
MAX_ALLOCS = 5_000_000
_POLL_S = 0.5


def freeze_startup_heap(collector: Any = gc) -> dict[str, int]:
    """Collect once, then move everything alive into the permanent generation."""
    before = collector.get_freeze_count()
    collector.collect()
    collector.freeze()
    return {"frozen_before": before, "frozen": collector.get_freeze_count()}


def unfreeze(collector: Any = gc) -> None:
    """Make frozen objects collectable again (model unload, tests)."""
    collector.unfreeze()


class TtftGcGuard:
    """Refcounted ``gc.disable`` for requests that have not produced a token yet."""

    def __init__(
        self,
        collector: Any = gc,
        *,
        max_hold_s: float = MAX_HOLD_S,
        max_allocs: int = MAX_ALLOCS,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = _POLL_S,
        watchdog: bool = True,
    ) -> None:
        self._gc = collector
        self.max_hold_s = max_hold_s
        self.max_allocs = max_allocs
        self._clock = clock
        self._poll_s = poll_s
        self._use_watchdog = watchdog
        self._lock = threading.Lock()
        self._waiting: set[Hashable] = set()
        self._held = False  # we disabled the collector and must re-enable it
        self._since = 0.0
        self._watch: threading.Thread | None = None
        self.holds = 0
        self.forced_releases = 0

    def begin(self, key: Hashable) -> None:
        with self._lock:
            self._waiting.add(key)
            if not self._held and self._gc.isenabled():
                self._gc.disable()
                self._held = True
                self._since = self._clock()
                self.holds += 1
                self._start_watchdog()

    def first_token(self, key: Hashable) -> None:
        """The request produced output or ended: stop waiting for it."""
        with self._lock:
            self._waiting.discard(key)
            if not self._waiting:
                self._release()

    end = first_token

    def _release(self) -> None:
        if self._held:
            self._held = False
            self._gc.enable()

    def _over_budget(self) -> bool:
        return (
            self._clock() - self._since > self.max_hold_s
            or self._gc.get_count()[0] > self.max_allocs
        )

    def check(self) -> bool:
        """Release a hold that exceeded its bound; True when it did."""
        with self._lock:
            if self._held and self._over_budget():
                self._release()
                self.forced_releases += 1
                logger.warning(
                    "gc guard: hold exceeded its bound; collector re-enabled"
                )
                return True
            return False

    def _start_watchdog(self) -> None:
        if not self._use_watchdog:
            return
        if self._watch is not None and self._watch.is_alive():
            return
        self._watch = threading.Thread(
            target=self._watchdog, name="gc-guard", daemon=True
        )
        self._watch.start()

    def _watchdog(self) -> None:
        while True:
            time.sleep(self._poll_s)
            with self._lock:
                if not self._held:
                    return
            self.check()


def install(policy: str, collector: Any = gc) -> TtftGcGuard | None:
    """Apply the thresholds of ``policy`` (after the startup heap is frozen)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown GC policy {policy!r}")
    if policy == "default":
        return None
    info = freeze_startup_heap(collector)
    logger.info("GC policy %s: froze %d objects", policy, info["frozen"])
    if policy == "freeze":
        return None
    g0, g1, _ = collector.get_threshold()
    collector.set_threshold(g0, g1, GEN2_MULTIPLIER)
    logger.info(
        "GC policy defer: thresholds %s, first-token hold armed",
        (g0, g1, GEN2_MULTIPLIER),
    )
    return TtftGcGuard(collector)
