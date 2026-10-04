"""Return memory the VLM runner no longer needs once it goes idle.

Two things stay behind after a request ends:

- arrays that only a reference cycle still holds (the finished request's generator,
  closures and caches) wait for Python's cyclic collector, which may not run for a long
  time on an idle server; one collection runs ``COLLECT_AFTER_S`` after the runner drains
  (long enough that back-to-back requests never pay for it);
- MLX's freed-buffer pool (bounded, kept on purpose so the next request's cache restore
  and prefill reuse buffers) is released after ``IDLE_TRIM_S`` without a request.

The timer thread only submits work; every MLX call runs on the serialized MLX executor,
and each stage re-checks that the runner is still idle.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

COLLECT_AFTER_S = 2.0
IDLE_TRIM_S = 30.0


class IdleMemory:
    def __init__(
        self,
        submit: Callable[[Callable[[], None]], object],
        busy: Callable[[], bool],
        *,
        collect_after_s: float = COLLECT_AFTER_S,
        trim_after_s: float = IDLE_TRIM_S,
        collect: Callable[[], None] | None = None,
        trim: Callable[[], None] | None = None,
    ) -> None:
        self._submit = submit
        self._busy = busy
        self._stages: list[tuple[float, Callable[[], None], str]] = [
            (collect_after_s, collect or _collect, "collections"),
            (max(0.0, trim_after_s - collect_after_s), trim or _trim, "trims"),
        ]
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self.collections = 0
        self.trims = 0

    def drained(self) -> None:
        """The runner has no work left: start the idle schedule."""
        self._arm(0)

    def activity(self) -> None:
        """A request arrived: cancel what is pending (the pool stays)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _arm(self, stage: int) -> None:
        if stage >= len(self._stages):
            return
        timer = threading.Timer(self._stages[stage][0], self._fire, args=(stage,))
        timer.daemon = True
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = timer
        timer.start()

    def _fire(self, stage: int) -> None:
        with self._lock:
            if self._timer is not threading.current_thread():
                return
            self._timer = None
        try:
            self._submit(lambda: self._run(stage))
        except Exception:  # noqa: BLE001
            logger.debug("idle memory stage could not be scheduled", exc_info=True)

    def _run(self, stage: int) -> None:
        if self._busy():
            return
        _, action, counter = self._stages[stage]
        try:
            action()
            setattr(self, counter, getattr(self, counter) + 1)
        except Exception:  # noqa: BLE001
            logger.debug("idle memory stage %d failed", stage, exc_info=True)
        self._arm(stage + 1)


def _collect() -> None:
    from .mlx_executor import synchronize_streams

    synchronize_streams()
    gc.collect()


def _trim() -> None:
    from .mlx_executor import sync_and_clear_cache

    gc.collect()
    sync_and_clear_cache()
