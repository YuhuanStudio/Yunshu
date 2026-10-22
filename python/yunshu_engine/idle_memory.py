"""Return memory the VLM runner no longer needs once it goes idle.

Two things stay behind after a request ends:

- arrays that only a reference cycle still holds (the finished request's generator,
  closures and caches) wait for Python's cyclic collector, which may not run for a long
  time on an idle server; one collection runs ``COLLECT_AFTER_S`` after the runner drains
  (long enough that back-to-back requests never pay for it);
- MLX's freed-buffer pool (bounded, kept on purpose so the next request's cache restore
  and prefill reuse buffers) is released after ``IDLE_TRIM_S`` without a request.

Until that trim the weights also stay wired: upstream's BatchGenerator raises the wired
limit while it lives and drops it on close, and after ~2 s without a command buffer Metal
stops keeping the weights resident, so the next request's generator re-pins all of them
(~200 ms for the 27B, measured 2026-10-22: idle follow-up TTFT 0.23 s within 2 s, 0.44-0.66 s
from 4 s on). Each heartbeat (every ``HEARTBEAT_S``) holds the limit and runs a tiny kernel
that touches no model array, so outputs are unchanged (~1 ms per second of idle); the trim
gives the limit back.

The timer thread only submits work; every MLX call runs on the serialized MLX executor,
and each stage re-checks that the runner is still idle.
"""

from __future__ import annotations

import gc
import logging
import math
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

COLLECT_AFTER_S = 2.0
IDLE_TRIM_S = 30.0
HEARTBEAT_S = 1.0


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
        heartbeat_s: float = HEARTBEAT_S,
        beat: Callable[[], None] | None = None,
        release: Callable[[], None] | None = None,
    ) -> None:
        self._submit = submit
        self._busy = busy
        self._stages: list[tuple[float, Callable[[], None], str]] = [
            (collect_after_s, collect or _collect, "collections"),
            (max(0.0, trim_after_s - collect_after_s), trim or _trim, "trims"),
        ]
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._heartbeat_s = heartbeat_s
        # beats strictly before the trim, which ends the window
        self._beats_per_idle = (
            0 if heartbeat_s <= 0 else max(0, math.ceil(trim_after_s / heartbeat_s) - 1)
        )
        self._beat = beat or _beat
        self._release = release or _release_wired
        self._holding = False
        self._beat_timer: threading.Timer | None = None
        self.collections = 0
        self.trims = 0
        self.beats = 0

    def drained(self) -> None:
        """The runner has no work left: start the idle schedule."""
        self._arm(0)
        self._arm_beat(self._beats_per_idle)

    def activity(self) -> None:
        """A request arrived: cancel what is pending (the pool stays)."""
        with self._lock:
            for timer in (self._timer, self._beat_timer):
                if timer is not None:
                    timer.cancel()
            self._timer = self._beat_timer = None

    def _arm_beat(self, left: int) -> None:
        if left <= 0:
            return
        timer = threading.Timer(self._heartbeat_s, self._fire_beat, args=(left,))
        timer.daemon = True
        with self._lock:
            if self._beat_timer is not None:
                self._beat_timer.cancel()
            self._beat_timer = timer
        timer.start()

    def _fire_beat(self, left: int) -> None:
        with self._lock:
            if self._beat_timer is not threading.current_thread():
                return
            self._beat_timer = None
        try:
            self._submit(lambda: self._run_beat(left))
        except Exception:  # noqa: BLE001
            logger.debug("idle heartbeat could not be scheduled", exc_info=True)

    def _run_beat(self, left: int) -> None:
        if self._busy():
            return
        try:
            self._holding = True
            self._beat()
            self.beats += 1
        except Exception:  # noqa: BLE001
            logger.debug("idle heartbeat failed", exc_info=True)
            return
        self._arm_beat(left - 1)

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
        if stage == len(self._stages) - 1 and self._holding:
            with self._lock:
                if self._beat_timer is not None:
                    self._beat_timer.cancel()
                    self._beat_timer = None
            self._holding = False
            try:
                self._release()
            except Exception:  # noqa: BLE001
                logger.debug("idle wired-limit release failed", exc_info=True)
        self._arm(stage + 1)


def _gib() -> tuple[float, float]:
    import mlx.core as mx

    return mx.get_active_memory() / 2**30, mx.get_cache_memory() / 2**30


_BEAT_OPERAND = None
_WIRED_BEFORE: int | None = None


def _beat() -> None:
    import mlx.core as mx

    global _BEAT_OPERAND, _WIRED_BEFORE
    if _WIRED_BEFORE is None:
        _WIRED_BEFORE = mx.set_wired_limit(
            int(mx.device_info()["max_recommended_working_set_size"])
        )
        logger.info("idle memory: holding the wired limit until the idle trim")
    if _BEAT_OPERAND is None:
        _BEAT_OPERAND = mx.ones((256, 256))
    mx.eval(_BEAT_OPERAND @ _BEAT_OPERAND)
    mx.synchronize()


def _release_wired() -> None:
    import mlx.core as mx

    global _WIRED_BEFORE
    if _WIRED_BEFORE is not None:
        mx.set_wired_limit(_WIRED_BEFORE)
        _WIRED_BEFORE = None
        logger.info("idle memory: released the wired limit")


def _collect() -> None:
    from .mlx_executor import synchronize_streams

    synchronize_streams()
    before = _gib()
    gc.collect()
    after = _gib()
    logger.info(
        "idle memory: collected cycle garbage, active %.2f -> %.2f GiB",
        before[0],
        after[0],
    )


def _trim() -> None:
    from .mlx_executor import sync_and_clear_cache

    gc.collect()
    before = _gib()
    sync_and_clear_cache()
    after = _gib()
    logger.info(
        "idle memory: released the allocator pool, cache %.2f -> %.2f GiB",
        before[1],
        after[1],
    )
