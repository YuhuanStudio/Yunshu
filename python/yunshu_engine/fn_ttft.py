"""Host-side stage timers of one cold Flash-Next request (perf_counter only: no added eval or sync).

One INFO line per request when its first token is ready, the same format on the round driver and on
the older speculative lane::

    fn_ttft path=<driver|lane> ids=N hit=H segs=K | submit=a wait=b ckpt=c capture=d head=e draw=f emit=g
        tail=h other=i total=t ms

``submit`` builds / queues the graphs (host), ``wait`` is the time of the eval that returns (where GPU
time shows), ``ckpt`` APC checkpoint stores, ``capture`` the drafter's feature absorb, ``head`` the MTP
head's prompt prefill graph, ``draw`` the first-token sampling graph, ``emit`` event creation, ``tail``
the end-of-step eval of the remaining caches. ``other`` = total minus the named stages.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("yunshu_engine.fn_ttft")

STAGES = ("submit", "wait", "ckpt", "capture", "head", "draw", "emit", "tail")


class Timer:
    """Accumulates stage seconds; ``mark(stage)`` charges the time since the previous mark."""

    def __init__(self) -> None:
        self.t0 = self.last = time.perf_counter()
        self.stage: dict[str, float] = dict.fromkeys(STAGES, 0.0)
        self.segs = 0

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.stage[stage] += now - self.last
        self.last = now

    def merge(self, other: Timer) -> None:
        for k, v in other.stage.items():
            self.stage[k] += v
        self.segs += other.segs
        self.t0 = min(self.t0, other.t0)
        self.last = max(self.last, other.last)

    def line(self, path: str, ids: int, hit: int, total_s: float | None = None) -> str:
        total = (self.last - self.t0) if total_s is None else total_s
        named = sum(self.stage.values())
        parts = " ".join(f"{k}={self.stage[k] * 1e3:.1f}" for k in STAGES)
        return (
            f"fn_ttft path={path} ids={ids} hit={hit} segs={self.segs} | {parts} "
            f"other={(total - named) * 1e3:.1f} total={total * 1e3:.1f} ms"
        )

    def log(self, path: str, ids: int, hit: int, total_s: float | None = None) -> None:
        logger.info(self.line(path, ids, hit, total_s))
