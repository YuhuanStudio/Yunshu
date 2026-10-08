"""Engine-side timings and prefill progress for the text-only mlx-lm fast path.

The VLM batch runner fills a ``RunStats`` that the gateway reads live (progress comments,
``x_yunshu``). The fast path (``_generate_fast`` / ``_stream_generate_fast``, one mlx-lm
``generate_step`` per request) feeds the same object through this small adapter, so TTFT,
prefill / decode tokens per second and the prefill percentage exist for those requests too.
All methods are cheap and never raise (they run on the GPU thread inside the token loop).
"""

from __future__ import annotations

import contextlib
import time
from typing import Any


class FastPathStats:
    """Fills a ``RunStats`` from a ``generate_step`` loop and attaches it to the request."""

    def __init__(self, cancel_event: Any, prompt_tokens: int) -> None:
        from .vlm_batch_runner import RunStats

        self.stats = RunStats()
        self.stats.prompt_tokens = prompt_tokens
        self.stats.t_submit = time.perf_counter()
        if cancel_event is not None:
            # what RequestTracker.stats reads; slotted event objects keep stats local
            with contextlib.suppress(Exception):
                cancel_event.run_stats = self.stats

    def admit(
        self,
        cached_tokens: int,
        to_prefill: int,
        tier: str | None = None,
        reload_ms: float | None = None,
    ) -> None:
        """The request left the queue and its prefill starts now. ``tier`` is where the
        cached prefix came from ("hot", "warm", "ssd", "none") and ``reload_ms`` the lookup time."""
        st = self.stats
        st.cached_tokens = int(cached_tokens)
        if tier is not None:
            st.cache_tier = tier
            st.cache_reload_ms = reload_ms
        st.prefill_total = int(to_prefill)
        st.prefill_done = 0
        st.t_admit = time.perf_counter()
        st.latency_marks.setdefault("prefill_start", st.t_admit)

    def progress(self, processed: int, total: int) -> None:
        """``generate_step(prompt_progress_callback=...)``: tokens prefilled so far."""
        st = self.stats
        st.prefill_total = max(int(total), 1)
        st.prefill_done = min(int(processed), st.prefill_total)
        if st.prefill_done == st.prefill_total and not st.t_prefill_end:
            st.t_prefill_end = time.perf_counter()

    def token(self, generated: int) -> None:
        """One generated token (``generated`` = count so far)."""
        st = self.stats
        now = time.perf_counter()
        if not st.t_first:
            st.t_first = now
            st.prefill_done = st.prefill_total
        st.generated = generated
        st.t_last = now

    def finish(self, reason: str | None) -> None:
        self.stats.finish_reason = reason or "stop"
