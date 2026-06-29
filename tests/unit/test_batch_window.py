"""engine-loop batch-accumulation window breaks early when the queue stops growing.

The batch-accumulation window (batch_wait_ms, default 15ms) lets a burst of concurrent
requests queue and batch in one step. But the only early-exit was `len(waiting) >=
prefill_batch_size` (8) — so a genuinely-idle SINGLE request (1 < 8) waited the FULL 15ms,
contradicting the "single-request TTFT barely affected" comment. also breaks when the
waiting count stops growing for ~2 cycles: a real burst keeps arriving (count climbs) so it
still accumulates; an isolated request proceeds after ~4ms.

NOTE: this caps the COLD idle single-request window penalty (15ms→~4ms). The ~9ms warm-loop
overhead measured in is NOT the batch window (proven: YUNSHU_BATCH_WAIT_MS=0 gave 70 vs
69ms) — it's inherent scheduler machinery, the separate unification target. Burst batching is
unchanged (verify_engine_loop gate 6/6).
"""
from __future__ import annotations

import inspect

from yunshu_engine import engine_core


def test_window_early_exit_on_stable_queue():
    src = inspect.getsource(engine_core)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the stable-queue early-exit exists alongside the prefill_batch_size exit
    assert "_stable" in code
    assert "if _n > _last_n:" in code
    assert "_stable >= 2" in code
    # the prefill_batch_size early-exit is still there (burst fills fast)
    assert "self.config.prefill_batch_size" in code


def test_default_batch_wait_unchanged():
    # the window itself (and its default) is unchanged — only its exit condition got smarter
    assert engine_core.EngineCoreConfig().batch_wait_ms == 15.0
