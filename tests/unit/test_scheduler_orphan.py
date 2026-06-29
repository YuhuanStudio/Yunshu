"""the engine-core scheduler must not orphan in-flight requests.

Two step() paths removed requests from the scheduler without emitting a
`finished` output, so the request's completion event never fired and the client
hung until request_timeout:
  - the cache-corruption branch called deep_reset() with no error outputs;
  - the memory-guard rejection branch dropped the in-hand to_insert requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from yunshu_engine.exceptions import CacheCorruptionError
from yunshu_engine.scheduler import Scheduler, SchedulerConfig


def _sched():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = [2]
    return Scheduler(model, tokenizer, SchedulerConfig(model_name="test-model"))


def test_corruption_reset_emits_error_outputs_for_inflight():
    sched = _sched()
    # Two in-flight requests (running → _has_active_requests True; in requests so
    # the deep_reset capture sees them).
    for rid in ("r1", "r2"):
        sched.requests[rid] = MagicMock(request_id=rid)
        sched.running[rid] = sched.requests[rid]

    # Make the BatchGenerator step raise a cache-corruption error.
    # is_cache_corruption_error matches on message substrings (e.g. "KVCache").
    bg = MagicMock()
    bg.next.side_effect = CacheCorruptionError("BatchKVCache state corrupted")
    sched._batch_gen = bg

    out = sched.step()
    emitted = {o.request_id: o for o in out.outputs}
    # Both in-flight requests get a finished error output (not orphaned).
    for rid in ("r1", "r2"):
        assert rid in emitted, f"{rid} was orphaned (no output)"
        assert emitted[rid].finished is True
        assert emitted[rid].finish_reason == "error"
    # deep_reset wiped the scheduler state.
    assert not sched.requests
    assert not sched.running
