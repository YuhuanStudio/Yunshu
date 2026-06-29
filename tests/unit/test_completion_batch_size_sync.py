"""(HIGH): adaptive batch sizing / AutoTuner mutated only
EngineCoreConfig.completion_batch_size, but the scheduler reads its OWN
SchedulerConfig copy (copied by value at init). So a tuned value never reached the
scheduler's admission cap (_effective_max_seqs) or spec-slot — adaptive batching
(incl. the ratchet) was a no-op for admission, and the token/spec budgets (which
DO read EngineCoreConfig) silently drifted away from the frozen admission width.

_set_completion_batch_size now writes BOTH configs so they stay consistent.
"""

from __future__ import annotations

import types

from yunshu_engine.engine_core import EngineCore


def _engine(eng_n=32, sched_n=32, with_scheduler=True):
    e = EngineCore.__new__(EngineCore)
    e.config = types.SimpleNamespace(completion_batch_size=eng_n)
    if with_scheduler:
        e.scheduler = types.SimpleNamespace(
            config=types.SimpleNamespace(completion_batch_size=sched_n)
        )
    return e


def test_sync_updates_both_configs():
    e = _engine()
    e._set_completion_batch_size(4)
    assert e.config.completion_batch_size == 4
    # the scheduler's frozen copy now follows → _effective_max_seqs throttles admission
    assert e.scheduler.config.completion_batch_size == 4


def test_no_drift_after_repeated_tuning():
    e = _engine()
    for n in (8, 16, 2, 31):
        e._set_completion_batch_size(n)
        assert e.config.completion_batch_size == n
        assert e.scheduler.config.completion_batch_size == n  # never diverges


def test_missing_scheduler_is_safe():
    e = _engine(with_scheduler=False)
    e.scheduler = None
    e._set_completion_batch_size(5)  # must not raise
    assert e.config.completion_batch_size == 5


def test_spec_aware_scheduler_max_num_seqs_synced():
    # the spec-aware scheduler snapshots its max_num_seqs at init as the     # effective cap; a tuned completion_batch_size must re-sync it too, or the spec slot
    # budget drifts from the real decode width.
    e = _engine()
    e.scheduler.config.max_num_seqs = 256
    e.scheduler._spec_aware_scheduler = types.SimpleNamespace(max_num_seqs=32)
    e._set_completion_batch_size(8)
    assert e.scheduler._spec_aware_scheduler.max_num_seqs == 8  # min(256, 8)


def test_no_spec_scheduler_is_safe():
    e = _engine()
    e.scheduler._spec_aware_scheduler = None  # n-gram-only / spec disabled
    e._set_completion_batch_size(4)  # must not raise
    assert e.config.completion_batch_size == 4


def test_apply_sites_route_through_helper():
    import inspect

    from yunshu_engine import engine_core

    src = inspect.getsource(engine_core.EngineCore)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # neither apply-site may write self.config.completion_batch_size directly anymore
    assert "self.config.completion_batch_size = max(" not in code
    # all THREE apply-sites (adaptive sizer, AutoTuner, memory-pressure rec) + the
    # helper-internal call go through the sync helper
    assert code.count("self._set_completion_batch_size(") >= 3
