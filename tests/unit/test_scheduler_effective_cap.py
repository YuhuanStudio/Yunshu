"""the engine-loop scheduler (opt-in YUNSHU_ENGINE_LOOP=1) admitted up to
max_num_seqs (default 256) requests, but mlx-lm's BatchGenerator only DECODES
completion_batch_size (default 32) at once — its _next() early-returns at that size and
the surplus sit in mlx-lm's internal _unprocessed_sequences FIFO. So admitting beyond 32
(a) silently defeated our PRIORITY/FAIR/aging policy (mlx-lm's blind FIFO, not our
waiting-queue ordering, picked which surplus actually decoded) and (b) prefilled and held
the KV of all 256 while only 32 decoded — the OOM surface on a small Mac.

Fix: _effective_max_seqs = min(max_num_seqs, completion_batch_size) governs admission, so
the surplus stays in OUR waiting queue (policy/aging apply). The max_num_seqs config field
is unchanged (still a configurable knob); only the *effective* admission cap is corrected.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from yunshu_engine.scheduler import Scheduler, SchedulerConfig


def _make_scheduler(**overrides) -> Scheduler:
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = [2]
    tokenizer.encode = MagicMock(return_value=[1])
    tokenizer.has_thinking = False
    config = SchedulerConfig(model_name="test-model", **overrides)
    return Scheduler(model, tokenizer, config)


def test_effective_cap_clamps_to_completion_batch_size():
    # defaults: max_num_seqs=256, completion_batch_size=32 → effective 32
    s = _make_scheduler()
    assert s.config.max_num_seqs == 256  # field untouched
    assert s.config.completion_batch_size == 32
    assert s._effective_max_seqs == 32


def test_effective_cap_takes_the_smaller_side():
    # if the operator sets max_num_seqs BELOW the decode batch, honour the smaller
    s = _make_scheduler(max_num_seqs=8, completion_batch_size=32)
    assert s._effective_max_seqs == 8


def test_effective_cap_tracks_a_raised_completion_batch():
    # raising both keeps them coupled
    s = _make_scheduler(max_num_seqs=256, completion_batch_size=64)
    assert s._effective_max_seqs == 64


def test_spec_aware_scheduler_uses_effective_cap():
    # spec slot budgeting must be computed against the real decode capacity, not 256
    s = _make_scheduler(max_num_seqs=256, completion_batch_size=32)
    assert s._spec_aware_scheduler.max_num_seqs == 32


def test_spec_aware_scheduler_keeps_effective_cap_after_deep_reset():
    # deep_reset re-created the spec scheduler with the BARE max_num_seqs
    # (256), dropping the W853 effective cap → compute_spec_budget over-admitted up to
    # 256 after any fail-recovery / model-reload reset while the BatchGenerator only
    # decodes completion_batch_size. The cap must survive a deep_reset.
    s = _make_scheduler(max_num_seqs=256, completion_batch_size=32)
    assert s._spec_aware_scheduler.max_num_seqs == 32
    s.deep_reset()
    assert s._spec_aware_scheduler.max_num_seqs == 32
