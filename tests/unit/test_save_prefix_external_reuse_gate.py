"""(KV radix/prefix-cache hunt): the radix tree was verified CLEAN (longest-prefix
match, node split, ref-counting, eviction safety, W689 ordering fix all hold; no KV
corruption). One real efficiency bug: _save_one_prefix gated the KVPrefixCache save on
req.cached_tokens, but paged_scheduler sets cached_tokens from the RADIX accounting match
(num_matched_tokens). The radix tree and the KVPrefixCache are independent — when radix
matches a prefix but the KVPrefixCache MISSES, the request does a FULL BatchGenerator
prefill, yet cached_tokens stays >0, so the old gate skipped saving the genuinely-complete
prefill KV → that prefix was NEVER cached (permanent missed reuse on the engine-loop).

Fix: gate on an exact _external_kv_reuse flag set at insert time (True only when the
request actually warm-started from an external cached_kv via insert_segments, which is the
one case where extract_cache returns a short/corrupt cache).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from yunshu_engine.scheduler import Scheduler, SchedulerConfig


def _make_scheduler():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = [2]
    tokenizer.encode = MagicMock(return_value=[1])
    tokenizer.has_thinking = False
    return Scheduler(model, tokenizer, SchedulerConfig(model_name="test"))


def _setup(sched, *, external_reuse: bool, cached_tokens: int):
    pc = MagicMock()
    bg = MagicMock()
    bg.extract_cache = MagicMock(return_value={7: (object(),)})  # {uid: (cache_data,)}
    sched._prefix_cache = pc
    sched._batch_gen = bg
    sched._saved_prefix_uids = set()
    req = SimpleNamespace(
        num_output_tokens=5,
        num_prompt_tokens=64,
        prompt_token_ids=list(range(64)),
        cached_tokens=cached_tokens,
        _external_kv_reuse=external_reuse,
    )
    sched.running = {"req-1": req}
    return pc


def test_full_prefill_with_radix_match_is_saved():
    # radix matched (cached_tokens=100) but the request did a FULL prefill
    # (_external_kv_reuse=False) → MUST save (the old cached_tokens gate wrongly skipped).
    sched = _make_scheduler()
    pc = _setup(sched, external_reuse=False, cached_tokens=100)
    sched._save_one_prefix(7, "req-1")
    pc.add.assert_called_once()
    assert 7 in sched._saved_prefix_uids


def test_external_kv_reuse_is_not_saved():
    # genuinely reused external KV → extract_cache would be short → must NOT save.
    sched = _make_scheduler()
    pc = _setup(sched, external_reuse=True, cached_tokens=100)
    sched._save_one_prefix(7, "req-1")
    pc.add.assert_not_called()
    assert 7 in sched._saved_prefix_uids  # marked so it isn't retried every step


def test_plain_full_prefill_still_saved():
    # unchanged baseline: full prefill, no radix match → save.
    sched = _make_scheduler()
    pc = _setup(sched, external_reuse=False, cached_tokens=0)
    sched._save_one_prefix(7, "req-1")
    pc.add.assert_called_once()
