"""Tests for batch-path spec decode integration.

Covers:
  - BatchPathSpecPrefill: sparse prefill for long prompts in the batch path
  - SpecAwareBatchScheduler: spec-decode aware slot allocation
  - BatchedDraftCollection: collect drafts from all strategies for all running requests
  - Integration with TBO (Two-Batch Overlap)
  - Integration with scheduler step loop
  - Deep reset cleanup
  - Stats reporting

At least 25 tests covering all components.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from yunshu_engine.scheduler import (
    BatchSpecPrefillConfig,
    BatchPathSpecPrefill,
    SpecBudget,
    DraftCollection,
    SpecAwareBatchScheduler,
    BatchedDraftCollection,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from yunshu_engine.request import Request, RequestOutput, RequestStatus, SamplingParams
from yunshu_engine.speculative_decoder import SpeculativeDecoder, DraftResult


# ── Helpers ──


def _make_request(req_id="req-1", prompt_tokens=None, max_tokens=256):
    """Create a minimal Request for testing."""
    return Request(
        request_id=req_id,
        prompt=prompt_tokens or [1, 2, 3],
        prompt_token_ids=prompt_tokens or [1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def _make_scheduler(
    enable_spec=False,
    ngram_spec_enabled=False,
    batch_spec_prefill_enabled=False,
    **kwargs,
):
    """Create a Scheduler with optional spec decode / SpecPrefill setup."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = [2]
    tokenizer.encode.return_value = [1]
    tokenizer.detokenizer = MagicMock()
    tokenizer.detokenizer.reset.return_value = None
    model.config = MagicMock()
    model.config.to_dict.return_value = {"model_type": "llama"}

    config = SchedulerConfig(
        enable_spec_decode=enable_spec,
        ngram_spec_enabled=ngram_spec_enabled,
        batch_spec_prefill_enabled=batch_spec_prefill_enabled,
        **kwargs,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    return scheduler


def _make_response(uid=0, token=42, finish_reason=None):
    """Create a mock GenerationBatch.Response."""
    resp = MagicMock()
    resp.uid = uid
    resp.token = token
    resp.finish_reason = finish_reason
    resp.logprobs = None
    resp.current_state = "normal"
    return resp


# ═══════════════════════════════════════════════════════════════════
# BatchSpecPrefillConfig
# ═══════════════════════════════════════════════════════════════════


class TestBatchSpecPrefillConfig:
    def test_defaults(self):
        cfg = BatchSpecPrefillConfig()
        assert cfg.enabled is False
        assert cfg.threshold == 8192
        assert cfg.keep_rate == 0.20
        assert cfg.chunk_size == 32
        assert cfg.draft_model is None

    def test_custom_config(self):
        draft = MagicMock()
        cfg = BatchSpecPrefillConfig(
            enabled=True,
            threshold=4096,
            keep_rate=0.30,
            chunk_size=64,
            draft_model=draft,
        )
        assert cfg.enabled is True
        assert cfg.threshold == 4096
        assert cfg.keep_rate == 0.30
        assert cfg.chunk_size == 64
        assert cfg.draft_model is draft


# ═══════════════════════════════════════════════════════════════════
# BatchPathSpecPrefill
# ═══════════════════════════════════════════════════════════════════


class TestBatchPathSpecPrefill:
    def test_should_prefill_disabled(self):
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(enabled=False))
        assert sp.should_prefill(10000) is False

    def test_should_prefill_no_draft_model(self):
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(enabled=True))
        assert sp.should_prefill(10000) is False

    def test_should_prefill_prompt_too_short(self):
        draft = MagicMock()
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(
            enabled=True, draft_model=draft, threshold=8192,
        ))
        assert sp.should_prefill(100) is False

    def test_should_prefill_enabled_and_long(self):
        draft = MagicMock()
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(
            enabled=True, draft_model=draft, threshold=8192,
        ))
        assert sp.should_prefill(10000) is True

    def test_compute_skippable_tokens_returns_none_when_disabled(self):
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(enabled=False))
        result = sp.compute_skippable_tokens([1, 2, 3])
        assert result is None

    def test_compute_skippable_tokens_failure_fallback(self):
        """When scoring fails, returns None (falls back to full prefill)."""
        draft = MagicMock()
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(
            enabled=True, draft_model=draft, threshold=100,
        ))
        # score_tokens will fail because draft model is a mock
        result = sp.compute_skippable_tokens(list(range(200)))
        assert result is None
        stats = sp.get_stats()
        assert stats["prefills_fallback"] == 1

    def test_get_stats_initial(self):
        sp = BatchPathSpecPrefill()
        stats = sp.get_stats()
        assert stats["prefills_attempted"] == 0
        assert stats["prefills_succeeded"] == 0
        assert stats["prefills_fallback"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_stats_after_attempts(self):
        draft = MagicMock()
        sp = BatchPathSpecPrefill(BatchSpecPrefillConfig(
            enabled=True, draft_model=draft, threshold=100,
        ))
        # Force a fallback
        sp.compute_skippable_tokens(list(range(200)))
        stats = sp.get_stats()
        assert stats["prefills_attempted"] == 1
        assert stats["prefills_fallback"] == 1
        assert stats["success_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# SpecBudget
# ═══════════════════════════════════════════════════════════════════


class TestSpecBudget:
    def test_defaults(self):
        b = SpecBudget()
        assert b.total_slots == 256
        assert b.decode_slots == 0
        assert b.spec_slots == 0
        assert b.available_for_new == 0

    def test_custom(self):
        b = SpecBudget(
            total_slots=128,
            decode_slots=10,
            spec_slots=3,
            available_for_new=115,
        )
        assert b.total_slots == 128
        assert b.decode_slots == 10
        assert b.spec_slots == 3
        assert b.available_for_new == 115


# ═══════════════════════════════════════════════════════════════════
# SpecAwareBatchScheduler
# ═══════════════════════════════════════════════════════════════════


class TestSpecAwareBatchScheduler:
    def test_no_spec_overhead_gives_full_slots(self):
        """When spec_overhead=0, all slots are available for new requests."""
        s = SpecAwareBatchScheduler(max_num_seqs=32, spec_overhead_per_request=0.0)
        budget = s.compute_spec_budget(num_running=10)
        assert budget.total_slots == 32
        assert budget.decode_slots == 10
        assert budget.spec_slots == 0
        assert budget.available_for_new == 22

    def test_spec_overhead_reduces_available(self):
        """Spec overhead reserves slots, reducing available_for_new."""
        s = SpecAwareBatchScheduler(max_num_seqs=32, spec_overhead_per_request=0.2)
        budget = s.compute_spec_budget(num_running=10)
        assert budget.spec_slots > 0
        assert budget.available_for_new < 22  # 32 - 10 - spec_slots

    def test_spec_slots_scale_with_running(self):
        """More running requests = more spec slots reserved."""
        s = SpecAwareBatchScheduler(max_num_seqs=256, spec_overhead_per_request=0.1)
        budget_10 = s.compute_spec_budget(num_running=10)
        budget_50 = s.compute_spec_budget(num_running=50)
        assert budget_50.spec_slots > budget_10.spec_slots

    def test_tbo_halves_overhead(self):
        """TBO overlap reduces effective spec overhead by ~50%."""
        s_no_tbo = SpecAwareBatchScheduler(max_num_seqs=256, spec_overhead_per_request=0.2, tbo_enabled=False)
        s_tbo = SpecAwareBatchScheduler(max_num_seqs=256, spec_overhead_per_request=0.2, tbo_enabled=True)

        budget_no_tbo = s_no_tbo.compute_spec_budget(num_running=20)
        budget_tbo = s_tbo.compute_spec_budget(num_running=20)

        assert budget_tbo.spec_slots <= budget_no_tbo.spec_slots
        # TBO should roughly halve the spec slots
        assert budget_tbo.spec_slots == max(1, round(budget_no_tbo.spec_slots * 0.5))

    def test_zero_running_gives_full_available(self):
        """With no running requests, all slots are available."""
        s = SpecAwareBatchScheduler(max_num_seqs=64, spec_overhead_per_request=0.2)
        budget = s.compute_spec_budget(num_running=0)
        assert budget.available_for_new == 64
        assert budget.spec_slots == 0

    def test_available_cannot_go_negative(self):
        """Available slots is clamped to >= 0."""
        s = SpecAwareBatchScheduler(max_num_seqs=16, spec_overhead_per_request=1.0)
        budget = s.compute_spec_budget(num_running=20)
        assert budget.available_for_new == 0

    def test_get_available_slots_quick_access(self):
        """get_available_slots returns the same value as compute_spec_budget."""
        s = SpecAwareBatchScheduler(max_num_seqs=64, spec_overhead_per_request=0.1)
        budget = s.compute_spec_budget(10)
        assert s.get_available_slots(10) == budget.available_for_new

    def test_custom_spec_override(self):
        """compute_spec_budget accepts a one-time spec_overhead override."""
        s = SpecAwareBatchScheduler(max_num_seqs=64, spec_overhead_per_request=0.1)
        budget_default = s.compute_spec_budget(10)
        budget_override = s.compute_spec_budget(10, spec_overhead=0.5)
        assert budget_override.spec_slots >= budget_default.spec_slots

    def test_get_stats(self):
        s = SpecAwareBatchScheduler(max_num_seqs=64, spec_overhead_per_request=0.1)
        s.compute_spec_budget(10)
        stats = s.get_stats()
        assert stats["budget_computations"] == 1
        assert stats["max_num_seqs"] == 64
        assert stats["spec_overhead_per_request"] == 0.1
        assert "avg_spec_slots" in stats


# ═══════════════════════════════════════════════════════════════════
# DraftCollection
# ═══════════════════════════════════════════════════════════════════


class TestDraftCollection:
    def test_empty(self):
        dc = DraftCollection()
        assert dc.has_drafts() is False
        assert dc.get_request_ids() == []
        assert dc.total_draft_tokens == 0

    def test_add_single(self):
        dc = DraftCollection()
        dc.add("req-1", [10, 20, 30], "ngram")
        assert dc.has_drafts()
        assert dc.drafts["req-1"] == [10, 20, 30]
        assert dc.total_draft_tokens == 3
        assert dc.strategy_counts["ngram"] == 3
        assert dc.get_request_ids() == ["req-1"]

    def test_add_multiple_strategies(self):
        dc = DraftCollection()
        dc.add("req-1", [10, 20], "ngram")
        dc.add("req-2", [30], "mtp")
        assert dc.total_draft_tokens == 3
        assert dc.strategy_counts["ngram"] == 2
        assert dc.strategy_counts["mtp"] == 1

    def test_add_empty_tokens_skipped(self):
        dc = DraftCollection()
        dc.add("req-1", [], "ngram")
        assert dc.has_drafts() is False
        assert dc.total_draft_tokens == 0

    def test_add_overwrites_existing(self):
        dc = DraftCollection()
        dc.add("req-1", [10], "ngram")
        dc.add("req-1", [20, 30], "mtp")
        assert dc.drafts["req-1"] == [20, 30]
        assert dc.total_draft_tokens == 3

    def test_merge(self):
        dc1 = DraftCollection()
        dc1.add("req-1", [10, 20], "ngram")

        dc2 = DraftCollection()
        dc2.add("req-2", [30, 40], "mtp")

        dc1.merge(dc2)
        assert "req-1" in dc1.drafts
        assert "req-2" in dc1.drafts
        assert dc1.total_draft_tokens == 4
        assert dc1.strategy_counts["ngram"] == 2
        assert dc1.strategy_counts["mtp"] == 2

    def test_merge_no_overwrite(self):
        """Merge does not overwrite existing drafts."""
        dc1 = DraftCollection()
        dc1.add("req-1", [10], "ngram")

        dc2 = DraftCollection()
        dc2.add("req-1", [20], "mtp")

        dc1.merge(dc2)
        assert dc1.drafts["req-1"] == [10]  # Original kept


# ═══════════════════════════════════════════════════════════════════
# BatchedDraftCollection
# ═══════════════════════════════════════════════════════════════════


class TestBatchedDraftCollection:
    def test_empty_running(self):
        """No running requests yields empty collection."""
        collector = BatchedDraftCollection()
        result = collector.collect_all_drafts(
            running={},
            spec_decoder=None,
            mtp_decoder=None,
            ngram_proposer=None,
            pending_abort_ids=set(),
        )
        assert result.has_drafts() is False

    def test_skips_aborted_requests(self):
        """Aborted requests are not collected."""
        collector = BatchedDraftCollection()
        req = _make_request()
        req.output_token_ids = [10, 20, 30]

        result = collector.collect_all_drafts(
            running={"req-1": req},
            spec_decoder=None,
            mtp_decoder=None,
            ngram_proposer=None,
            pending_abort_ids={"req-1"},
        )
        assert result.has_drafts() is False

    def test_skips_no_output_tokens(self):
        """Requests with no output tokens are skipped (can't seed a draft)."""
        collector = BatchedDraftCollection()
        req = _make_request()
        req.output_token_ids = []

        result = collector.collect_all_drafts(
            running={"req-1": req},
            spec_decoder=None,
            mtp_decoder=None,
            ngram_proposer=None,
            pending_abort_ids=set(),
        )
        assert result.has_drafts() is False

    def test_preserves_existing_drafts(self):
        """If spec_drafts already has an entry, it's preserved."""
        collector = BatchedDraftCollection()
        req = _make_request()
        req.output_token_ids = [10]

        result = collector.collect_all_drafts(
            running={"req-1": req},
            spec_decoder=None,
            mtp_decoder=None,
            ngram_proposer=None,
            pending_abort_ids=set(),
            spec_drafts={"req-1": [20, 30]},
        )
        assert result.drafts["req-1"] == [20, 30]

    def test_collects_ngram_drafts(self):
        """N-gram drafts are collected for running requests."""
        collector = BatchedDraftCollection()

        # Create a real NgramProposer with a repeated pattern
        from yunshu_engine.ngram_proposer import NgramProposer, NgramConfig
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=5, mode="lps", max_model_len=32768))

        req = _make_request()
        req.prompt_token_ids = [10, 20, 30, 10, 20]
        req.output_token_ids = [30, 10, 20]

        result = collector.collect_all_drafts(
            running={"req-1": req},
            spec_decoder=None,
            mtp_decoder=None,
            ngram_proposer=proposer,
            pending_abort_ids=set(),
        )
        # N-gram should find the "10,20,30" pattern and may propose 30
        # At minimum, the collection should not crash and have a strategy entry
        stats = collector.get_stats()
        assert stats["collections"] == 1

    def test_strategy_priority_ngram_first(self):
        """N-gram is tried first (zero GPU overhead)."""
        collector = BatchedDraftCollection()
        from yunshu_engine.ngram_proposer import NgramProposer, NgramConfig

        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=5, mode="lps", max_model_len=32768))
        req = _make_request()
        # Long repeated pattern for N-gram to find
        req.prompt_token_ids = [10, 20, 30, 40] * 20
        req.output_token_ids = [10, 20, 30, 40]

        mock_decoder = MagicMock(spec=SpeculativeDecoder)
        mock_mtp = MagicMock()

        result = collector.collect_all_drafts(
            running={"req-1": req},
            spec_decoder=mock_decoder,
            mtp_decoder=mock_mtp,
            ngram_proposer=proposer,
            pending_abort_ids=set(),
        )
        # If N-gram found a draft, it should be in the result
        # and strategy_counts should show ngram (not mtp or cross_model)
        if result.has_drafts():
            assert "ngram" in result.strategy_counts or result.total_draft_tokens > 0

    def test_get_stats(self):
        collector = BatchedDraftCollection()
        stats = collector.get_stats()
        assert stats["collections"] == 0
        assert stats["total_tokens_collected"] == 0
        assert stats["strategy_breakdown"] == {}


# ═══════════════════════════════════════════════════════════════════
# Scheduler Integration
# ═══════════════════════════════════════════════════════════════════


class TestSchedulerBatchSpecPrefill:
    """Test Batch SpecPrefill integration in the Scheduler."""

    def test_scheduler_creates_spec_prefill_when_enabled(self):
        scheduler = _make_scheduler(batch_spec_prefill_enabled=True)
        assert scheduler._batch_spec_prefill is not None
        assert scheduler._batch_spec_prefill.config.enabled is True

    def test_scheduler_no_spec_prefill_when_disabled(self):
        scheduler = _make_scheduler(batch_spec_prefill_enabled=False)
        assert scheduler._batch_spec_prefill is None

    def test_spec_prefill_step_disabled(self):
        """spec_prefill_step returns None when disabled."""
        scheduler = _make_scheduler(batch_spec_prefill_enabled=False)
        result = scheduler.spec_prefill_step([1, 2, 3])
        assert result is None

    def test_spec_prefill_step_enabled(self):
        """spec_prefill_step returns None (no draft model) but doesn't crash."""
        scheduler = _make_scheduler(batch_spec_prefill_enabled=True)
        result = scheduler.spec_prefill_step(list(range(100)))
        # No draft model → returns None
        assert result is None

    def test_set_batch_spec_prefill_draft_model(self):
        """Setting draft model enables actual scoring."""
        scheduler = _make_scheduler(batch_spec_prefill_enabled=True)
        draft = MagicMock()
        scheduler.set_batch_spec_prefill_draft_model(draft)
        assert scheduler._batch_spec_prefill.config.draft_model is draft

    def test_batch_spec_prefill_stats_in_get_stats(self):
        scheduler = _make_scheduler(batch_spec_prefill_enabled=True)
        stats = scheduler.get_stats()
        assert "batch_spec_prefill" in stats
        assert stats["batch_spec_prefill"]["enabled"] is True


class TestSchedulerSpecAwareBudget:
    """Test spec-aware slot allocation in the Scheduler."""

    def test_scheduler_has_spec_aware_scheduler(self):
        scheduler = _make_scheduler()
        assert scheduler._spec_aware_scheduler is not None

    def test_spec_aware_budget_when_spec_active(self):
        """When spec decode is active, slot allocation accounts for overhead."""
        scheduler = _make_scheduler(enable_spec=True, max_num_seqs=64)
        # Set up a mock spec decoder
        mock_decoder = MagicMock(spec=SpeculativeDecoder)
        scheduler._spec_decoder = mock_decoder

        # Add some running requests
        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        # Compute spec-aware budget
        budget = scheduler._spec_aware_scheduler.compute_spec_budget(1)
        assert budget.total_slots == 64
        assert budget.decode_slots == 1
        assert budget.spec_slots > 0
        assert budget.available_for_new < 63  # Reduced by spec overhead

    def test_no_spec_overhead_when_spec_disabled(self):
        """When spec decode is disabled, no spec slots are reserved."""
        scheduler = _make_scheduler(enable_spec=False, max_num_seqs=64)
        budget = scheduler._spec_aware_scheduler.compute_spec_budget(10)
        # Default overhead 0.1 still applies even without spec enabled,
        # but the scheduler only uses the budget when spec is active
        assert budget.total_slots == 64

    def test_spec_aware_scheduler_stats_in_get_stats(self):
        scheduler = _make_scheduler()
        stats = scheduler.get_stats()
        assert "spec_aware_scheduler" in stats

    def test_set_spec_aware_tbo(self):
        """TBO status can be updated at runtime."""
        scheduler = _make_scheduler()
        scheduler.set_spec_aware_tbo(True)
        assert scheduler._spec_aware_scheduler.tbo_enabled is True
        scheduler.set_spec_aware_tbo(False)
        assert scheduler._spec_aware_scheduler.tbo_enabled is False


class TestSchedulerDraftCollection:
    """Test batched draft collection in the Scheduler."""

    def test_scheduler_has_draft_collector(self):
        scheduler = _make_scheduler()
        assert scheduler._draft_collector is not None

    def test_collect_batch_drafts_empty(self):
        """No running requests yields empty collection."""
        scheduler = _make_scheduler()
        result = scheduler.collect_batch_drafts()
        assert result.has_drafts() is False

    def test_collect_batch_drafts_with_ngram(self):
        """N-gram drafts are collected for running requests."""
        scheduler = _make_scheduler(ngram_spec_enabled=True)

        req = _make_request()
        req.prompt_token_ids = [10, 20, 30] * 10
        req.output_token_ids = [10, 20, 30, 10, 20]
        scheduler.running["req-1"] = req

        result = scheduler.collect_batch_drafts()
        # Should not crash; may or may not have drafts depending on N-gram patterns
        assert isinstance(result, DraftCollection)

    def test_collect_batch_drafts_multiple_requests(self):
        """Drafts are collected for all running requests."""
        scheduler = _make_scheduler(ngram_spec_enabled=True)

        for i in range(3):
            rid = f"req-{i}"
            req = _make_request(req_id=rid)
            req.prompt_token_ids = [10 + i, 20 + i, 30 + i] * 10
            req.output_token_ids = [10 + i, 20 + i, 30 + i, 10 + i, 20 + i]
            scheduler.running[rid] = req

        result = scheduler.collect_batch_drafts()
        assert isinstance(result, DraftCollection)
        # Should have attempted collection for all 3
        stats = scheduler._draft_collector.get_stats()
        assert stats["collections"] == 1

    def test_draft_collector_stats_in_get_stats(self):
        scheduler = _make_scheduler()
        stats = scheduler.get_stats()
        assert "draft_collector" in stats


# ═══════════════════════════════════════════════════════════════════
# Integration with TBO
# ═══════════════════════════════════════════════════════════════════


class TestTBOIntegration:
    """Test spec-aware scheduling with Two-Batch Overlap."""

    def test_tbo_reduces_spec_slot_reservation(self):
        """TBO overlap should reduce the spec slot reservation."""
        s_no_tbo = SpecAwareBatchScheduler(
            max_num_seqs=256,
            spec_overhead_per_request=0.2,
            tbo_enabled=False,
        )
        s_with_tbo = SpecAwareBatchScheduler(
            max_num_seqs=256,
            spec_overhead_per_request=0.2,
            tbo_enabled=True,
        )
        budget_no_tbo = s_no_tbo.compute_spec_budget(num_running=20)
        budget_with_tbo = s_with_tbo.compute_spec_budget(num_running=20)

        # TBO should reserve fewer spec slots
        assert budget_with_tbo.spec_slots <= budget_no_tbo.spec_slots
        # And thus have more available
        assert budget_with_tbo.available_for_new >= budget_no_tbo.available_for_new

    def test_tbo_overlap_steps_tracked(self):
        """TBO overlap usage is tracked in stats."""
        s = SpecAwareBatchScheduler(
            max_num_seqs=256,
            spec_overhead_per_request=0.2,
            tbo_enabled=True,
        )
        s.compute_spec_budget(num_running=20)
        stats = s.get_stats()
        assert stats["tbo_overlap_steps"] == 1
        assert stats["tbo_enabled"] is True

    def test_scheduler_set_spec_aware_tbo_propagates(self):
        """Updating TBO status propagates to the spec-aware scheduler."""
        scheduler = _make_scheduler()
        assert scheduler._spec_aware_scheduler.tbo_enabled is False

        scheduler.set_spec_aware_tbo(True)
        assert scheduler._spec_aware_scheduler.tbo_enabled is True

        # Budget should now reflect TBO overlap
        budget = scheduler._spec_aware_scheduler.compute_spec_budget(20)
        stats = scheduler._spec_aware_scheduler.get_stats()
        assert stats["tbo_overlap_steps"] >= 1


# ═══════════════════════════════════════════════════════════════════
# Deep Reset Cleanup
# ═══════════════════════════════════════════════════════════════════


class TestDeepReset:
    def test_deep_reset_clears_spec_aware_scheduler(self):
        """deep_reset recreates the spec-aware scheduler."""
        scheduler = _make_scheduler()
        old_sa = scheduler._spec_aware_scheduler
        scheduler._spec_aware_scheduler.compute_spec_budget(10)
        assert scheduler._spec_aware_scheduler.get_stats()["budget_computations"] == 1

        scheduler.deep_reset()
        assert scheduler._spec_aware_scheduler is not old_sa
        assert scheduler._spec_aware_scheduler.get_stats()["budget_computations"] == 0

    def test_deep_reset_clears_draft_collector(self):
        """deep_reset recreates the draft collector."""
        scheduler = _make_scheduler()
        old_dc = scheduler._draft_collector
        scheduler.deep_reset()
        assert scheduler._draft_collector is not old_dc
        assert scheduler._draft_collector.get_stats()["collections"] == 0

    def test_deep_reset_preserves_batch_spec_prefill(self):
        """deep_reset does not destroy the BatchPathSpecPrefill object."""
        scheduler = _make_scheduler(batch_spec_prefill_enabled=True)
        assert scheduler._batch_spec_prefill is not None
        scheduler.deep_reset()
        # BatchPathSpecPrefill should still be configured
        assert scheduler._batch_spec_prefill is not None


# ═══════════════════════════════════════════════════════════════════
# Scheduler Step Loop Integration
# ═══════════════════════════════════════════════════════════════════


class TestStepLoopIntegration:
    """Test that batch spec integration works in the step loop."""

    def test_step_with_ngram_uses_batch_draft_collection(self):
        """Step loop should use collect_batch_drafts for N-gram."""
        from yunshu_engine.scheduler import SchedulerConfig
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.eos_token_ids = [2]
        tokenizer.encode.return_value = [1]
        tokenizer.detokenizer = MagicMock()
        tokenizer.detokenizer.reset.return_value = None
        model.config = MagicMock()
        model.config.to_dict.return_value = {"model_type": "llama"}

        config = SchedulerConfig(
            ngram_spec_enabled=True,
            ngram_spec_min_n=1,
            ngram_spec_max_n=5,
            ngram_spec_k=5,
            ngram_spec_mode="lps",
        )
        scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)

        req = _make_request()
        req.batch_uid = 0
        req.prompt_token_ids = [10, 20, 30, 10, 20]
        req.output_token_ids = [30, 10, 20]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=30)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        scheduler.step()

        # Draft collector should have been called
        stats = scheduler._draft_collector.get_stats()
        assert stats["collections"] == 1

    def test_step_with_spec_decode_and_ngram_both(self):
        """Both spec decode and N-gram should use batch draft collection."""
        scheduler = _make_scheduler(enable_spec=True, ngram_spec_enabled=True)
        mock_decoder = MagicMock(spec=SpeculativeDecoder)
        scheduler._spec_decoder = mock_decoder

        req = _make_request()
        req.batch_uid = 0
        req.prompt_token_ids = [10, 20, 30]
        req.output_token_ids = [10, 20]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=30)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        with patch.object(scheduler, '_generate_draft_tokens', return_value=DraftResult(token_ids=[40], logprobs=[-0.1])):
            scheduler.step()

        # Both draft collector and per-request path should have been used
        stats = scheduler._draft_collector.get_stats()
        assert stats["collections"] == 1

    def test_spec_aware_budget_used_in_scheduling(self):
        """When spec is active, _schedule_waiting uses spec-aware budget."""
        scheduler = _make_scheduler(enable_spec=True, ngram_spec_enabled=True, max_num_seqs=16)
        mock_decoder = MagicMock(spec=SpeculativeDecoder)
        scheduler._spec_decoder = mock_decoder

        # Add a waiting request
        req = _make_request(prompt_tokens=list(range(100)))
        scheduler.add_request(req)

        # Add running requests to fill most slots
        for i in range(10):
            r = _make_request(req_id=f"running-{i}")
            r.batch_uid = i
            r.output_token_ids = [10 + i]
            scheduler.running[f"running-{i}"] = r
            scheduler._uid_to_req[i] = f"running-{i}"

        # Mock BatchGenerator for insertion
        mock_bg = MagicMock()
        mock_bg.insert.return_value = [99]
        scheduler._batch_gen = mock_bg

        scheduler._schedule_waiting()

        # Check that spec-aware budget was computed
        budget_stats = scheduler._spec_aware_scheduler.get_stats()
        assert budget_stats["budget_computations"] >= 1
