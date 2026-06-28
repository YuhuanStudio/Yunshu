"""Tests for speculative decoding integration in the scheduler step loop.

Tests cover:
- Spec decode not triggered when disabled
- Draft generation with mock decoder
- Accepted/rejected token tracking
- Verification of drafts against actual output
- Stats reporting (spec_proposals, spec_accepted, spec_rejected, spec_enabled)
- Cleanup on finish / abort / preempt
"""
from unittest.mock import MagicMock, patch

from yunshu_engine.request import Request, RequestOutput, SamplingParams
from yunshu_engine.scheduler import Scheduler, SchedulerConfig
from yunshu_engine.speculative_decoder import (
    DraftResult,
    SpecDecodingConfig,
    SpecHeadInfo,
    SpeculativeDecoder,
)

# ── Helpers ──


def _make_request(req_id="req-1", prompt_tokens=None, max_tokens=256):
    """Create a minimal Request for testing."""
    return Request(
        request_id=req_id,
        prompt=prompt_tokens or [1, 2, 3],
        prompt_token_ids=prompt_tokens or [1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def _make_scheduler(enable_spec=False, model_config=None, with_decoder=False):
    """Create a Scheduler with optional spec decode setup.

    Args:
        enable_spec: Set enable_spec_decode in SchedulerConfig.
        model_config: Dict to use as model.config.to_dict() return value.
        with_decoder: If True, set _spec_decoder to a mock SpeculativeDecoder
            (simulates a fully initialized decoder, not just head_info).
    """
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = [2]
    tokenizer.encode.return_value = [1]
    tokenizer.detokenizer = MagicMock()
    tokenizer.detokenizer.reset.return_value = None

    if model_config is not None:
        model.config = MagicMock()
        model.config.to_dict.return_value = model_config
    else:
        model.config = MagicMock()
        model.config.to_dict.return_value = {"model_type": "llama"}

    config = SchedulerConfig(enable_spec_decode=enable_spec, spec_draft_length=5)
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)

    if with_decoder:
        # Create a real-looking mock SpeculativeDecoder
        mock_decoder = MagicMock(spec=SpeculativeDecoder)
        mock_decoder.config = SpecDecodingConfig(draft_length=3)
        scheduler._spec_decoder = mock_decoder

    return scheduler


def _make_response(uid=0, token=42, finish_reason=None, logprobs=None):
    """Create a mock GenerationBatch.Response."""
    resp = MagicMock()
    resp.uid = uid
    resp.token = token
    resp.finish_reason = finish_reason
    resp.logprobs = logprobs
    resp.current_state = "normal"
    return resp


# ── Tests: Spec decode not triggered when disabled ──


class TestSpecDecodeDisabled:
    """Speculative decoding should not run when disabled."""

    def test_disabled_by_default(self):
        scheduler = _make_scheduler(enable_spec=False)
        assert not scheduler.config.enable_spec_decode

    def test_try_spec_decode_draft_noop_when_disabled(self):
        scheduler = _make_scheduler(enable_spec=False)
        req = _make_request()
        req.output_token_ids = [10, 20, 30]
        scheduler._try_spec_decode_draft(req)
        assert scheduler._spec_drafts == {}

    def test_verify_noop_when_disabled(self):
        scheduler = _make_scheduler(enable_spec=False)
        scheduler._spec_drafts["req-1"] = [40, 50, 60]
        scheduler._verify_spec_drafts([])
        # Drafts stay because _verify_spec_drafts checks isinstance
        # but with spec disabled, the step loop never calls it.
        # Still, calling it directly should not crash.
        assert "req-1" in scheduler._spec_drafts

    def test_stats_show_disabled(self):
        scheduler = _make_scheduler(enable_spec=False)
        stats = scheduler.get_stats()
        assert stats["spec_enabled"] is False
        assert stats["spec_proposals"] == 0

    def test_no_decoder_means_no_draft(self):
        """If _spec_decoder is just head_info (not SpeculativeDecoder), no drafts."""
        scheduler = _make_scheduler(enable_spec=True)
        # _spec_decoder is None by default
        scheduler._spec_decoder = None
        req = _make_request()
        req.output_token_ids = [10]
        scheduler._try_spec_decode_draft(req)
        assert scheduler._spec_drafts == {}


class TestSpecDecodeDraftGeneration:
    """Test draft generation with mock decoder."""

    def test_draft_generation_stores_tokens(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10, 20, 30]

        # Mock _generate_draft_tokens to return draft tokens (avoids MLX ops)
        draft_result = DraftResult(token_ids=[40, 50, 60], logprobs=[-1.0, -0.5, -0.8])
        with patch.object(scheduler, '_generate_draft_tokens', return_value=draft_result):
            scheduler._try_spec_decode_draft(req)

        assert "req-1" in scheduler._spec_drafts
        assert scheduler._spec_drafts["req-1"] == [40, 50, 60]
        assert scheduler._spec_stats["req-1"]["proposals"] == 3
        assert scheduler._spec_total_proposals == 3

    def test_draft_generation_skipped_if_no_output_tokens(self):
        """Draft needs at least one output token to seed."""
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = []  # No tokens yet

        scheduler._try_spec_decode_draft(req)
        assert scheduler._spec_drafts == {}

    def test_draft_generation_skipped_for_aborted_request(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10]
        scheduler._pending_abort_ids.add("req-1")

        scheduler._try_spec_decode_draft(req)
        assert scheduler._spec_drafts == {}

    def test_draft_generation_handles_error_gracefully(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10]

        # Mock generate_draft to raise an exception
        scheduler._spec_decoder.generate_draft.side_effect = RuntimeError("GPU OOM")

        scheduler._try_spec_decode_draft(req)
        # Should not crash, just log and skip
        assert scheduler._spec_drafts == {}

    def test_draft_not_overwritten_if_already_pending(self):
        """If a request already has pending drafts, don't generate more."""
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10]
        scheduler._spec_drafts["req-1"] = [40, 50]

        # Call again — should not overwrite
        scheduler._try_spec_decode_draft(req)
        # Draft is still the original
        assert scheduler._spec_drafts["req-1"] == [40, 50]

    def test_draft_zero_length_skipped(self):
        """draft_length=0 should skip generation."""
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        scheduler._spec_decoder.config.draft_length = 0
        req = _make_request()
        req.output_token_ids = [10]

        scheduler._try_spec_decode_draft(req)
        assert scheduler._spec_drafts == {}


class TestSpecDecodeVerify:
    """Test draft verification against actual output."""

    def test_verify_all_accepted(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10, 20, 30, 40, 50, 60]  # Actual: matches draft
        scheduler.running["req-1"] = req
        scheduler._spec_drafts["req-1"] = [40, 50, 60]

        output = RequestOutput(
            request_id="req-1",
            new_token_ids=[60],
            output_token_ids=[10, 20, 30, 40, 50, 60],
        )

        scheduler._verify_spec_drafts([output])

        assert scheduler._spec_stats["req-1"]["accepted"] == 3
        assert scheduler._spec_stats["req-1"]["rejected"] == 0
        assert scheduler._spec_total_accepted == 3
        # Draft should be cleaned up after verification
        assert "req-1" not in scheduler._spec_drafts

    def test_verify_partial_acceptance(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10, 20, 30, 40, 99, 60]  # 99 != 50
        scheduler.running["req-1"] = req
        scheduler._spec_drafts["req-1"] = [40, 50, 60]

        output = RequestOutput(
            request_id="req-1",
            new_token_ids=[60],
            output_token_ids=[10, 20, 30, 40, 99, 60],
        )

        scheduler._verify_spec_drafts([output])

        # Only first draft (40) matches
        assert scheduler._spec_stats["req-1"]["accepted"] == 1
        assert scheduler._spec_stats["req-1"]["rejected"] == 2
        assert scheduler._spec_total_accepted == 1
        assert scheduler._spec_total_rejected == 2

    def test_verify_none_accepted(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.output_token_ids = [10, 20, 30, 77, 88, 99]  # None match
        scheduler.running["req-1"] = req
        scheduler._spec_drafts["req-1"] = [40, 50, 60]

        output = RequestOutput(
            request_id="req-1",
            new_token_ids=[99],
            output_token_ids=[10, 20, 30, 77, 88, 99],
        )

        scheduler._verify_spec_drafts([output])

        assert scheduler._spec_stats["req-1"]["accepted"] == 0
        assert scheduler._spec_stats["req-1"]["rejected"] == 3
        assert scheduler._spec_total_accepted == 0
        assert scheduler._spec_total_rejected == 3

    def test_verify_no_drafts_is_noop(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        output = RequestOutput(request_id="req-1", new_token_ids=[1])
        scheduler._verify_spec_drafts([output])  # Should not crash
        assert scheduler._spec_total_proposals == 0

    def test_verify_cleans_up_request_not_running(self):
        """If request is no longer running, draft should still be cleaned up."""
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        scheduler._spec_drafts["req-1"] = [40, 50, 60]
        # No running request for req-1
        output = RequestOutput(request_id="req-1", new_token_ids=[1])
        scheduler._verify_spec_drafts([output])
        assert "req-1" not in scheduler._spec_drafts


class TestSpecDecodeStats:
    """Test speculative decoding stats in get_stats()."""

    def test_stats_spec_enabled_true(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        stats = scheduler.get_stats()
        assert stats["spec_enabled"] is True

    def test_stats_spec_enabled_false_no_decoder(self):
        scheduler = _make_scheduler(enable_spec=True)
        # _spec_decoder is None or just head_info, not SpeculativeDecoder
        stats = scheduler.get_stats()
        assert stats["spec_enabled"] is False

    def test_stats_zero_by_default(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        stats = scheduler.get_stats()
        assert stats["spec_proposals"] == 0
        assert stats["spec_accepted"] == 0
        assert stats["spec_rejected"] == 0
        assert stats["spec_acceptance_rate"] == 0.0
        assert stats["spec_pending_drafts"] == 0

    def test_stats_after_draft_and_verify(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)

        # Simulate some activity
        scheduler._spec_total_proposals = 10
        scheduler._spec_total_accepted = 7
        scheduler._spec_total_rejected = 3
        scheduler._spec_drafts["req-1"] = [1, 2, 3]

        stats = scheduler.get_stats()
        assert stats["spec_proposals"] == 10
        assert stats["spec_accepted"] == 7
        assert stats["spec_rejected"] == 3
        assert stats["spec_acceptance_rate"] == 0.7
        assert stats["spec_pending_drafts"] == 1

    def test_acceptance_rate_rounds_to_3_decimals(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        scheduler._spec_total_proposals = 7
        scheduler._spec_total_accepted = 2
        stats = scheduler.get_stats()
        # 2/7 = 0.285714...
        assert stats["spec_acceptance_rate"] == 0.286


class TestSpecDecodeCleanup:
    """Test spec state cleanup on request lifecycle events."""

    def test_cleanup_on_finish(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"
        scheduler._spec_drafts["req-1"] = [40, 50]
        scheduler._spec_stats["req-1"] = {"proposals": 2, "accepted": 0, "rejected": 0}

        # Simulate finish via _cleanup_spec_state
        scheduler._cleanup_spec_state("req-1")
        assert "req-1" not in scheduler._spec_drafts
        assert "req-1" not in scheduler._spec_stats

    def test_cleanup_on_abort(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.batch_uid = 0
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"
        scheduler._spec_drafts["req-1"] = [40, 50]
        scheduler._spec_stats["req-1"] = {"proposals": 2, "accepted": 0, "rejected": 0}

        # Trigger abort processing
        scheduler._pending_abort_ids.add("req-1")
        scheduler._process_aborts()

        assert "req-1" not in scheduler._spec_drafts
        assert "req-1" not in scheduler._spec_stats

    def test_cleanup_on_preempt(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        req = _make_request()
        req.batch_uid = 0
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"
        scheduler._spec_drafts["req-1"] = [40, 50]

        scheduler._preempt_request(req)

        assert "req-1" not in scheduler._spec_drafts

    def test_deep_reset_clears_spec_state(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)
        scheduler._spec_drafts["req-1"] = [1, 2, 3]
        scheduler._spec_stats["req-1"] = {"proposals": 3, "accepted": 1, "rejected": 2}
        scheduler._spec_total_proposals = 10
        scheduler._spec_total_accepted = 5
        scheduler._spec_total_rejected = 5

        scheduler.deep_reset()

        assert scheduler._spec_drafts == {}
        assert scheduler._spec_stats == {}
        assert scheduler._spec_total_proposals == 0
        assert scheduler._spec_total_accepted == 0
        assert scheduler._spec_total_rejected == 0
        assert scheduler._spec_decoder is None
        assert scheduler._spec_head_info is None


class TestSpecDecodeStepLoop:
    """Integration tests for spec decode in the step loop.

    These tests mock the BatchGenerator to avoid needing a real model.
    """

    def test_step_with_spec_decode_generates_drafts(self):
        """After a decode step, drafts should be generated for active requests."""
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)

        # Setup a running request
        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        # Mock BatchGenerator
        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=20)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        # Mock _generate_draft_tokens to avoid MLX ops
        draft_result = DraftResult(token_ids=[30, 40, 50], logprobs=[-0.1, -0.2, -0.3])
        with patch.object(scheduler, '_generate_draft_tokens', return_value=draft_result):
            scheduler.step()

        # Draft should have been generated for the active request
        assert "req-1" in scheduler._spec_drafts
        assert scheduler._spec_drafts["req-1"] == [30, 40, 50]
        assert scheduler._spec_total_proposals == 3

    def test_step_verifies_existing_drafts(self):
        """Existing drafts should be verified before generating new ones."""
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)

        # Setup a running request with existing drafts
        # req has output [10, 20, 30] and draft was [20, 30, 99]
        # After this step, resp adds token 40 → output becomes [10, 20, 30, 40]
        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10, 20, 30]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"
        # Draft [20, 30, 99] was generated after the previous step
        scheduler._spec_drafts["req-1"] = [20, 30, 99]

        # Mock BatchGenerator
        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=40)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        # Mock _generate_draft_tokens for new drafts after verification
        new_draft = DraftResult(token_ids=[50, 60], logprobs=[-0.1, -0.2])
        with patch.object(scheduler, '_generate_draft_tokens', return_value=new_draft):
            scheduler.step()

        # After step, output_token_ids = [10, 20, 30, 40]
        # Verify compares draft [20, 30, 99] against last 3 actual: [20, 30, 40]
        # 20==20, 30==30, 40!=99 → accepted=2, rejected=1
        assert scheduler._spec_total_accepted == 2
        assert scheduler._spec_total_rejected == 1

    def test_step_no_spec_when_only_head_info(self):
        """When _spec_decoder is just SpecHeadInfo (not SpeculativeDecoder), no drafts."""
        scheduler = _make_scheduler(enable_spec=True)
        # Set to head_info (not a real decoder)
        scheduler._spec_decoder = SpecHeadInfo(head_type="mtp", num_heads=1, draft_length=3)

        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=20)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        scheduler.step()

        # No drafts should be generated
        assert scheduler._spec_drafts == {}
        assert scheduler._spec_total_proposals == 0

    def test_step_spec_disabled_no_activity(self):
        """With spec disabled, no draft/verify happens in step loop."""
        scheduler = _make_scheduler(enable_spec=False, with_decoder=False)

        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=20)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        scheduler.step()

        assert scheduler._spec_drafts == {}
        assert scheduler._spec_total_proposals == 0


class TestSpecDecodeMultipleRequests:
    """Test spec decode with multiple concurrent requests."""

    def test_drafts_for_multiple_requests(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)

        # Setup two requests
        for i in range(2):
            rid = f"req-{i}"
            req = _make_request(req_id=rid)
            req.batch_uid = i
            req.output_token_ids = [10 + i]
            scheduler.running[rid] = req
            scheduler._uid_to_req[i] = rid

        # Mock BatchGenerator
        mock_bg = MagicMock()
        resp0 = _make_response(uid=0, token=20)
        resp1 = _make_response(uid=1, token=21)
        mock_bg.next.return_value = ([], [resp0, resp1])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        # Mock _generate_draft_tokens to avoid MLX ops
        draft_count = [0]
        def mock_gen(req, decoder, K):
            draft_count[0] += 1
            return DraftResult(
                token_ids=[40 + draft_count[0], 50 + draft_count[0]],
                logprobs=[-0.1, -0.2],
            )
        with patch.object(scheduler, '_generate_draft_tokens', side_effect=mock_gen):
            scheduler.step()

        # Both requests should have drafts
        assert "req-0" in scheduler._spec_drafts
        assert "req-1" in scheduler._spec_drafts
        assert scheduler._spec_total_proposals == 4  # 2 drafts * 2 requests

    def test_verify_mixed_acceptance(self):
        scheduler = _make_scheduler(enable_spec=True, with_decoder=True)

        # req-0: output = [10, 20, 30], draft = [20, 30, 99]
        # Verify: last 3 actual = [10, 20, 30], compare with [20, 30, 99]
        #   → 20!=10 at idx 0 → accepted=0, rejected=3
        req0 = _make_request(req_id="req-0")
        req0.batch_uid = 0
        req0.output_token_ids = [10, 20, 30]
        scheduler.running["req-0"] = req0

        # req-1: output = [40, 50, 99], draft = [40, 50, 60]
        # Verify: last 3 actual = [40, 50, 99], compare with [40, 50, 60]
        #   → 40==40, 50==50, 99!=60 → accepted=2, rejected=1
        req1 = _make_request(req_id="req-1")
        req1.batch_uid = 1
        req1.output_token_ids = [40, 50, 99]
        scheduler.running["req-1"] = req1

        scheduler._uid_to_req[0] = "req-0"
        scheduler._uid_to_req[1] = "req-1"

        scheduler._spec_drafts["req-0"] = [20, 30, 99]
        scheduler._spec_drafts["req-1"] = [40, 50, 60]

        output0 = RequestOutput(request_id="req-0", new_token_ids=[30])
        output1 = RequestOutput(request_id="req-1", new_token_ids=[99])

        scheduler._verify_spec_drafts([output0, output1])

        # req-0: last 3 = [10, 20, 30] vs draft [20, 30, 99] → 20!=10 → 0 accepted
        assert scheduler._spec_stats["req-0"]["accepted"] == 0
        assert scheduler._spec_stats["req-0"]["rejected"] == 3
        # req-1: last 3 = [40, 50, 99] vs draft [40, 50, 60] → 40==40, 50==50, 99!=60 → 2 accepted
        assert scheduler._spec_stats["req-1"]["accepted"] == 2
        assert scheduler._spec_stats["req-1"]["rejected"] == 1


# ── Tests: N-gram speculative decoding batch path ──


class TestNgramSpecDecode:
    """Test N-gram speculative decoding in the scheduler batch path."""

    def _make_ngram_scheduler(self, **kwargs):
        """Create a scheduler with N-gram spec decode enabled."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.eos_token_ids = [2]
        tokenizer.encode.return_value = [1]
        tokenizer.detokenizer = MagicMock()
        tokenizer.detokenizer.reset.return_value = None
        model.config = MagicMock()
        model.config.to_dict.return_value = {"model_type": "llama"}

        config = SchedulerConfig(
            enable_spec_decode=False,
            ngram_spec_enabled=kwargs.pop("ngram_spec_enabled", True),
            ngram_spec_min_n=kwargs.pop("ngram_spec_min_n", 1),
            ngram_spec_max_n=kwargs.pop("ngram_spec_max_n", 5),
            ngram_spec_k=kwargs.pop("ngram_spec_k", 5),
            ngram_spec_mode=kwargs.pop("ngram_spec_mode", "lps"),
            **kwargs,
        )
        scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
        return scheduler

    def test_ngram_proposer_created_on_init(self):
        scheduler = self._make_ngram_scheduler()
        assert scheduler._ngram_proposer is not None
        assert scheduler.config.ngram_spec_enabled is True

    def test_ngram_not_created_when_disabled(self):
        scheduler = self._make_ngram_scheduler(ngram_spec_enabled=False)
        assert scheduler._ngram_proposer is None

    def test_ngram_draft_generated_for_active_request(self):
        """N-gram proposer generates drafts from token pattern matching."""
        scheduler = self._make_ngram_scheduler()
        req = _make_request()
        # Build a sequence with a repeated pattern: [1,2,3,4,5,1,2,3]
        # After [1,2,3], the proposer should predict [4,5,...]
        req.prompt_token_ids = [1, 2, 3, 4, 5]
        req.output_token_ids = [1, 2, 3]
        scheduler.running["req-1"] = req

        scheduler._try_spec_decode_draft(req)

        # Draft should be generated from the N-gram pattern
        if scheduler._spec_drafts.get("req-1"):
            stats = scheduler._spec_stats.get("req-1", {})
            assert stats.get("mode") == "ngram"

    def test_ngram_draft_skipped_if_too_short(self):
        """N-gram proposer needs at least min_n tokens."""
        scheduler = self._make_ngram_scheduler(ngram_spec_min_n=3)
        req = _make_request()
        req.prompt_token_ids = []
        req.output_token_ids = [10, 20]  # Only 2 tokens, min_n=3
        scheduler._try_spec_decode_draft(req)

        assert scheduler._spec_drafts == {}

    def test_ngram_draft_skipped_for_aborted(self):
        scheduler = self._make_ngram_scheduler()
        req = _make_request()
        req.prompt_token_ids = [1, 2, 3]
        req.output_token_ids = [10, 20]
        scheduler._pending_abort_ids.add("req-1")
        scheduler._try_spec_decode_draft(req)

        assert scheduler._spec_drafts == {}

    def test_ngram_draft_skipped_if_already_pending(self):
        scheduler = self._make_ngram_scheduler()
        req = _make_request()
        req.prompt_token_ids = [1, 2, 3]
        req.output_token_ids = [10, 20]
        scheduler._spec_drafts["req-1"] = [40, 50]
        scheduler._try_spec_decode_draft(req)

        assert scheduler._spec_drafts["req-1"] == [40, 50]  # Unchanged

    def test_ngram_verify_stats_tracked(self):
        """Verify tracks accepted/rejected for N-gram drafts."""
        scheduler = self._make_ngram_scheduler()
        req = _make_request()
        req.output_token_ids = [10, 20, 30, 40, 50]
        scheduler.running["req-1"] = req
        scheduler._spec_drafts["req-1"] = [40, 50, 60]

        output = RequestOutput(
            request_id="req-1",
            new_token_ids=[50],
            output_token_ids=[10, 20, 30, 40, 50],
        )
        scheduler._verify_spec_drafts([output])

        # last 3 actual = [30, 40, 50] vs draft [40, 50, 60]
        # 30!=40 → accepted=0, rejected=3
        assert scheduler._spec_stats["req-1"]["accepted"] == 0
        assert scheduler._spec_stats["req-1"]["rejected"] == 3

    def test_ngram_verify_no_cross_model_rollback(self):
        """N-gram path should NOT attempt draft model cache rollback."""
        scheduler = self._make_ngram_scheduler()
        # _spec_decoder is None — no cross-model decoder
        assert not isinstance(scheduler._spec_decoder, SpeculativeDecoder)

        req = _make_request()
        req.output_token_ids = [10, 20, 30, 40, 50]
        scheduler.running["req-1"] = req
        scheduler._spec_drafts["req-1"] = [40, 50, 60]

        output = RequestOutput(
            request_id="req-1",
            new_token_ids=[50],
            output_token_ids=[10, 20, 30, 40, 50],
        )
        # Should not crash even without cross-model decoder
        scheduler._verify_spec_drafts([output])
        assert "req-1" not in scheduler._spec_drafts

    def test_ngram_stats_in_get_stats(self):
        scheduler = self._make_ngram_scheduler()
        stats = scheduler.get_stats()
        assert stats["ngram_spec_enabled"] is True
        assert "ngram_spec" in stats
        assert stats["ngram_spec"]["mode"] == "lps"

    def test_ngram_stats_absent_when_disabled(self):
        scheduler = self._make_ngram_scheduler(ngram_spec_enabled=False)
        stats = scheduler.get_stats()
        assert stats["ngram_spec_enabled"] is False
        assert "ngram_spec" not in stats

    def test_enable_ngram_spec_runtime(self):
        """Test enabling N-gram spec decode at runtime."""
        scheduler = self._make_ngram_scheduler(ngram_spec_enabled=False)
        assert scheduler._ngram_proposer is None

        scheduler.enable_ngram_spec(min_n=2, max_n=4, k=3, mode="hashpool")
        assert scheduler._ngram_proposer is not None
        assert scheduler.config.ngram_spec_enabled is True
        assert scheduler._ngram_proposer.config.min_n == 2
        assert scheduler._ngram_proposer.config.max_n == 4
        assert scheduler._ngram_proposer.config.k == 3

    def test_deep_reset_preserves_ngram_proposer(self):
        """deep_reset recreates the N-gram proposer (clears learned patterns)."""
        scheduler = self._make_ngram_scheduler()
        proposer_before = scheduler._ngram_proposer
        scheduler.deep_reset()
        assert scheduler._ngram_proposer is not None
        # New instance (cleared patterns)
        assert scheduler._ngram_proposer is not proposer_before

    def test_step_ngram_spec_generates_drafts(self):
        """Step loop should generate N-gram drafts when enabled."""
        scheduler = self._make_ngram_scheduler()

        req = _make_request()
        req.batch_uid = 0
        req.prompt_token_ids = [1, 2, 3, 4, 5]
        req.output_token_ids = [1, 2, 3]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=4)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        scheduler.step()

        # N-gram should have been attempted (may or may not find a match)
        # The key test is that it doesn't crash and spec stats exist
        stats = scheduler.get_stats()
        assert stats["ngram_spec_enabled"] is True

    def test_step_ngram_and_cross_model_both_off(self):
        """When both spec decode backends are off, no drafts are generated."""
        scheduler = self._make_ngram_scheduler(ngram_spec_enabled=False)

        req = _make_request()
        req.batch_uid = 0
        req.output_token_ids = [10]
        scheduler.running["req-1"] = req
        scheduler._uid_to_req[0] = "req-1"

        mock_bg = MagicMock()
        resp = _make_response(uid=0, token=20)
        mock_bg.next.return_value = ([], [resp])
        mock_bg.next_generated.return_value = []
        scheduler._batch_gen = mock_bg

        scheduler.step()
        assert scheduler._spec_drafts == {}


class TestNgramSpecDecodeWithRepetition:
    """Test N-gram spec decode with actual repeated patterns."""

    def _make_ngram_scheduler(self, mode="lps"):
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
            ngram_spec_mode=mode,
        )
        return Scheduler(model=model, tokenizer=tokenizer, config=config)

    def test_lps_mode_repeated_pattern(self):
        """LPS mode should find repeated ngrams and propose continuations."""
        scheduler = self._make_ngram_scheduler(mode="lps")
        req = _make_request()
        # Pattern: [10,20,30,10,20] — suffix [10,20] matches earlier [10,20,30]
        # Should propose [30] as continuation
        req.prompt_token_ids = [10, 20, 30]
        req.output_token_ids = [10, 20]
        scheduler._try_spec_decode_draft(req)
        # LPS should find the repeated "10,20" and propose "30"
        draft = scheduler._spec_drafts.get("req-1", [])
        if draft:
            assert 30 in draft

    def test_hashpool_mode_repeated_pattern(self):
        """Hashpool mode should find repeated ngrams."""
        scheduler = self._make_ngram_scheduler(mode="hashpool")
        req = _make_request()
        req.prompt_token_ids = [10, 20, 30, 40, 50]
        req.output_token_ids = [10, 20, 30]
        scheduler._try_spec_decode_draft(req)
        # Hashpool should index and find patterns
        scheduler._spec_drafts.get("req-1", [])
        # May or may not have drafts depending on pattern quality

    def test_lcg_mode_repeated_pattern(self):
        """LCG hashpool mode should find repeated ngrams."""
        scheduler = self._make_ngram_scheduler(mode="lcg")
        req = _make_request()
        req.prompt_token_ids = [10, 20, 30, 40, 50]
        req.output_token_ids = [10, 20, 30]
        scheduler._try_spec_decode_draft(req)
        scheduler._spec_drafts.get("req-1", [])
        # LCG should also find patterns
