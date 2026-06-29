"""Tests for streaming_optimizer — TokenPipeline, PrefetchSampler,
BatchedDetokenizer, StreamingBackpressureController.

Run: uv run pytest tests/unit/test_streaming_optimizer.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from yunshu_engine.streaming_optimizer import (
    BackpressureConfig,
    BatchedDetokenizer,
    PipelineConfig,
    PipelineStage,
    PipelineToken,
    PrefetchSampler,
    SamplingPlan,
    StreamingBackpressureController,
    TokenPipeline,
    _softmax,
)

# ===================================================================
# TokenPipeline tests
# ===================================================================


class TestPipelineToken:
    """Tests for PipelineToken data class."""

    def test_default_state(self):
        tok = PipelineToken()
        assert tok.token_id == -1
        assert tok.text == ""
        assert tok.stage == PipelineStage.IDLE
        assert tok.total_latency_ms == 0.0
        assert tok.overlap_savings_ms == 0.0

    def test_mark_enter(self):
        tok = PipelineToken()
        tok.mark_enter()
        assert tok.timestamp_enter > 0
        assert tok.stage == PipelineStage.IDLE

    def test_stage_transitions(self):
        tok = PipelineToken()
        tok.mark_enter()
        tok.mark_stage1_done()
        assert tok.stage == PipelineStage.GPU_SAMPLING
        assert tok.timestamp_stage1_done > 0

        tok.mark_stage2_done()
        assert tok.stage == PipelineStage.CPU_POST
        assert tok.timestamp_stage2_done > 0

        tok.mark_stage3_done()
        assert tok.stage == PipelineStage.IDLE
        assert tok.timestamp_stage3_done > 0

    def test_latency_tracking(self):
        tok = PipelineToken()
        tok.mark_enter()
        time.sleep(0.001)
        tok.mark_stage1_done()
        time.sleep(0.001)
        tok.mark_stage2_done()
        time.sleep(0.001)
        tok.mark_stage3_done()
        assert tok.total_latency_ms > 0
        assert tok.overlap_savings_ms > 0


class TestTokenPipelineInit:
    """Tests for TokenPipeline initialization and lifecycle."""

    def test_default_config(self):
        p = TokenPipeline()
        assert p.config.enable_overlap is True
        assert p.config.pipeline_depth == 2
        assert not p.is_running
        assert not p.is_finished

    def test_custom_config(self):
        cfg = PipelineConfig(
            enable_overlap=False,
            pipeline_depth=4,
            async_eval=False,
        )
        p = TokenPipeline(cfg)
        assert p.config.enable_overlap is False
        assert p.config.pipeline_depth == 4

    def test_start_pipeline(self):
        p = TokenPipeline()
        p.start_pipeline()
        assert p.is_running
        assert not p.is_finished
        assert p.tokens_generated == 0

    def test_stop_pipeline(self):
        p = TokenPipeline()
        p.start_pipeline()
        p.stop()
        assert not p.is_running

    def test_finish_pipeline(self):
        p = TokenPipeline()
        p.start_pipeline()
        p.finish()
        assert p.is_finished
        assert not p.is_running


class TestTokenPipelineStages:
    """Tests for TokenPipeline stage submission."""

    def test_submit_stage1_result(self):
        p = TokenPipeline()
        p.start_pipeline()
        tok = p.submit_stage1_result(logits=None, token_id=42)
        assert tok.token_id == -1  # stage1 doesn't set token_id
        assert tok.stage == PipelineStage.GPU_SAMPLING
        assert p.tokens_generated == 1

    def test_submit_stage2_result(self):
        p = TokenPipeline()
        p.start_pipeline()
        tok = p.submit_stage1_result(logits=None)
        tok = p.submit_stage2_result(tok, sampled_id=42)
        assert tok.token_id == 42
        assert tok.stage == PipelineStage.CPU_POST

    @pytest.mark.asyncio
    async def test_next_token_without_overlap(self):
        p = TokenPipeline(PipelineConfig(enable_overlap=False))
        p.start_pipeline()
        tok = p.submit_stage1_result(logits=None)
        tok = p.submit_stage2_result(tok, sampled_id=99)

        result = await p.next_token()
        assert result is not None
        assert result.token_id == 99
        assert result.stage == PipelineStage.IDLE
        assert p.tokens_yielded == 1

    @pytest.mark.asyncio
    async def test_next_token_with_sync_detokenize(self):
        p = TokenPipeline(PipelineConfig(enable_overlap=False))
        p.start_pipeline()
        tok = p.submit_stage1_result(logits=None)
        tok = p.submit_stage2_result(tok, sampled_id=10)

        def detok(tid):
            return f"token_{tid}"

        result = await p.next_token(detokenize_fn=detok)
        assert result.text == "token_10"

    @pytest.mark.asyncio
    async def test_next_token_returns_none_when_not_running(self):
        p = TokenPipeline()
        result = await p.next_token()
        assert result is None

    @pytest.mark.asyncio
    async def test_pipeline_stats(self):
        p = TokenPipeline(PipelineConfig(enable_overlap=False))
        p.start_pipeline()
        tok = p.submit_stage1_result(logits=None)
        tok = p.submit_stage2_result(tok, sampled_id=1)
        await p.next_token()

        stats = p.get_stats()
        assert stats["tokens_generated"] == 1
        assert stats["tokens_yielded"] == 1
        assert stats["overlap_enabled"] is False
        assert stats["elapsed_s"] >= 0  # may round to 0 in fast tests


# ===================================================================
# PrefetchSampler tests
# ===================================================================


class TestSamplingPlan:
    """Tests for SamplingPlan."""

    def test_deterministic_plan(self):
        plan = SamplingPlan(deterministic=True)
        assert plan.deterministic is True

    def test_stochastic_plan(self):
        plan = SamplingPlan(deterministic=False, temperature=0.7, top_p=0.9, top_k=50)
        assert plan.deterministic is False
        assert plan.temperature == 0.7
        assert plan.top_p == 0.9
        assert plan.top_k == 50

    def test_deterministic_apply_argmax(self):
        plan = SamplingPlan(deterministic=True)
        logits = np.array([0.1, 0.5, 0.9, 0.3])
        result = plan.apply(logits)
        assert result[0] == 2  # argmax index

    def test_stochastic_apply_produces_valid_token(self):
        plan = SamplingPlan(
            deterministic=False,
            temperature=1.0,
            seed=42,
        )
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        result = plan.apply(logits)
        assert 0 <= result[0] < 4

    def test_temperature_scaling(self):
        plan = SamplingPlan(deterministic=False, temperature=0.5, seed=42)
        logits = np.array([0.0, 1.0, 2.0, 3.0])
        # With low temperature, should strongly prefer the highest logit
        results = [plan.apply(logits)[0] for _ in range(20)]
        # Most samples should be index 3
        assert results.count(3) >= 15

    def test_prepare_random_noop_for_deterministic(self):
        plan = SamplingPlan(deterministic=True)
        plan.prepare_random(vocab_size=1000)
        assert plan._precomputed_gumbel is None

    def test_prepare_random_stochastic(self):
        plan = SamplingPlan(deterministic=False, seed=42)
        plan.prepare_random(vocab_size=100)
        assert plan._precomputed_gumbel is not None
        assert len(plan._precomputed_gumbel) == 100

    def test_top_k_filtering(self):
        plan = SamplingPlan(deterministic=False, temperature=1.0, top_k=2, seed=42)
        logits = np.array([0.1, 5.0, 0.2, 4.9])
        # With top_k=2, only indices 1 and 3 should be sampled
        results = [plan.apply(logits)[0] for _ in range(50)]
        assert all(r in [1, 3] for r in results)

    def test_top_p_filtering(self):
        plan = SamplingPlan(deterministic=False, temperature=1.0, top_p=0.5, seed=42)
        logits = np.array([0.0, 0.0, 10.0, 0.0])
        # With top_p=0.5, nearly all probability on index 2
        results = [plan.apply(logits)[0] for _ in range(50)]
        assert results.count(2) >= 45


class TestPrefetchSampler:
    """Tests for PrefetchSampler."""

    def test_prepare_returns_plan(self):
        ps = PrefetchSampler()
        plan = ps.prepare(temperature=0.0)
        assert isinstance(plan, SamplingPlan)
        assert plan.deterministic is True

    def test_prepare_with_dict(self):
        ps = PrefetchSampler()
        plan = ps.prepare(
            sampling_params={"temperature": 0.8, "top_k": 10},
            vocab_size=32000,
        )
        assert plan.temperature == 0.8
        assert plan.top_k == 10

    def test_apply_deterministic(self):
        ps = PrefetchSampler()
        ps.prepare(temperature=0.0)
        logits = np.array([0.1, 0.5, 0.9, 0.3])
        result = ps.apply(logits)
        assert result[0] == 2

    def test_apply_without_prepare_raises(self):
        ps = PrefetchSampler()
        with pytest.raises(RuntimeError, match="prepare"):
            ps.apply(np.array([0.1, 0.2]))

    def test_keywords_override_dict(self):
        ps = PrefetchSampler()
        plan = ps.prepare(
            sampling_params={"temperature": 0.8},
            temperature=0.0,
        )
        assert plan.deterministic is True  # keyword took precedence

    def test_stats_tracking(self):
        ps = PrefetchSampler()
        ps.prepare(temperature=0.0)
        ps.apply(np.array([0.1, 0.5, 0.9]))
        stats = ps.get_stats()
        assert stats["prepare_count"] == 1
        assert stats["apply_count"] == 1
        assert stats["avg_prepare_ms"] >= 0
        assert stats["avg_apply_ms"] >= 0

    def test_reset(self):
        ps = PrefetchSampler()
        ps.prepare(temperature=0.0)
        ps.apply(np.array([0.1, 0.5, 0.9]))
        ps.reset()
        assert ps.prepare_count == 0
        assert ps.apply_count == 0
        assert ps.plan is None

    def test_sequential_prepare_apply(self):
        """Verify prepare/apply works across multiple steps."""
        ps = PrefetchSampler()
        logits_list = [
            np.array([0.1, 0.9, 0.5]),
            np.array([0.3, 0.2, 0.8]),
            np.array([0.7, 0.1, 0.3]),
        ]
        tokens = []
        for logits in logits_list:
            ps.prepare(temperature=0.0)
            tok = ps.apply(logits)
            tokens.append(tok[0])
        assert tokens == [1, 2, 0]

    def test_vocab_size_precompute(self):
        ps = PrefetchSampler()
        plan = ps.prepare(temperature=0.7, vocab_size=5000, seed=123)
        assert plan._precomputed_gumbel is not None
        assert len(plan._precomputed_gumbel) == 5000


# ===================================================================
# BatchedDetokenizer tests
# ===================================================================


class FakeTokenizer:
    """Simple tokenizer for testing that joins token ids with space."""

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"t{id}" for id in token_ids)


class TestBatchedDetokenizer:
    """Tests for BatchedDetokenizer."""

    def test_empty_flush(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        result = bd.flush()
        assert result == {}
        assert bd.flush_count == 1

    def test_add_and_flush_single_request(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1, 2, 3])
        result = bd.flush()
        assert result["req-1"] == "t1 t2 t3"

    def test_add_and_flush_multiple_requests(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1, 2])
        bd.add_tokens("req-2", [3, 4])
        bd.add_tokens("req-3", [5])
        result = bd.flush()
        assert len(result) == 3
        assert result["req-1"] == "t1 t2"
        assert result["req-2"] == "t3 t4"
        assert result["req-3"] == "t5"

    def test_get_segment_after_flush(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [10, 20])
        bd.flush()
        text = bd.get_segment("req-1")
        assert text == "t10 t20"

    def test_get_segment_pops(self):
        """get_segment removes the segment from cache."""
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1])
        bd.flush()
        _ = bd.get_segment("req-1")
        assert bd.get_segment("req-1") == ""

    def test_get_segment_not_found(self):
        bd = BatchedDetokenizer()
        assert bd.get_segment("nonexistent") == ""

    def test_has_segment(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        assert not bd.has_segment("req-1")
        bd.add_tokens("req-1", [1])
        bd.flush()
        assert bd.has_segment("req-1")
        bd.get_segment("req-1")
        assert not bd.has_segment("req-1")

    def test_add_tokens_merges_before_flush(self):
        """Multiple add_tokens for same request before flush merges tokens."""
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1, 2])
        bd.add_tokens("req-1", [3, 4])
        result = bd.flush()
        assert result["req-1"] == "t1 t2 t3 t4"

    def test_add_empty_tokens_noop(self):
        bd = BatchedDetokenizer()
        bd.add_tokens("req-1", [])
        assert bd.pending_count == 0

    def test_no_tokenizer_returns_empty(self):
        bd = BatchedDetokenizer()
        bd.add_tokens("req-1", [1, 2])
        result = bd.flush()
        assert result["req-1"] == ""

    def test_clear(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1])
        bd.clear()
        assert bd.pending_count == 0
        assert not bd.has_segment("req-1")

    def test_pending_count(self):
        bd = BatchedDetokenizer()
        assert bd.pending_count == 0
        bd.add_tokens("req-1", [1])
        assert bd.pending_count == 1
        bd.add_tokens("req-2", [2])
        assert bd.pending_count == 2
        bd.flush()
        assert bd.pending_count == 0

    def test_stats(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1, 2, 3])
        bd.flush()
        stats = bd.get_stats()
        assert stats["flush_count"] == 1
        assert stats["total_tokens_processed"] == 3
        assert stats["avg_flush_ms"] >= 0

    def test_multiple_flushes(self):
        bd = BatchedDetokenizer(FakeTokenizer())
        bd.add_tokens("req-1", [1])
        bd.flush()
        bd.add_tokens("req-2", [2, 3])
        bd.flush()
        assert bd.flush_count == 2
        assert bd.total_tokens_processed == 3


# ===================================================================
# StreamingBackpressureController tests
# ===================================================================


class TestBackpressureConfig:
    """Tests for BackpressureConfig."""

    def test_defaults(self):
        cfg = BackpressureConfig()
        assert cfg.max_queue_size == 100
        assert cfg.initial_delay_ms == 1.0
        assert cfg.max_delay_ms == 50.0
        assert cfg.ramp_factor == 0.5
        assert cfg.cooldown_factor == 0.9

    def test_custom(self):
        cfg = BackpressureConfig(
            max_queue_size=200,
            initial_delay_ms=2.0,
            max_delay_ms=100.0,
        )
        assert cfg.max_queue_size == 200
        assert cfg.initial_delay_ms == 2.0
        assert cfg.max_delay_ms == 100.0


class TestStreamingBackpressureController:
    """Tests for StreamingBackpressureController."""

    def test_no_backpressure_below_threshold(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        assert not bpc.check_backpressure(50)
        assert not bpc.check_backpressure(99)

    def test_backpressure_at_threshold(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        assert bpc.check_backpressure(100)
        assert bpc.check_backpressure(150)

    def test_delay_at_threshold(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        delay = bpc.get_delay_ms(100)
        assert delay >= bpc.config.initial_delay_ms

    def test_delay_ramps_up(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        d1 = bpc.get_delay_ms(100)
        d2 = bpc.get_delay_ms(110)
        d3 = bpc.get_delay_ms(120)
        assert d1 <= d2  # delay increases with excess
        # d3 should be at least d2 due to max(ramp, current)
        assert d3 >= d2

    def test_delay_capped_at_max(self):
        bpc = StreamingBackpressureController(
            max_queue_size=100,
            config=BackpressureConfig(
                max_queue_size=100,
                max_delay_ms=10.0,
                ramp_factor=1.0,
            ),
        )
        delay = bpc.get_delay_ms(1000)
        assert delay <= 10.0

    def test_cooldown_below_threshold(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        # First, go above threshold to establish delay
        bpc.get_delay_ms(100)
        assert bpc.current_delay_ms > 0
        # Now drop below — delay should decay
        bpc.get_delay_ms(50)
        delay_after = bpc.current_delay_ms
        bpc.get_delay_ms(50)
        assert bpc.current_delay_ms < delay_after

    def test_cooldown_to_zero(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        bpc.get_delay_ms(100)
        # Decay many times
        for _ in range(100):
            bpc.get_delay_ms(0)
        assert bpc.current_delay_ms == 0.0

    def test_custom_max_queue_override(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        assert bpc.check_backpressure(50, max_queue=50)
        assert not bpc.check_backpressure(50, max_queue=60)

    def test_stats(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        bpc.check_backpressure(100)
        bpc.get_delay_ms(100)
        stats = bpc.get_stats()
        assert stats["backpressure_count"] == 1
        assert stats["max_queue_size"] == 100
        assert stats["max_queue_seen"] == 100
        assert stats["current_delay_ms"] >= 0

    def test_reset(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        bpc.get_delay_ms(100)
        bpc.reset()
        assert bpc.current_delay_ms == 0.0
        assert bpc.backpressure_count == 0
        assert bpc.max_queue_seen == 0

    def test_max_queue_seen_tracking(self):
        bpc = StreamingBackpressureController(max_queue_size=100)
        bpc.check_backpressure(50)
        assert bpc.max_queue_seen == 50
        bpc.check_backpressure(75)
        assert bpc.max_queue_seen == 75
        bpc.check_backpressure(30)
        assert bpc.max_queue_seen == 75  # doesn't decrease


# ===================================================================
# Softmax helper test
# ===================================================================


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        probs = _softmax(x)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_large_values_no_overflow(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        probs = _softmax(x)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_uniform_input(self):
        x = np.array([2.0, 2.0, 2.0])
        probs = _softmax(x)
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)
