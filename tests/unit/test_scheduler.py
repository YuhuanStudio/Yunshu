"""Tests for Yunshu Scheduler — sampler wiring, repetition penalty, and penalty processors."""

import pytest
from unittest.mock import patch, MagicMock
import mlx.core as mx

from yunshu_engine.scheduler import Scheduler, SchedulerConfig, _LogitsProcessorSampler
from yunshu_engine.request import SamplingParams


class TestLogitsProcessorSampler:
    """Test the _LogitsProcessorSampler wrapper that applies logits processors."""

    def test_no_processors_passthrough(self):
        """Without processors, the base sampler is called directly."""
        calls = []
        def base_sampler(logits):
            calls.append(logits)
            return mx.array([42])

        sampler = _LogitsProcessorSampler(base_sampler, [])
        logits = mx.zeros((1, 100))
        result = sampler(logits)
        assert len(calls) == 1
        assert result.item() == 42

    def test_processor_receives_tokens(self):
        """Logits processors receive accumulated token list."""
        tokens_seen = []
        def fake_processor(tokens, logits):
            tokens_seen.append(list(tokens))
            return logits

        def base_sampler(logits):
            return mx.array([7])

        sampler = _LogitsProcessorSampler(base_sampler, [fake_processor])
        logits = mx.zeros((1, 100))

        # First call: no tokens yet
        sampler(logits)
        assert tokens_seen[-1] == []

        # Second call: should have previous token
        sampler(logits)
        assert tokens_seen[-1] == [7]

        # Third call: two tokens accumulated
        sampler(logits)
        assert tokens_seen[-1] == [7, 7]

    def test_reset_clears_tokens(self):
        """reset() clears accumulated tokens."""
        def base_sampler(logits):
            return mx.array([1])

        sampler = _LogitsProcessorSampler(base_sampler, [])
        sampler(mx.zeros((1, 100)))
        sampler(mx.zeros((1, 100)))
        assert len(sampler._tokens) == 2

        sampler.reset()
        assert len(sampler._tokens) == 0


class TestMakeSamplerRepetitionPenalty:
    """Test that _make_sampler wires repetition_penalty correctly."""

    @pytest.fixture
    def scheduler(self):
        """Create a Scheduler with mock model/tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.eos_token_ids = [2]
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.has_thinking = False
        config = SchedulerConfig(model_name="test-model")
        return Scheduler(model, tokenizer, config)

    def test_default_repetition_penalty_creates_base_sampler(self, scheduler):
        """Default repetition_penalty=1.0 should not create a processor wrapper."""
        sp = SamplingParams(repetition_penalty=1.0)
        sampler = scheduler._make_sampler(sp)
        # Should be a plain sampler function, not _LogitsProcessorSampler
        assert not isinstance(sampler, _LogitsProcessorSampler)

    def test_repetition_penalty_creates_processor(self, scheduler):
        """repetition_penalty != 1.0 should create _LogitsProcessorSampler."""
        sp = SamplingParams(repetition_penalty=1.5)
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 1

    def test_presence_penalty_creates_processor(self, scheduler):
        """presence_penalty != 0 should create _LogitsProcessorSampler."""
        sp = SamplingParams(presence_penalty=0.5)
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 1

    def test_frequency_penalty_creates_processor(self, scheduler):
        """frequency_penalty != 0 should create _LogitsProcessorSampler."""
        sp = SamplingParams(frequency_penalty=0.3)
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 1

    def test_all_penalties_creates_three_processors(self, scheduler):
        """All three penalties active should create 3 processors."""
        sp = SamplingParams(
            repetition_penalty=1.2,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 3

    def test_repetition_penalty_modifies_logits(self, scheduler):
        """Verify the repetition penalty actually modifies logits for repeated tokens."""
        sp = SamplingParams(
            repetition_penalty=2.0,
            temperature=1.0,
        )
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)

        # Create logits where token 5 has high score
        logits = mx.zeros((1, 100))
        logits[:, 5] = 10.0

        # First call: no tokens yet, token 5 should be selected
        result = sampler(logits)
        mx.eval(result)
        assert result.item() == 5

        # Second call with same logits: token 5 is now penalized
        # With repetition_penalty=2.0, positive logits are divided by 2
        # So token 5's logit becomes 5.0 instead of 10.0
        logits2 = mx.zeros((1, 100))
        logits2[:, 5] = 10.0
        logits2[:, 10] = 8.0  # Token 10 should now be preferred
        result2 = sampler(logits2)
        mx.eval(result2)
        # After penalty: token 5 = 5.0, token 10 = 8.0, so token 10 should win
        assert result2.item() == 10


class TestSchedulerStats:
    """Test scheduler stats reporting."""

    def test_initial_stats(self):
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(model_name="test-model")
        sched = Scheduler(model, tokenizer, config)
        stats = sched.get_stats()
        assert stats["waiting"] == 0
        assert stats["running"] == 0
        assert stats["total_requests"] == 0
        assert stats["step_counter"] == 0


class TestRequestTimeout:
    def test_timeout_status_exists(self):
        from yunshu_engine.request import RequestStatus
        assert hasattr(RequestStatus, "FINISHED_TIMEOUT")

    def test_timeout_finish_reason(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_TIMEOUT) == "timeout"

    def test_timeout_is_finished(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.is_finished(RequestStatus.FINISHED_TIMEOUT) is True

    def test_config_has_timeout(self):
        from yunshu_engine.scheduler import SchedulerConfig
        config = SchedulerConfig()
        assert config.request_timeout_seconds == 300

    def test_config_custom_timeout(self):
        from yunshu_engine.scheduler import SchedulerConfig
        config = SchedulerConfig(request_timeout_seconds=60)
        assert config.request_timeout_seconds == 60


class TestSeedParameter:
    def test_seed_in_sampling_params(self):
        from yunshu_engine.request import SamplingParams
        sp = SamplingParams(seed=42)
        assert sp.seed == 42

    def test_seed_default_none(self):
        from yunshu_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.seed is None


class TestQueueDepthLimit:
    def test_reject_when_queue_full(self):
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request, SamplingParams, RequestStatus

        config = SchedulerConfig(max_waiting_requests=2)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = config
        scheduler.requests = {}
        scheduler.waiting = type('deque', (), {'append': lambda s, x: None, '__len__': lambda s: 2})()
        scheduler.running = {}
        scheduler._pending_abort_ids = set()

        req = Request(request_id="overflow-1", prompt="test", prompt_token_ids=[1,2,3], sampling_params=SamplingParams())
        scheduler.add_request(req)
        assert req.status == RequestStatus.FINISHED_ERROR
        assert req.finish_reason == "queue_full"

    def test_max_waiting_config_default(self):
        config = SchedulerConfig()
        assert config.max_waiting_requests == 1024


class TestSarathiChunkedPrefill:
    """Test Sarathi-style hybrid chunked prefill scheduling."""

    def test_config_defaults(self):
        """Hybrid prefill config should have sensible defaults."""
        config = SchedulerConfig()
        assert config.hybrid_chunk_size == 512
        assert config.enable_hybrid_prefill is False

    def test_config_custom(self):
        """Hybrid prefill config should be customizable."""
        config = SchedulerConfig(
            enable_hybrid_prefill=True,
            hybrid_chunk_size=256,
        )
        assert config.hybrid_chunk_size == 256
        assert config.enable_hybrid_prefill is True

    def test_pending_prefill_initialized(self):
        """Scheduler should have _pending_prefill dict."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        assert hasattr(sched, '_pending_prefill')
        assert isinstance(sched._pending_prefill, dict)
        assert len(sched._pending_prefill) == 0

    def test_pending_prefill_cleared_on_deep_reset(self):
        """deep_reset should clear _pending_prefill."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        sched._pending_prefill["test-id"] = {"remaining_tokens": [1, 2, 3]}
        sched.deep_reset()
        assert len(sched._pending_prefill) == 0

    def test_process_pending_prefill_noop_when_empty(self):
        """_process_pending_prefill should be a no-op when no pending prefills."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        # Should not raise
        sched._process_pending_prefill()

    def test_process_pending_prefill_removes_empty(self):
        """_process_pending_prefill should remove entries with no remaining tokens."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        sched._pending_prefill["req-1"] = {"remaining_tokens": []}
        sched._process_pending_prefill()
        assert "req-1" not in sched._pending_prefill

    def test_process_pending_prefill_removes_missing_running(self):
        """_process_pending_prefill should remove entries for non-running requests."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        sched._pending_prefill["req-1"] = {"remaining_tokens": [1, 2, 3]}
        # req-1 is not in running dict, so it should be cleaned up
        sched._process_pending_prefill()
        assert "req-1" not in sched._pending_prefill
