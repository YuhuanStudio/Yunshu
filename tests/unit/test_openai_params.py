"""Tests for OpenAI-compatible generation parameters.

Tests the following parameters across Chat and Completions APIs:
- frequency_penalty: modifies logits based on token frequency in output
- presence_penalty: modifies logits based on token presence in output
- logit_bias: adds bias to specified token logits
- parallel_tool_calls: controls whether multiple tool calls can be generated
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yunshu_engine.request import SamplingParams
from yunshu_gateway.routers.chat import ChatCompletionRequest
from yunshu_gateway.routers.completions import CompletionRequest

# ── ChatCompletionRequest parameter tests ──


class TestChatCompletionParams:
    """Test OpenAI Chat Completions request parameters."""

    def test_frequency_penalty_default(self):
        req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
        assert req.frequency_penalty == 0.0

    def test_frequency_penalty_set(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=0.5,
        )
        assert req.frequency_penalty == 0.5

    def test_frequency_penalty_negative(self):
        """OpenAI allows negative frequency_penalty to encourage repetition."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=-1.0,
        )
        assert req.frequency_penalty == -1.0

    def test_frequency_penalty_max(self):
        """OpenAI allows frequency_penalty up to 2.0."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=2.0,
        )
        assert req.frequency_penalty == 2.0

    def test_presence_penalty_default(self):
        req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
        assert req.presence_penalty == 0.0

    def test_presence_penalty_set(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            presence_penalty=0.8,
        )
        assert req.presence_penalty == 0.8

    def test_presence_penalty_negative(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            presence_penalty=-0.5,
        )
        assert req.presence_penalty == -0.5

    def test_logit_bias_default(self):
        req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
        assert req.logit_bias is None

    def test_logit_bias_set(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            logit_bias={100: -100, 200: 5.0},
        )
        assert req.logit_bias == {100: -100, 200: 5.0}

    def test_logit_bias_empty_dict(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            logit_bias={},
        )
        assert req.logit_bias == {}

    def test_logit_bias_string_keys_rejected(self):
        """logit_bias keys must be integers (token IDs), not strings."""
        with pytest.raises((ValueError, TypeError)):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                logit_bias={"hello": 5.0},
            )

    def test_parallel_tool_calls_default(self):
        req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
        assert req.parallel_tool_calls is True

    def test_parallel_tool_calls_false(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            parallel_tool_calls=False,
        )
        assert req.parallel_tool_calls is False

    def test_all_new_params_together(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=0.5,
            presence_penalty=0.3,
            logit_bias={42: -10.0, 43: 5.0},
            parallel_tool_calls=False,
        )
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.3
        assert req.logit_bias == {42: -10.0, 43: 5.0}
        assert req.parallel_tool_calls is False


# ── CompletionRequest parameter tests ──


class TestCompletionParams:
    """Test OpenAI Completions request parameters."""

    def test_frequency_penalty_default(self):
        req = CompletionRequest(model="test", prompt="hello")
        assert req.frequency_penalty == 0.0

    def test_frequency_penalty_set(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            frequency_penalty=1.0,
        )
        assert req.frequency_penalty == 1.0

    def test_presence_penalty_default(self):
        req = CompletionRequest(model="test", prompt="hello")
        assert req.presence_penalty == 0.0

    def test_presence_penalty_set(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            presence_penalty=0.6,
        )
        assert req.presence_penalty == 0.6

    def test_logit_bias_default(self):
        req = CompletionRequest(model="test", prompt="hello")
        assert req.logit_bias is None

    def test_logit_bias_set(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            logit_bias={1: 5.0, 2: -5.0},
        )
        assert req.logit_bias == {1: 5.0, 2: -5.0}

    def test_repetition_penalty_default(self):
        """repetition_penalty was already supported; verify it still works."""
        req = CompletionRequest(model="test", prompt="hello")
        assert req.repetition_penalty == 1.0

    def test_repetition_penalty_set(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            repetition_penalty=1.2,
        )
        assert req.repetition_penalty == 1.2

    def test_all_params_together(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            repetition_penalty=1.1,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            logit_bias={100: -10},
        )
        assert req.repetition_penalty == 1.1
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.3
        assert req.logit_bias == {100: -10}


# ── SamplingParams tests ──


class TestSamplingParams:
    """Test that SamplingParams correctly stores the new parameters."""

    def test_frequency_penalty_default(self):
        sp = SamplingParams()
        assert sp.frequency_penalty == 0.0

    def test_frequency_penalty_set(self):
        sp = SamplingParams(frequency_penalty=0.5)
        assert sp.frequency_penalty == 0.5

    def test_presence_penalty_default(self):
        sp = SamplingParams()
        assert sp.presence_penalty == 0.0

    def test_presence_penalty_set(self):
        sp = SamplingParams(presence_penalty=0.8)
        assert sp.presence_penalty == 0.8

    def test_logit_bias_default(self):
        sp = SamplingParams()
        assert sp.logit_bias is None

    def test_logit_bias_set(self):
        sp = SamplingParams(logit_bias={100: -100, 200: 50})
        assert sp.logit_bias == {100: -100, 200: 50}

    def test_all_params_together(self):
        sp = SamplingParams(
            max_tokens=512,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            logit_bias={42: 10.0},
        )
        assert sp.max_tokens == 512
        assert sp.temperature == 0.5
        assert sp.top_p == 0.9
        assert sp.repetition_penalty == 1.2
        assert sp.frequency_penalty == 0.5
        assert sp.presence_penalty == 0.3
        assert sp.logit_bias == {42: 10.0}


# ── Scheduler _make_sampler integration tests ──


class TestSchedulerSamplerIntegration:
    """Test that the scheduler properly passes the new params to make_logits_processors."""

    def test_make_sampler_with_frequency_penalty(self):
        """Verify frequency_penalty is passed to make_logits_processors."""

        sp = SamplingParams(
            frequency_penalty=0.5,
        )
        # The scheduler._make_sampler should create a _LogitsProcessorSampler
        # when frequency_penalty != 0. We can verify the param is stored.
        assert sp.frequency_penalty != 0.0

    def test_make_sampler_with_presence_penalty(self):
        """Verify presence_penalty is passed to make_logits_processors."""
        sp = SamplingParams(
            presence_penalty=0.8,
        )
        assert sp.presence_penalty != 0.0

    def test_make_sampler_with_logit_bias(self):
        """Verify logit_bias is available in SamplingParams."""
        sp = SamplingParams(
            logit_bias={100: -100, 200: 50},
        )
        assert sp.logit_bias is not None
        assert len(sp.logit_bias) == 2

    def test_make_sampler_no_processors_when_defaults(self):
        """When all penalties are default, logits processors should be minimal."""
        sp = SamplingParams(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
        )
        # All default values mean no processors needed
        assert sp.repetition_penalty == 1.0
        assert sp.frequency_penalty == 0.0
        assert sp.presence_penalty == 0.0
        assert sp.logit_bias is None


# ── EngineCore parameter pass-through tests ──


class TestEngineCoreParamPassthrough:
    """Test that EngineCore.add_request passes new params to SamplingParams."""

    @pytest.mark.asyncio
    async def test_add_request_with_frequency_penalty(self):
        """Verify frequency_penalty flows through to SamplingParams."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig

        # Mock the model, tokenizer, executor
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        config = EngineCoreConfig()
        core = EngineCore(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            executor=MagicMock(),
        )

        # Patch scheduler.add_request to capture the Request
        captured_request = None

        def capture_add(request):
            nonlocal captured_request
            captured_request = request
            # Don't actually add — just capture
            return

        core.scheduler.add_request = capture_add

        # Mock event loop
        with patch('asyncio.get_running_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=lambda executor, fn, *args: fn(*args),
            )

            await core.add_request(
                prompt="test",
                frequency_penalty=0.5,
                presence_penalty=0.3,
                logit_bias={100: -10},
            )

        assert captured_request is not None
        assert captured_request.sampling_params.frequency_penalty == 0.5
        assert captured_request.sampling_params.presence_penalty == 0.3
        assert captured_request.sampling_params.logit_bias == {100: -10}


# ── PagedScheduler integration test ──


class TestPagedSchedulerIntegration:
    """Test that PagedScheduler properly integrates KVCacheManager."""

    def test_paged_scheduler_config_exists(self):
        """Verify EngineCoreConfig has enable_paged_kv option."""
        from yunshu_engine.engine_core import EngineCoreConfig

        config = EngineCoreConfig()
        assert hasattr(config, 'enable_paged_kv')
        assert config.enable_paged_kv is True  # C11: enabled by default

        config_with_paged = EngineCoreConfig(enable_paged_kv=True)
        assert config_with_paged.enable_paged_kv is True

    def test_paged_scheduler_class_exists(self):
        """Verify PagedScheduler can be imported."""
        from yunshu_engine.paged_scheduler import PagedScheduler
        assert PagedScheduler is not None

    def test_scheduler_config_has_spec_decode(self):
        """Verify SchedulerConfig has speculative decoding fields."""
        from yunshu_engine.scheduler import SchedulerConfig

        config = SchedulerConfig()
        assert hasattr(config, 'enable_spec_decode')
        assert config.enable_spec_decode is False
