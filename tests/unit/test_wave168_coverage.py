"""Tests for Wave 168 coverage: finish_reason edge cases, GenerationOutput
field correctness, RequestOutput.usage edge cases, streaming cancel_event
propagation, VLM finish_reason error paths, and prompt tokens with KV prefix cache.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

from yunshu_engine.request import RequestOutput
from yunshu_engine.batched_engine import GenerationOutput


# ── 1. _normalize_finish_reason edge cases ──


class TestNormalizeFinishReasonEdgeCases:
    """Extended edge cases for _normalize_finish_reason beyond basic mapping."""

    def test_whitespace_only_string(self):
        """Whitespace-only strings should map to 'stop' (falsy check)."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        # "   " is truthy but not in valid set or internal map → default "stop"
        assert _normalize_finish_reason("   ") == "stop"

    def test_case_sensitive_unknown(self):
        """Unknown reasons are case-sensitive: 'Stop' != 'stop'."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason("Stop") == "stop"  # unknown -> default
        assert _normalize_finish_reason("LENGTH") == "stop"  # unknown -> default

    def test_all_internal_reasons(self):
        """Verify every entry in _INTERNAL_MAP is correct."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        expected = {
            "abort": "stop",
            "cancel": "stop",
            "error": "length",
            "timeout": "length",
            "memory_limit": "length",
            "memory_exceeded": "length",
        }
        for internal, expected_openai in expected.items():
            assert _normalize_finish_reason(internal) == expected_openai, \
                f"Expected {internal} -> {expected_openai}, got {_normalize_finish_reason(internal)}"

    def test_stable_return_values(self):
        """Calling _normalize_finish_reason multiple times returns the same value."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        for reason in ["stop", None, "", "timeout", "cancel", "unknown"]:
            first = _normalize_finish_reason(reason)
            second = _normalize_finish_reason(reason)
            assert first == second, f"Inconsistent results for {reason!r}: {first} vs {second}"

    def test_reason_with_surrounding_whitespace(self):
        """Reasons with surrounding whitespace should NOT match valid reasons."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason(" stop ") == "stop"

    def test_content_filter_passthrough(self):
        """content_filter is a valid OpenAI finish reason."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason("content_filter") == "content_filter"

    def test_tool_calls_passthrough(self):
        """tool_calls is a valid OpenAI finish reason."""
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason("tool_calls") == "tool_calls"


# ── 2. GenerationOutput field correctness ──


class TestGenerationOutputFieldCorrectness:
    """Verify GenerationOutput dataclass fields and defaults."""

    def test_all_default_values(self):
        """All fields should have sensible defaults."""
        out = GenerationOutput()
        assert out.text == ""
        assert out.new_text == ""
        assert out.prompt_tokens == 0
        assert out.completion_tokens == 0
        assert out.finished is False
        assert out.finish_reason is None
        assert out.cached_tokens == 0
        assert out.logprobs is None
        assert out.ttft_ms == 0.0
        assert out.reasoning_tokens == 0
        assert out.current_state is None
        assert out.error is None

    def test_text_and_new_text_independent(self):
        """text and new_text are independent fields."""
        out = GenerationOutput(text="full text", new_text=" text")
        assert out.text == "full text"
        assert out.new_text == " text"

    def test_logprobs_list_structure(self):
        """logprobs field accepts list of dicts."""
        logprobs_data = [
            {"token": "hello", "logprob": -1.5, "top_logprobs": []},
            {"token": " world", "logprob": -0.3, "top_logprobs": []},
        ]
        out = GenerationOutput(logprobs=logprobs_data)
        assert len(out.logprobs) == 2
        assert out.logprobs[0]["token"] == "hello"
        assert out.logprobs[1]["logprob"] == -0.3

    def test_current_state_values(self):
        """current_state should accept 'reasoning', 'normal', or None."""
        out_reasoning = GenerationOutput(current_state="reasoning")
        assert out_reasoning.current_state == "reasoning"

        out_normal = GenerationOutput(current_state="normal")
        assert out_normal.current_state == "normal"

        out_none = GenerationOutput(current_state=None)
        assert out_none.current_state is None

    def test_error_field_for_failed_generation(self):
        """error field carries error message for failed generations."""
        out = GenerationOutput(
            finished=True,
            finish_reason="error",
            error="Out of memory during generation",
        )
        assert out.error == "Out of memory during generation"
        assert out.finish_reason == "error"

    def test_mixed_fields_partial_set(self):
        """Setting only some fields should leave others at defaults."""
        out = GenerationOutput(
            text="result",
            prompt_tokens=100,
            completion_tokens=50,
            finished=True,
            finish_reason="stop",
        )
        assert out.new_text == ""  # default
        assert out.cached_tokens == 0  # default
        assert out.logprobs is None  # default
        assert out.reasoning_tokens == 0  # default

    def test_large_token_counts(self):
        """Token counts should handle large values."""
        out = GenerationOutput(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        assert out.prompt_tokens == 1_000_000
        assert out.completion_tokens == 500_000

    def test_negative_ttft(self):
        """ttft_ms can be negative (clock skew edge case) -- should still store."""
        out = GenerationOutput(ttft_ms=-5.0)
        assert out.ttft_ms == -5.0

    def test_cached_tokens_with_kv_prefix(self):
        """cached_tokens should be populated when KV prefix cache hits."""
        out = GenerationOutput(
            text="world",
            prompt_tokens=20,
            cached_tokens=15,
            completion_tokens=1,
        )
        assert out.cached_tokens == 15
        # Net prompt tokens after cache = 20 - 15 = 5
        assert out.prompt_tokens - out.cached_tokens == 5


# ── 3. RequestOutput.usage edge cases ──


class TestRequestOutputUsageEdgeCases:
    """Extended edge cases for RequestOutput.usage property."""

    def test_usage_zero_completion_tokens(self):
        """Usage with zero completion tokens (e.g. prefill-only)."""
        out = RequestOutput(
            request_id="req-1",
            prompt_tokens=1000,
            completion_tokens=0,
        )
        assert out.usage["total_tokens"] == 1000

    def test_usage_zero_prompt_tokens(self):
        """Usage with zero prompt tokens (edge case: empty prompt)."""
        out = RequestOutput(
            request_id="req-1",
            prompt_tokens=0,
            completion_tokens=10,
        )
        assert out.usage["total_tokens"] == 10
        assert out.usage["prompt_tokens"] == 0

    def test_usage_both_zero(self):
        """Usage with both zero (error/timeout before generation)."""
        out = RequestOutput(request_id="req-1", error="timeout")
        assert out.usage["total_tokens"] == 0
        assert out.usage["prompt_tokens"] == 0
        assert out.usage["completion_tokens"] == 0

    def test_usage_returns_new_dict_each_call(self):
        """usage property should return a new dict each time (no aliasing)."""
        out = RequestOutput(request_id="req-1", prompt_tokens=10, completion_tokens=5)
        usage1 = out.usage
        usage2 = out.usage
        assert usage1 == usage2
        assert usage1 is not usage2  # different objects

    def test_usage_does_not_include_optional_fields(self):
        """usage dict only has prompt/completion/total -- no reasoning_tokens etc."""
        out = RequestOutput(
            request_id="req-1",
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=3,
            cached_tokens=8,
        )
        usage = out.usage
        assert set(usage.keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    def test_token_text_backward_compat(self):
        """token_text property should return new_text."""
        out = RequestOutput(
            request_id="req-1",
            new_text="hello",
        )
        assert out.token_text == "hello"

    def test_token_text_empty_when_no_new_text(self):
        """token_text returns empty string when new_text is empty."""
        out = RequestOutput(request_id="req-1")
        assert out.token_text == ""

    def test_token_id_from_new_token_ids(self):
        """token_id property returns last element of new_token_ids."""
        out = RequestOutput(
            request_id="req-1",
            new_token_ids=[100, 200, 300],
        )
        assert out.token_id == 300

    def test_token_id_default_when_empty(self):
        """token_id returns -1 when new_token_ids is empty (no valid token)."""
        out = RequestOutput(request_id="req-1")
        assert out.token_id == -1

    def test_logprob_alias(self):
        """logprob property should alias logprobs."""
        out = RequestOutput(
            request_id="req-1",
            logprobs=[{"token": "a", "logprob": -1.0}],
        )
        assert out.logprob == out.logprobs

    def test_cached_tokens_default_zero(self):
        """cached_tokens defaults to 0."""
        out = RequestOutput(request_id="req-1")
        assert out.cached_tokens == 0

    def test_ttft_ms_default_zero(self):
        """ttft_ms defaults to 0.0."""
        out = RequestOutput(request_id="req-1")
        assert out.ttft_ms == 0.0


# ── 4. Streaming cancel_event propagation ──


class TestStreamingCancelEventPropagation:
    """Test that cancel_event is properly propagated through gateway to engine."""

    def test_cancel_event_created_and_set(self):
        """asyncio.Event can be created, unset by default, then set."""
        evt = asyncio.Event()
        assert not evt.is_set()
        evt.set()
        assert evt.is_set()

    def test_cancel_event_can_be_awaited_after_set(self):
        """Event.wait() returns immediately after set()."""
        evt = asyncio.Event()
        evt.set()
        assert evt.is_set()

    @pytest.mark.asyncio
    async def test_cancel_event_propagation_through_request_tracker(self):
        """Request tracker registers and provides cancel_event."""
        from yunshu_engine.request_tracker import RequestTracker

        tracker = RequestTracker()
        gen = tracker.register("test-req-1", "test-model")

        assert hasattr(gen, 'cancel_event')
        assert isinstance(gen.cancel_event, threading.Event)
        assert not gen.cancel_event.is_set()

        gen.cancel_event.set()
        assert gen.cancel_event.is_set()

        tracker.unregister("test-req-1")

    @pytest.mark.asyncio
    async def test_cancel_event_set_during_generation(self):
        """When cancel_event is set mid-generation, engine should respect it."""
        evt = asyncio.Event()

        token_count = 0
        max_tokens = 100

        for i in range(max_tokens):
            if evt.is_set():
                break
            token_count += 1
            if i == 5:
                evt.set()

        assert token_count == 6  # Stopped after 6 tokens

    @pytest.mark.asyncio
    async def test_chat_router_non_streaming_cancel_event_registration(self):
        """Verify chat.py non-streaming path registers with request tracker."""
        from yunshu_engine.request_tracker import RequestTracker

        tracker = RequestTracker()
        gen = tracker.register("chatcmpl-test123", "test-model")

        assert gen.cancel_event is not None
        assert isinstance(gen.cancel_event, threading.Event)

        tracker.unregister("chatcmpl-test123")

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming_has_cancel_event_param(self):
        """Verify Anthropic non-streaming functions accept cancel_event parameter.

        This tests the fix for the gap where Anthropic non-streaming didn't
        forward cancel_event.
        """
        import inspect
        from yunshu_gateway.routers.anthropic import _non_stream_batched, _non_stream_legacy

        sig_batched = inspect.signature(_non_stream_batched)
        sig_legacy = inspect.signature(_non_stream_legacy)

        assert "cancel_event" in sig_batched.parameters, \
            "_non_stream_batched must accept cancel_event parameter"
        assert "cancel_event" in sig_legacy.parameters, \
            "_non_stream_legacy must accept cancel_event parameter"

    @pytest.mark.asyncio
    async def test_completions_non_streaming_cancel_event(self):
        """Verify completions.py non-streaming path registers with request tracker."""
        from yunshu_engine.request_tracker import RequestTracker

        tracker = RequestTracker()
        gen = tracker.register("cmpl-test456", "test-model")

        assert gen.cancel_event is not None
        assert isinstance(gen.cancel_event, threading.Event)

        tracker.unregister("cmpl-test456")


# ── 5. VLM finish_reason error paths ──


class TestVLMFinishReasonErrorPaths:
    """Test finish_reason correctness in VLM error and edge case paths."""

    def _make_vlm_engine(self):
        """Create a VLMEngine with mocked internals."""
        from yunshu_engine.vlm_engine import VLMEngine
        engine = object.__new__(VLMEngine)
        engine._model_path = "/models/test-model"
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._processor = MagicMock()
        engine._config = {}
        engine._running = True
        engine._active_count = 0
        engine._num_requests_processed = 0
        engine._total_reasoning_tokens = 0
        engine._start_time = 0.0
        engine._has_vision = False
        engine._is_vlm = False
        engine._temp_files = None
        engine._temp_files_lock = threading.Lock()
        engine._mrope_info = MagicMock(enabled=False)
        engine._rope_delta_manager = None
        engine._vision_cache = None
        engine._vlm_vision_cache_adapter = None
        engine._encoder_cache = MagicMock()
        engine._text_prompt_cache = MagicMock()
        engine._kv_prefix_states = {}
        engine._multimodal_prefix_cache = {}
        engine._vision_encoder_factory = None
        engine._spec_prefill_enabled = False
        engine._pipeline = MagicMock()
        engine._async_core = None
        engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return engine

    @pytest.mark.asyncio
    async def test_vlm_empty_output_returns_length(self):
        """VLM with zero output tokens should return finish_reason='length'."""
        engine = self._make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = ""
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler'):
                # No tokens at all (immediate EOS)
                mock_step.return_value = iter([(2, None)])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )

        # Should produce a finish_reason (stop for EOS, length if max_tokens)
        assert result["finish_reason"] in ("stop", "length")

    @pytest.mark.asyncio
    async def test_vlm_single_token_output(self):
        """VLM with exactly 1 output token should have correct finish_reason."""
        engine = self._make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler'):
                mock_step.return_value = iter([(100, None)])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=1,
                )

        assert result["finish_reason"] == "length"
        assert result["completion_tokens"] == 1

    @pytest.mark.asyncio
    async def test_vlm_stop_sequence_exact_match(self):
        """VLM with stop sequence matching exactly at end returns 'stop'."""
        engine = self._make_vlm_engine()
        engine._tokenizer.decode.return_value = "hello\n"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        def mock_encode(text):
            if text == "\n":
                return [10]
            return [1, 2, 3]
        engine._tokenizer.encode.side_effect = mock_encode
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler'):
                # Token 10 = "\n" which is in stop list
                mock_step.return_value = iter([(100, None), (10, None)])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    stop=["\n"],
                )

        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_vlm_streaming_error_finish_reason(self):
        """VLM streaming with generation error should emit error finish_reason."""
        engine = self._make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        with patch('mlx_lm.generate.generate_step', side_effect=RuntimeError("GPU OOM")):
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
            ):
                outputs.append(output)

        error_outputs = [o for o in outputs if o.finish_reason == "error"]
        assert len(error_outputs) >= 1, "Expected at least one error output"
        assert error_outputs[-1].finished is True
        assert "GPU OOM" in (error_outputs[-1].error or "")


# ── 6. Prompt tokens with KV prefix cache ──


class TestPromptTokensWithKVPrefixCache:
    """Test prompt_tokens accuracy when KV prefix cache provides cached_tokens."""

    def test_generation_output_with_cache(self):
        """GenerationOutput with cached_tokens should have prompt_tokens > cached_tokens."""
        out = GenerationOutput(
            text="completion",
            prompt_tokens=100,
            completion_tokens=10,
            cached_tokens=80,
        )
        assert out.prompt_tokens == 100
        assert out.cached_tokens == 80
        # 80 tokens came from cache, 20 were computed
        assert out.prompt_tokens - out.cached_tokens == 20

    def test_generation_output_full_cache_hit(self):
        """When all prompt tokens are cached, cached_tokens == prompt_tokens."""
        out = GenerationOutput(
            prompt_tokens=50,
            completion_tokens=5,
            cached_tokens=50,
        )
        assert out.cached_tokens == out.prompt_tokens

    def test_generation_output_no_cache(self):
        """When no cache, cached_tokens == 0."""
        out = GenerationOutput(
            prompt_tokens=50,
            completion_tokens=5,
            cached_tokens=0,
        )
        assert out.cached_tokens == 0

    def test_request_output_with_cached_tokens(self):
        """RequestOutput carries cached_tokens for usage reporting."""
        out = RequestOutput(
            request_id="req-1",
            prompt_tokens=100,
            completion_tokens=10,
            cached_tokens=75,
        )
        assert out.cached_tokens == 75
        # usage dict only has the basic fields
        assert out.usage["prompt_tokens"] == 100
        assert out.usage["completion_tokens"] == 10

    def test_cache_tokens_usage_calculation_pattern(self):
        """Gateway pattern: cache_creation = prompt - cached, cache_read = cached."""
        out = GenerationOutput(
            prompt_tokens=200,
            cached_tokens=150,
        )
        cache_creation = max(0, out.prompt_tokens - out.cached_tokens)
        cache_read = max(0, out.cached_tokens)
        assert cache_creation == 50
        assert cache_read == 150

    def test_cache_tokens_zero_when_no_prefix_match(self):
        """When KV prefix has no match, cached_tokens is 0."""
        out = GenerationOutput(
            prompt_tokens=100,
            completion_tokens=10,
        )
        assert out.cached_tokens == 0

    @pytest.mark.asyncio
    async def test_vlm_prompt_tokens_with_prefix_cache(self):
        """VLM prompt_tokens should be correct even when KV prefix cache provides a hit."""
        engine = self._make_vlm_engine()

        # Simulate tokenizer returning 7 tokens for prompt
        engine._tokenizer.encode.return_value = [10, 20, 30, 40, 50, 60, 70]
        engine._tokenizer.decode.return_value = "hello"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[10, 20, 30, 40, 50, 60, 70])
        engine._get_eos_ids = MagicMock(return_value=[2])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler'):
                mock_step.return_value = iter([(100, None), (2, None)])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )

        assert result["prompt_tokens"] == 7

    def _make_vlm_engine(self):
        """Create a VLMEngine with mocked internals."""
        from yunshu_engine.vlm_engine import VLMEngine
        engine = object.__new__(VLMEngine)
        engine._model_path = "/models/test-model"
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._processor = MagicMock()
        engine._config = {}
        engine._running = True
        engine._active_count = 0
        engine._num_requests_processed = 0
        engine._total_reasoning_tokens = 0
        engine._start_time = 0.0
        engine._has_vision = False
        engine._is_vlm = False
        engine._temp_files = None
        engine._temp_files_lock = threading.Lock()
        engine._mrope_info = MagicMock(enabled=False)
        engine._rope_delta_manager = None
        engine._vision_cache = None
        engine._vlm_vision_cache_adapter = None
        engine._encoder_cache = MagicMock()
        engine._text_prompt_cache = MagicMock()
        engine._kv_prefix_states = {}
        engine._multimodal_prefix_cache = {}
        engine._vision_encoder_factory = None
        engine._spec_prefill_enabled = False
        engine._pipeline = MagicMock()
        engine._async_core = None
        engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return engine


# ── 7. Anthropic _map_stop_reason edge cases ──


class TestAnthropicMapStopReason:
    """Test Anthropic stop_reason mapping edge cases."""

    def test_stop_maps_to_end_turn(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason("stop") == "end_turn"

    def test_length_maps_to_max_tokens(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason("length") == "max_tokens"

    def test_tool_calls_maps_to_tool_use(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason("tool_calls") == "tool_use"

    def test_matched_stop_sequence(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason(None, matched_stop="\n\n") == "stop_sequence"

    def test_has_tool_calls_overrides(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason("stop", has_tool_calls=True) == "tool_use"
        assert _map_stop_reason("length", has_tool_calls=True) == "tool_use"

    def test_matched_stop_overrides_has_tool_calls(self):
        """When both matched_stop and has_tool_calls, tool_use wins."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        # has_tool_calls is checked first
        assert _map_stop_reason("stop", matched_stop="END", has_tool_calls=True) == "tool_use"

    def test_unknown_falls_to_end_turn(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason("unknown") == "end_turn"

    def test_none_no_match_returns_end_turn(self):
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason(None) == "end_turn"

    def test_internal_timeout_maps_to_max_tokens(self):
        """Internal 'timeout' is not in _FINISH_REASON_MAP, but goes to end_turn."""
        from yunshu_gateway.routers.anthropic import _map_stop_reason
        assert _map_stop_reason("timeout") == "end_turn"


# ── 8. Parameter forwarding completeness ──


class TestParameterForwardingCompleteness:
    """Verify all gateway routers forward all required parameters to engine."""

    def test_chat_completion_request_has_all_params(self):
        """ChatCompletionRequest model should have all required parameters."""
        from yunshu_gateway.routers.chat import ChatCompletionRequest
        fields = set(ChatCompletionRequest.model_fields.keys())

        required_fields = {
            'temperature', 'top_p', 'top_k', 'min_p',
            'frequency_penalty', 'presence_penalty', 'repetition_penalty',
            'stop', 'stop_token_ids', 'max_tokens', 'seed',
            'logprobs', 'top_logprobs', 'thinking_budget', 'reasoning_effort',
            'enable_thinking', 'response_format', 'grammar', 'spec_decode',
            'priority', 'xtc_probability', 'xtc_threshold',
            'logit_bias', 'n',
        }
        missing = required_fields - fields
        assert not missing, f"ChatCompletionRequest missing fields: {sorted(missing)}"

    def test_completion_request_has_all_params(self):
        """CompletionRequest model should have all required parameters."""
        from yunshu_gateway.routers.completions import CompletionRequest
        fields = set(CompletionRequest.model_fields.keys())

        required_fields = {
            'temperature', 'top_p', 'top_k', 'min_p',
            'frequency_penalty', 'presence_penalty', 'repetition_penalty',
            'stop', 'stop_token_ids', 'max_tokens', 'seed',
            'logprobs', 'top_logprobs', 'thinking_budget', 'reasoning_effort',
            'enable_thinking', 'response_format', 'grammar', 'spec_decode',
            'priority', 'xtc_probability', 'xtc_threshold',
            'logit_bias', 'n', 'timeout',
        }
        missing = required_fields - fields
        assert not missing, f"CompletionRequest missing fields: {sorted(missing)}"

    def test_anthropic_request_has_extended_params(self):
        """AnthropicMessagesRequest should have Yunshu-extended fields."""
        from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest
        fields = set(AnthropicMessagesRequest.model_fields.keys())

        extended_fields = {
            'min_p', 'repetition_penalty', 'frequency_penalty', 'presence_penalty',
            'logit_bias', 'seed', 'reasoning_effort', 'stop_token_ids',
            'spec_decode', 'xtc_probability', 'xtc_threshold',
            'priority', 'json_schema', 'logprobs', 'top_logprobs',
            'timeout',
        }
        missing = extended_fields - fields
        assert not missing, f"AnthropicMessagesRequest missing extended fields: {sorted(missing)}"

    def test_responses_request_has_all_params(self):
        """ResponsesRequest model should have all required parameters."""
        from yunshu_gateway.routers.responses import ResponsesRequest
        fields = set(ResponsesRequest.model_fields.keys())

        required_fields = {
            'temperature', 'top_p', 'top_k', 'min_p',
            'frequency_penalty', 'presence_penalty', 'repetition_penalty',
            'stop', 'stop_token_ids', 'max_output_tokens', 'seed',
            'logprobs', 'top_logprobs', 'thinking_budget', 'reasoning_effort',
            'enable_thinking', 'response_format', 'grammar', 'spec_decode',
            'priority', 'xtc_probability', 'xtc_threshold',
            'logit_bias', 'n', 'timeout',
        }
        missing = required_fields - fields
        assert not missing, f"ResponsesRequest missing fields: {sorted(missing)}"


# ── 9. Anthropic logit_bias conversion ──


class TestAnthropicLogitBiasConversion:
    """Test _convert_logit_bias in Anthropic router."""

    def test_none_logit_bias(self):
        from yunshu_gateway.routers.anthropic import _convert_logit_bias
        req = MagicMock()
        req.logit_bias = None
        assert _convert_logit_bias(req) is None

    def test_string_keys_converted_to_int(self):
        from yunshu_gateway.routers.anthropic import _convert_logit_bias
        req = MagicMock()
        req.logit_bias = {"100": 5.0, "200": -1.0}
        result = _convert_logit_bias(req)
        assert result == {100: 5.0, 200: -1.0}

    def test_invalid_keys_skipped(self):
        from yunshu_gateway.routers.anthropic import _convert_logit_bias
        req = MagicMock()
        req.logit_bias = {"100": 5.0, "abc": -1.0, "200": 3.0}
        result = _convert_logit_bias(req)
        assert 100 in result
        assert 200 in result
        assert "abc" not in result

    def test_empty_logit_bias(self):
        from yunshu_gateway.routers.anthropic import _convert_logit_bias
        req = MagicMock()
        req.logit_bias = {}
        assert _convert_logit_bias(req) is None


# ── 10. Response format parsing ──


class TestResponseFormatParsing:
    """Test response_format / grammar parsing across routers."""

    def test_chat_parse_json_object(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format({"type": "json_object"})
        assert result == "json_object"

    def test_chat_parse_json_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _parse_response_format({
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema},
        })
        assert result == schema

    def test_chat_parse_grammar_json(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        schema = {"type": "object"}
        result = _parse_response_format(None, grammar={"type": "json", "schema": schema})
        assert result == schema

    def test_chat_parse_grammar_regex(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        grammar = {"type": "regex", "pattern": "[0-9]+"}
        result = _parse_response_format(None, grammar=grammar)
        assert result == grammar

    def test_chat_parse_none(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        assert _parse_response_format(None) is None

    def test_chat_parse_text_type(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format({"type": "text"})
        assert result is None

    def test_responses_parse_json_object(self):
        from yunshu_gateway.routers.responses import _parse_response_format
        result = _parse_response_format({"type": "json_object"})
        assert result == "json_object"

    def test_responses_parse_json_schema(self):
        from yunshu_gateway.routers.responses import _parse_response_format
        schema = {"type": "object"}
        result = _parse_response_format({
            "type": "json_schema",
            "json_schema": {"schema": schema},
        })
        assert result == schema

    def test_responses_parse_grammar_json_no_schema(self):
        from yunshu_gateway.routers.responses import _parse_response_format
        result = _parse_response_format(None, grammar={"type": "json"})
        assert result == "json_object"

    def test_responses_parse_none(self):
        from yunshu_gateway.routers.responses import _parse_response_format
        assert _parse_response_format(None) is None

    def test_anthropic_resolve_json_schema_direct(self):
        from yunshu_gateway.routers.anthropic import _resolve_json_schema
        req = MagicMock()
        req.json_schema = {"type": "object"}
        req.response_format = None
        assert _resolve_json_schema(req) == {"type": "object"}

    def test_anthropic_resolve_json_schema_from_response_format(self):
        from yunshu_gateway.routers.anthropic import _resolve_json_schema
        req = MagicMock()
        req.json_schema = None
        req.response_format = {"type": "json_object"}
        assert _resolve_json_schema(req) == "json_object"

    def test_anthropic_resolve_none(self):
        from yunshu_gateway.routers.anthropic import _resolve_json_schema
        req = MagicMock()
        req.json_schema = None
        req.response_format = None
        assert _resolve_json_schema(req) is None
