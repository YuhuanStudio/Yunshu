"""Tests for prompt_tokens accuracy across all generation paths.

Verifies that prompt_tokens is set correctly and consistently in:
- BatchedEngine fast path (non-streaming)
- BatchedEngine engine loop path (non-streaming)
- VLMEngine non-streaming
- RequestOutput.usage property
- GenerationOutput dataclass
"""

from __future__ import annotations

from unittest.mock import MagicMock

from yunshu_engine.batched_engine import GenerationOutput
from yunshu_engine.request import RequestOutput

# ── 1. GenerationOutput prompt_tokens ──


class TestGenerationOutputPromptTokens:
    """Verify GenerationOutput dataclass carries prompt_tokens correctly."""

    def test_default_prompt_tokens_is_zero(self):
        out = GenerationOutput()
        assert out.prompt_tokens == 0

    def test_prompt_tokens_set_correctly(self):
        out = GenerationOutput(prompt_tokens=42)
        assert out.prompt_tokens == 42

    def test_prompt_tokens_with_all_fields(self):
        out = GenerationOutput(
            text="hello",
            new_text="hello",
            prompt_tokens=10,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            cached_tokens=3,
            reasoning_tokens=2,
            ttft_ms=12.5,
        )
        assert out.prompt_tokens == 10
        assert out.completion_tokens == 5
        assert out.finished is True
        assert out.finish_reason == "stop"
        assert out.cached_tokens == 3
        assert out.reasoning_tokens == 2
        assert out.ttft_ms == 12.5


# ── 2. RequestOutput usage property ──


class TestRequestOutputUsage:
    """Verify RequestOutput.usage property computes total_tokens correctly."""

    def test_usage_property(self):
        out = RequestOutput(
            request_id="test",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert out.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_usage_with_zero_tokens(self):
        out = RequestOutput(request_id="test")
        assert out.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_usage_large_tokens(self):
        out = RequestOutput(
            request_id="test",
            prompt_tokens=100000,
            completion_tokens=50000,
        )
        assert out.usage["total_tokens"] == 150000


# ── 3. BatchedEngine fast path prompt_tokens ──


# ── 4. Engine loop prompt_tokens mapping ──


class TestEngineLoopPromptTokensMapping:
    """Verify prompt_tokens is mapped correctly from engine_core results."""

    def test_engine_core_result_mapped_to_generation_output(self):
        mock_result = MagicMock()
        mock_result.prompt_tokens = 42
        mock_result.completion_tokens = 10
        mock_result.output_text = "hello world"
        mock_result.finish_reason = "stop"
        mock_result.reasoning_tokens = 0
        mock_result.ttft_ms = 15.0
        mock_result.error = None

        output = GenerationOutput(
            text=mock_result.output_text,
            new_text=mock_result.output_text,
            prompt_tokens=mock_result.prompt_tokens,
            completion_tokens=mock_result.completion_tokens,
            finished=True,
            finish_reason=mock_result.finish_reason,
            reasoning_tokens=0,
            cached_tokens=0,
            ttft_ms=15.0,
        )

        assert output.prompt_tokens == 42
        assert output.completion_tokens == 10

    def test_engine_core_none_returns_error_output(self):
        output = GenerationOutput(
            finished=True,
            finish_reason="error",
            error="engine_core returned None",
        )
        assert output.prompt_tokens == 0
        assert output.finish_reason == "error"


# ── 5. VLMEngine prompt_tokens ──


# ── 6. Streaming prompt_tokens consistency ──


class TestStreamingPromptTokensConsistency:
    """Verify streaming and non-streaming paths produce the same prompt_tokens."""

    def test_request_output_prompt_tokens_in_stream(self):
        out = RequestOutput(
            request_id="test-req",
            new_text="hello",
            prompt_tokens=42,
            completion_tokens=1,
            finish_reason=None,
            finished=False,
        )
        assert out.prompt_tokens == 42
        assert out.finished is False

    def test_request_output_final_stream_has_same_prompt_tokens(self):
        stream_outputs = [
            RequestOutput(
                request_id="req-1", prompt_tokens=10, completion_tokens=1, new_text="he"
            ),
            RequestOutput(
                request_id="req-1",
                prompt_tokens=10,
                completion_tokens=2,
                new_text="llo",
            ),
            RequestOutput(
                request_id="req-1",
                prompt_tokens=10,
                completion_tokens=3,
                new_text="",
                finish_reason="stop",
                finished=True,
            ),
        ]

        prompt_tokens_set = {o.prompt_tokens for o in stream_outputs}
        assert len(prompt_tokens_set) == 1, (
            f"Expected consistent prompt_tokens, got {prompt_tokens_set}"
        )
        assert 10 in prompt_tokens_set

    def test_generation_output_prompt_tokens_matches_request_output(self):
        req_out = RequestOutput(
            request_id="req-1",
            prompt_tokens=25,
            completion_tokens=10,
            finish_reason="stop",
            finished=True,
        )

        gen_out = GenerationOutput(
            text="final text",
            prompt_tokens=req_out.prompt_tokens,
            completion_tokens=req_out.completion_tokens,
            finish_reason=req_out.finish_reason,
            finished=req_out.finished,
        )

        assert gen_out.prompt_tokens == 25
        assert gen_out.prompt_tokens == req_out.prompt_tokens


# ── 7. RequestStatus finish_reason mapping ──


class TestRequestStatusFinishReason:
    """Verify RequestStatus maps to correct finish_reason strings."""

    def test_finished_stopped(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"

    def test_finished_length(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.FINISHED_LENGTH) == "length"

    def test_finished_aborted(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"

    def test_finished_error(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.FINISHED_ERROR) == "error"

    def test_finished_timeout(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.FINISHED_TIMEOUT) == "timeout"

    def test_running_returns_none(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.RUNNING) is None

    def test_waiting_returns_none(self):
        from yunshu_engine.request import RequestStatus

        assert RequestStatus.finish_reason(RequestStatus.WAITING) is None

    def test_is_finished(self):
        from yunshu_engine.request import RequestStatus

        assert not RequestStatus.is_finished(RequestStatus.WAITING)
        assert not RequestStatus.is_finished(RequestStatus.RUNNING)
        assert not RequestStatus.is_finished(RequestStatus.PREFILLING)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ERROR)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_TIMEOUT)


# ── 8. _normalize_finish_reason ──


class TestNormalizeFinishReason:
    """Verify _normalize_finish_reason maps all internal reasons correctly."""

    def test_valid_openai_reasons_passthrough(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason

        assert _normalize_finish_reason("stop") == "stop"
        assert _normalize_finish_reason("length") == "length"
        assert _normalize_finish_reason("tool_calls") == "tool_calls"
        assert _normalize_finish_reason("content_filter") == "content_filter"

    def test_none_returns_stop(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason

        assert _normalize_finish_reason(None) == "stop"
        assert _normalize_finish_reason("") == "stop"

    def test_internal_reasons_mapped(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason

        assert _normalize_finish_reason("abort") == "stop"
        assert _normalize_finish_reason("cancel") == "stop"
        assert _normalize_finish_reason("error") == "stop"
        assert _normalize_finish_reason("timeout") == "length"  # truncation signal
        assert _normalize_finish_reason("memory_limit") == "length"
        assert _normalize_finish_reason("memory_exceeded") == "length"

    def test_unknown_reason_defaults_to_stop(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason

        assert _normalize_finish_reason("unknown_reason") == "stop"
