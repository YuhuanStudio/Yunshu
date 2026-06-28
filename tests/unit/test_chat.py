"""OpenAI Chat Completions API edge case tests.

Covers:
- max_tokens=0 (prompt-only return)
- max_tokens=1 (single token generation)
- Empty messages array (400 error)
- System-only messages (400 error)
- temperature=0 with streaming (deterministic)
- stop=["\\n"] with streaming (stop sequence truncation)
- n>1 with streaming (400 error)
- Empty model (400 error)
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from yunshu_gateway.routers.chat import (
    ChatCompletionRequest,
    ChatMessage,
    _extract_messages,
    _normalize_finish_reason,
    _validate_sampling_params,
)


class TestImagePartNormalization:
    """image / image_data content parts must be normalized to
    image_url so the VLM engine (which only handles image_url) actually sees
    them instead of silently dropping → hallucination."""

    def test_image_url_passthrough(self):
        from yunshu_gateway.routers.chat import _normalize_image_part
        p = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
        assert _normalize_image_part(p) == p

    def test_image_type_with_nested_url(self):
        from yunshu_gateway.routers.chat import _normalize_image_part
        out = _normalize_image_part({"type": "image", "image_url": {"url": "https://x/a.jpg"}})
        assert out == {"type": "image_url", "image_url": {"url": "https://x/a.jpg"}}

    def test_image_data_bare_base64_wrapped(self):
        from yunshu_gateway.routers.chat import _normalize_image_part
        out = _normalize_image_part({"type": "image_data", "data": "QUJD"})
        assert out["type"] == "image_url"
        assert out["image_url"]["url"] == "data:image/png;base64,QUJD"

    def test_image_data_existing_data_url_untouched(self):
        from yunshu_gateway.routers.chat import _normalize_image_part
        out = _normalize_image_part({"type": "image_data", "data": "data:image/jpeg;base64,QUJD"})
        assert out["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

    def test_text_part_untouched(self):
        from yunshu_gateway.routers.chat import _normalize_image_part
        p = {"type": "text", "text": "hi"}
        assert _normalize_image_part(p) == p

    def test_extract_messages_normalizes_image_type(self):
        msgs = [ChatMessage(role="user", content=[
            {"type": "text", "text": "what is this"},
            {"type": "image", "image_url": {"url": "https://x/a.png"}},
        ])]
        out = _extract_messages(msgs)
        parts = out[0]["content"]
        assert any(p.get("type") == "image_url" and p["image_url"]["url"] == "https://x/a.png" for p in parts)
        assert not any(p.get("type") == "image" for p in parts)


class TestChatRequestValidation:
    """ChatCompletionRequest validation edge cases."""

    def test_empty_messages_rejected(self):
        """Empty messages array should return validation error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ChatCompletionRequest(model="test", messages=[])

    def test_system_only_messages_rejected(self):
        """System-only messages (no user message) should return validation error."""
        with pytest.raises(ValidationError, match="at least one message with role 'user'"):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="system", content="You are helpful.")],
            )

    def test_developer_only_messages_rejected(self):
        """Developer-only messages (no user message) should return validation error."""
        with pytest.raises(ValidationError, match="at least one message with role 'user'"):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="developer", content="Be precise.")],
            )

    def test_assistant_only_messages_rejected(self):
        """Assistant-only messages should return validation error."""
        with pytest.raises(ValidationError, match="at least one message with role 'user'"):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="assistant", content="Hello")],
            )

    def test_system_plus_user_allowed(self):
        """System + user messages should be accepted."""
        req = ChatCompletionRequest(
            model="test",
            messages=[
                ChatMessage(role="system", content="Be helpful."),
                ChatMessage(role="user", content="Hello"),
            ],
        )
        assert len(req.messages) == 2

    def test_user_only_allowed(self):
        """Single user message should be accepted."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.messages[0].role == "user"

    def test_empty_model_rejected(self):
        """Empty model string should return validation error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ChatCompletionRequest(
                model="",
                messages=[ChatMessage(role="user", content="Hi")],
            )

    def test_whitespace_model_rejected(self):
        """Whitespace-only model should return validation error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ChatCompletionRequest(
                model="   ",
                messages=[ChatMessage(role="user", content="Hi")],
            )

    def test_max_tokens_zero_allowed(self):
        """max_tokens=0 should be accepted (returns prompt_tokens only)."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=0,
        )
        assert req.max_tokens == 0
        assert req.effective_max_tokens() == 0

    def test_max_completion_tokens_zero_allowed(self):
        """max_completion_tokens=0 should be accepted."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            max_completion_tokens=0,
        )
        assert req.effective_max_tokens() == 0

    def test_max_tokens_one_allowed(self):
        """max_tokens=1 should be accepted (single token generation)."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=1,
        )
        assert req.max_tokens == 1

    def test_n_gt_1_with_streaming_rejected(self):
        """n > 1 with streaming should return validation error."""
        with pytest.raises(ValidationError, match="n > 1 is not supported when stream is True"):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                n=3,
                stream=True,
            )

    def test_n_eq_1_with_streaming_allowed(self):
        """n=1 with streaming should be accepted."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            n=1,
            stream=True,
        )
        assert req.n == 1
        assert req.stream is True

    def test_n_gt_1_non_streaming_allowed(self):
        """n > 1 without streaming should be accepted."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            n=5,
            stream=False,
        )
        assert req.n == 5

    def test_stop_max_16(self):
        """More than 16 stop sequences should be rejected."""
        stops = [f"stop{i}" for i in range(17)]
        with pytest.raises(ValidationError, match="maximum 16 stop sequences"):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                stop=stops,
            )

    def test_temperature_zero_allowed(self):
        """temperature=0 should be accepted (deterministic generation)."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0,
        )
        assert req.temperature == 0


class TestSamplingParamsValidation:
    """_validate_sampling_params edge cases."""

    def test_max_tokens_zero_allowed(self):
        """max_tokens=0 should not raise."""
        _validate_sampling_params(temperature=0.7, max_tokens=0, top_p=1.0)

    def test_max_tokens_negative_rejected(self):
        """Negative max_tokens should raise 422."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_sampling_params(temperature=0.7, max_tokens=-1, top_p=1.0)
        assert exc_info.value.status_code == 422

    def test_max_tokens_one_allowed(self):
        """max_tokens=1 should not raise."""
        _validate_sampling_params(temperature=0.7, max_tokens=1, top_p=1.0)

    def test_temperature_zero_allowed(self):
        """temperature=0 should not raise (deterministic)."""
        _validate_sampling_params(temperature=0, max_tokens=100, top_p=1.0)

    def test_temperature_above_two_rejected(self):
        """temperature > 2 should raise 422."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_sampling_params(temperature=2.1, max_tokens=100, top_p=1.0)
        assert exc_info.value.status_code == 422

    def test_top_p_zero_accepted(self):
        """top_p=0 is now ACCEPTED (matches Pydantic Field
        ge=0.0, le=1.0). Prior behavior was inconsistent — Pydantic
        accepted then validator rejected ."""
        _validate_sampling_params(temperature=0.7, max_tokens=100, top_p=0)
        _validate_sampling_params(temperature=0.7, max_tokens=100, top_p=0.0)

    def test_top_p_nan_rejected(self):
        """NEW: NaN/Inf rejected even though numerically [0,1]."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_sampling_params(temperature=0.7, max_tokens=100, top_p=float('nan'))
        assert exc_info.value.status_code == 422

    def test_top_p_inf_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_sampling_params(temperature=0.7, max_tokens=100, top_p=float('inf'))
        assert exc_info.value.status_code == 422


class TestNormalizeFinishReason:
    """_normalize_finish_reason edge cases."""

    def test_none_returns_stop(self):
        assert _normalize_finish_reason(None) == "stop"

    def test_empty_returns_stop(self):
        assert _normalize_finish_reason("") == "stop"

    def test_valid_reasons_passthrough(self):
        assert _normalize_finish_reason("stop") == "stop"
        assert _normalize_finish_reason("length") == "length"
        assert _normalize_finish_reason("tool_calls") == "tool_calls"
        assert _normalize_finish_reason("content_filter") == "content_filter"

    def test_internal_reasons_mapped(self):
        assert _normalize_finish_reason("abort") == "stop"
        assert _normalize_finish_reason("cancel") == "stop"
        # timeout truncates → "length" (was "stop", which masked truncation).
        assert _normalize_finish_reason("timeout") == "length"
        assert _normalize_finish_reason("memory_limit") == "length"

    def test_unknown_returns_stop(self):
        assert _normalize_finish_reason("something_unknown") == "stop"


class TestExtractMessages:
    """_extract_messages edge cases."""

    def test_none_content_becomes_empty_string(self):
        """Messages with None content should convert to empty string."""
        msgs = _extract_messages([ChatMessage(role="user", content=None)])
        assert msgs[0]["content"] == ""

    def test_string_content_preserved(self):
        msgs = _extract_messages([ChatMessage(role="user", content="hello")])
        assert msgs[0]["content"] == "hello"

    def test_tool_calls_preserved(self):
        """Tool call fields should be preserved in extracted messages."""
        from yunshu_gateway.routers.chat import ToolCall, ToolCallFunction
        msgs = _extract_messages([
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        function=ToolCallFunction(name="get_weather", arguments='{"city": "SF"}'),
                    ),
                ],
            ),
        ])
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_role_with_call_id(self):
        """Tool role messages should preserve tool_call_id."""
        msgs = _extract_messages([
            ChatMessage(role="tool", content="sunny", tool_call_id="call_123"),
        ])
        assert msgs[0]["tool_call_id"] == "call_123"


class TestChatMessageValidation:
    """ChatMessage field validation."""

    def test_invalid_role_rejected(self):
        """Invalid role should raise validation error."""
        with pytest.raises(ValidationError, match="invalid role"):
            ChatMessage(role="invalid_role", content="test")

    def test_tool_without_call_id_rejected(self):
        """Tool message without tool_call_id should raise validation error."""
        with pytest.raises(ValidationError, match="tool_call_id"):
            ChatMessage(role="tool", content="result")

    def test_valid_roles_accepted(self):
        """All valid roles should be accepted."""
        for role in ["system", "user", "assistant", "function", "developer"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role
        # Tool role requires tool_call_id
        msg = ChatMessage(role="tool", content="result", tool_call_id="call_123")
        assert msg.role == "tool"


class TestChatMaxTokensZeroEndpoint:
    """Integration tests for max_tokens=0 fast path in chat endpoint."""

    def test_max_tokens_zero_returns_prompt_only(self):
        """Chat endpoint with max_tokens=0 returns empty completion with prompt_tokens."""
        from yunshu_gateway.routers.chat import create_chat_completion

        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=0,
        )

        mock_request = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.state = MagicMock()
        mock_request.state.rbac_key = None
        mock_request.state.request_id = "test"

        mock_engine = MagicMock()
        mock_engine.is_loaded = False
        mock_engine.resolve_model_id.return_value = False

        with patch("yunshu_gateway.routers.chat.get_engine", return_value=mock_engine):
            import asyncio
            result = asyncio.new_event_loop().run_until_complete(
                create_chat_completion(req, mock_request)
            )
            # Should return JSONResponse with completion_tokens=0
            assert result.status_code == 200
            import json
            body = json.loads(result.body)
            assert body["usage"]["completion_tokens"] == 0
            assert body["choices"][0]["finish_reason"] == "length"


class TestSamplerFeatureSchemaW735:
    """chat/completions request models accept the vLLM/SGLang parity
    params min_tokens / ignore_eos / suppress_tokens."""

    def test_chat_request_accepts_new_fields(self):
        r = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            min_tokens=8, ignore_eos=True, suppress_tokens=[1, 2, 3],
        )
        assert r.min_tokens == 8
        assert r.ignore_eos is True
        assert r.suppress_tokens == [1, 2, 3]

    def test_chat_request_defaults(self):
        r = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
        assert r.min_tokens == 0
        assert r.ignore_eos is False
        assert r.suppress_tokens is None

    def test_completion_request_accepts_new_fields(self):
        from yunshu_gateway.routers.completions import CompletionRequest
        r = CompletionRequest(model="m", prompt="hi", min_tokens=4, ignore_eos=True,
                              suppress_tokens=[5])
        assert r.min_tokens == 4 and r.ignore_eos is True and r.suppress_tokens == [5]


class TestGuidedAliasesW736:
    """vLLM/SGLang guided_* params map onto Yunshu's grammar/json_schema."""

    def _req(self, **kw):
        return ChatCompletionRequest(
            model="m", messages=[ChatMessage(role="user", content="hi")], **kw
        )

    def test_guided_regex_maps_to_grammar(self):
        r = self._req(guided_regex=r"\d{3}-\d{4}")
        assert r.grammar == {"type": "regex", "pattern": r"\d{3}-\d{4}"}

    def test_guided_choice_maps_to_grammar(self):
        r = self._req(guided_choice=["yes", "no"])
        assert r.grammar == {"type": "choice", "choices": ["yes", "no"]}

    def test_guided_grammar_maps_to_cfg(self):
        r = self._req(guided_grammar="start: \"a\"")
        assert r.grammar == {"type": "cfg", "grammar": "start: \"a\""}

    def test_guided_json_maps_to_response_format(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        r = self._req(guided_json=schema)
        assert r.response_format == {"type": "json_schema", "json_schema": {"schema": schema}}

    def test_native_grammar_takes_priority(self):
        r = self._req(grammar={"type": "regex", "pattern": "a+"}, guided_regex="b+")
        assert r.grammar == {"type": "regex", "pattern": "a+"}  # native wins

    def test_guided_regex_flows_through_parse(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        r = self._req(guided_regex="x+")
        parsed = _parse_response_format(r.response_format, r.grammar)
        assert parsed == {"type": "regex", "pattern": "x+"}


class TestChatPromptLogprobsSchemaW744:
    def test_accepts_prompt_logprobs(self):
        r = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")], prompt_logprobs=3)
        assert r.prompt_logprobs == 3

    def test_default_none(self):
        r = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
        assert r.prompt_logprobs is None
