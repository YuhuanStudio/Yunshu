"""Tests for ResponsesRequest Pydantic validator (model-free, pure logic).

Covers each ValueError branch in ResponsesRequest.validate_request:
- empty/whitespace model
- empty/whitespace input string
- empty input list
- bad response_format.type
- bad grammar.type
- stop list overflow (>16)
- empty individual stop sequence

Also covers the max_completion_tokens -> max_output_tokens aliasing.
"""

import pytest
from pydantic import ValidationError

from yunshu_gateway.routers.responses import ResponsesRequest


def _err_text(exc_info):
    return str(exc_info.value)


class TestResponsesRequestValidation:
    def test_valid_minimal(self):
        req = ResponsesRequest(model="qwen", input="hello")
        assert req.model == "qwen"
        assert req.input == "hello"

    def test_empty_model(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="", input="hi")
        assert "model" in _err_text(exc)

    def test_whitespace_model(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="   ", input="hi")
        assert "model" in _err_text(exc)

    def test_empty_input_string(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="qwen", input="")
        assert "input" in _err_text(exc)

    def test_whitespace_input_string(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="qwen", input="   \t\n")
        assert "input" in _err_text(exc)

    def test_empty_input_list(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="qwen", input=[])
        assert "input" in _err_text(exc)

    def test_bad_response_format_type(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(
                model="qwen",
                input="hi",
                response_format={"type": "bogus"},
            )
        assert "response_format.type" in _err_text(exc)

    def test_valid_response_format_types(self):
        for t in ("json_object", "json_schema", "text"):
            req = ResponsesRequest(
                model="qwen", input="hi", response_format={"type": t}
            )
            assert req.response_format["type"] == t

    def test_bad_grammar_type(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(
                model="qwen",
                input="hi",
                grammar={"type": "yaml"},
            )
        assert "grammar.type" in _err_text(exc)

    def test_valid_grammar_types(self):
        for t in ("json", "regex", "choice", "cfg"):
            req = ResponsesRequest(model="qwen", input="hi", grammar={"type": t})
            assert req.grammar["type"] == t

    def test_stop_overflow(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(
                model="qwen",
                input="hi",
                stop=[f"s{i}" for i in range(17)],
            )
        assert "stop" in _err_text(exc)

    def test_stop_exactly_16_ok(self):
        req = ResponsesRequest(
            model="qwen",
            input="hi",
            stop=[f"s{i}" for i in range(16)],
        )
        assert len(req.stop) == 16

    def test_empty_stop_sequence(self):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="qwen", input="hi", stop=["ok", ""])
        assert "stop" in _err_text(exc)

    def test_max_completion_tokens_aliases_max_output(self):
        req = ResponsesRequest(model="qwen", input="hi", max_completion_tokens=999)
        assert req.max_output_tokens == 999

    def test_input_list_of_dict_blocks(self):
        req = ResponsesRequest(
            model="qwen",
            input=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(req.input, list)


class TestResponsesToolConversation:
    """tool-conversation continuation via the Responses input list.
    Previously a function_call_output item was rejected (content required) /
    mangled into an empty user message; function_call was dropped from chaining."""

    def test_function_call_output_item_accepted_and_mapped(self):
        from yunshu_gateway.routers.responses import _convert_to_messages

        req = ResponsesRequest(
            model="m",
            input=[
                {"type": "message", "role": "user", "content": "weather in SF?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "72F sunny",
                },
            ],
        )
        msgs = _convert_to_messages(req)
        assert msgs[0] == {"role": "user", "content": "weather in SF?"}
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["tool_calls"][0]["id"] == "call_1"
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "72F sunny",
        }

    def test_message_item_still_requires_content(self):
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input=[{"type": "message", "role": "user"}])

    def test_plain_string_input_unchanged(self):
        from yunshu_gateway.routers.responses import _convert_to_messages

        req = ResponsesRequest(model="m", input="hi")
        assert _convert_to_messages(req) == [{"role": "user", "content": "hi"}]


class TestResponsesMultimodalOrder:
    """_extract_input_text must preserve document order (text/image)
    and unwrap a nested image_url dict."""

    def test_text_image_text_order_preserved(self):
        from yunshu_gateway.routers.responses import _extract_input_text

        r = _extract_input_text(
            [
                {"type": "input_text", "text": "A"},
                {"type": "input_image", "image_url": "data:image/png;base64,xxx"},
                {"type": "input_text", "text": "B"},
            ]
        )
        assert [b["type"] for b in r] == ["text", "image_url", "text"]
        assert r[0]["text"] == "A" and r[2]["text"] == "B"

    def test_nested_image_url_dict_unwrapped(self):
        from yunshu_gateway.routers.responses import _extract_input_text

        r = _extract_input_text(
            [
                {
                    "type": "input_image",
                    "image_url": {"url": "data:image/png;base64,yyy"},
                },
            ]
        )
        assert r[0]["image_url"]["url"] == "data:image/png;base64,yyy"

    def test_text_only_returns_string(self):
        from yunshu_gateway.routers.responses import _extract_input_text

        assert _extract_input_text([{"type": "input_text", "text": "hello"}]) == "hello"
