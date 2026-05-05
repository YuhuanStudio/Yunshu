"""Tests for tool calling integration.

Covers:
- extract_tool_calls() with various formats (Hermes tags, code blocks, direct JSON)
- clean_tool_call_markup() removes tags correctly
- ToolCallStreamer incremental detection
- Tool calling system prompt generation (_inject_tool_system_prompt)
- OpenAI response format with tool_calls (format_openai_non_stream)
- Streaming tool call detection edge cases (partial tags, multiple calls)
- tool_choice parameter handling
"""

import json

import pytest

from yunshu_gateway.streaming import (
    clean_tool_call_markup,
    extract_tool_calls,
    format_openai_non_stream,
)
from yunshu_gateway.routers.chat import (
    ChatCompletionRequest,
    ToolChoiceFunction,
    ToolDefinition,
    ToolFunction,
    _inject_tool_system_prompt,
)
from yunshu_engine.tool_call_streamer import (
    StreamOutput,
    StreamState,
    ToolCallResult,
    ToolCallStreamer,
)


# ── extract_tool_calls ──


class TestExtractToolCallsFormats:
    """Test extract_tool_calls with all supported formats."""

    def test_hermes_self_closing_tag(self):
        text = '<tool_call/>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call/>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        args = json.loads(calls[0]["arguments"])
        assert args == {"city": "SF"}

    def test_hermes_with_space_in_tag(self):
        text = '<tool_call >{"name": "calc", "arguments": {}}\n</tool_call >'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "calc"

    def test_multiple_hermes_calls(self):
        text = (
            '<tool_call/>{"name": "fn1", "arguments": {"a": 1}}</tool_call/>'
            '<tool_call/>{"name": "fn2", "arguments": {"b": 2}}</tool_call/>'
        )
        calls = extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "fn1"
        assert calls[1]["name"] == "fn2"

    def test_code_block_json(self):
        text = '```json\n{"name": "search", "arguments": {"q": "test"}}\n```'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"

    def test_code_block_python(self):
        text = '```python\n{"name": "run_code", "arguments": {"code": "print(1)"}}\n```'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "run_code"

    def test_code_block_tool(self):
        text = '```tool\n{"name": "my_tool", "arguments": {"x": 1}}\n```'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "my_tool"

    def test_code_block_list_of_calls(self):
        text = '```json\n[{"name": "fn1", "arguments": {}}, {"name": "fn2", "arguments": {}}]\n```'
        calls = extract_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        assert extract_tool_calls("just normal text") == []
        assert extract_tool_calls("") == []

    def test_malformed_json_in_tags(self):
        text = '<tool_call/>not valid json</tool_call/>'
        calls = extract_tool_calls(text)
        assert calls == []

    def test_arguments_as_parameters(self):
        text = '<tool_call/>{"name": "fn", "parameters": {"x": 1}}</tool_call/>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["arguments"])
        assert args == {"x": 1}

    def test_text_surrounding_tool_call(self):
        text = (
            'Let me check.\n'
            '<tool_call/>{"name": "lookup", "arguments": {"id": 42}}</tool_call/>\n'
            'Here is the result.'
        )
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "lookup"


# ── clean_tool_call_markup ──


class TestCleanToolCallMarkup:
    def test_removes_tool_call_tags(self):
        text = 'before<tool_call/>{"name": "fn"}</tool_call/>after'
        cleaned = clean_tool_call_markup(text)
        assert "<tool_call" not in cleaned
        assert "</tool_call" not in cleaned
        assert "before" in cleaned
        assert "after" in cleaned

    def test_cleans_extra_newlines(self):
        text = "hello\n\n\n\nworld"
        cleaned = clean_tool_call_markup(text)
        assert cleaned == "hello\n\nworld"

    def test_no_markup(self):
        text = "just text"
        cleaned = clean_tool_call_markup(text)
        assert cleaned == "just text"

    def test_only_tool_call(self):
        text = '<tool_call/>{"name": "fn"}</tool_call/>'
        cleaned = clean_tool_call_markup(text)
        assert cleaned == ""


# ── ToolCallStreamer ──


class TestToolCallStreamer:
    """Test incremental tool call detection from streaming tokens."""

    def test_plain_text_passes_through(self):
        streamer = ToolCallStreamer(flush_threshold=5)
        outputs = streamer.process_token("Hello ")
        assert all(o.text or o.tool_call is None for o in outputs)
        streamer.flush()
        full_text = "".join(o.text for o in streamer.flush())
        assert full_text == ""

    def test_single_tool_call_in_one_chunk(self):
        streamer = ToolCallStreamer()
        chunk = '<tool_call/>{"name": "weather", "arguments": {"city": "NYC"}}</tool_call/>'
        outputs = streamer.process_token(chunk)
        tool_calls = [o for o in outputs if o.tool_call is not None]
        assert len(tool_calls) >= 1
        tc = tool_calls[0].tool_call
        assert tc.name == "weather"
        args = json.loads(tc.arguments)
        assert args == {"city": "NYC"}

    def test_tool_call_split_across_tokens(self):
        streamer = ToolCallStreamer(flush_threshold=20)
        all_outputs: list[StreamOutput] = []
        tokens = [
            '<tool_call',
            '/>',
            '{"name": "weather", ',
            '"arguments": {"city": "NYC"}}',
            '</tool_call/>',
        ]
        for token in tokens:
            outputs = streamer.process_token(token)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.name == "weather"

    def test_text_before_and_after_tool_call(self):
        streamer = ToolCallStreamer(flush_threshold=10)
        all_outputs: list[StreamOutput] = []
        for token in [
            "Let me check. ",
            '<tool_call/>{"name": "fn", "arguments": {}}</tool_call/>',
            " Done.",
        ]:
            outputs = streamer.process_token(token)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        texts = "".join(o.text for o in all_outputs if o.text)
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.name == "fn"
        assert "Let me check." in texts
        assert "Done." in texts

    def test_multiple_tool_calls_in_sequence(self):
        streamer = ToolCallStreamer()
        all_outputs: list[StreamOutput] = []
        text = (
            '<tool_call/>{"name": "fn1", "arguments": {"a": 1}}</tool_call/>'
            '<tool_call/>{"name": "fn2", "arguments": {"b": 2}}</tool_call/>'
        )
        for char in text:
            outputs = streamer.process_token(char)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_call.name == "fn1"
        assert tool_calls[1].tool_call.name == "fn2"

    def test_call_ids_are_unique(self):
        streamer = ToolCallStreamer()
        all_outputs: list[StreamOutput] = []
        text = (
            '<tool_call/>{"name": "a", "arguments": {}}</tool_call/>'
            '<tool_call/>{"name": "b", "arguments": {}}</tool_call/>'
        )
        for char in text:
            outputs = streamer.process_token(char)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_call.id != tool_calls[1].tool_call.id

    def test_flush_threshold_forces_text_emission(self):
        streamer = ToolCallStreamer(flush_threshold=10)
        outputs = streamer.process_token(
            "This is a long piece of text that exceeds the threshold"
        )
        texts = "".join(o.text for o in outputs if o.text)
        assert len(texts) > 0

    def test_partial_tag_then_completion(self):
        streamer = ToolCallStreamer(flush_threshold=30)
        all_outputs: list[StreamOutput] = []
        tokens = [
            "<tool_",
            "call/>",
            '{"name": "test_fn", "arguments": {"x": 1}}',
            "</tool_call/>",
        ]
        for token in tokens:
            outputs = streamer.process_token(token)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.name == "test_fn"

    def test_flush_at_end_of_stream(self):
        streamer = ToolCallStreamer(flush_threshold=100)
        streamer.process_token("Hello")
        outputs = streamer.flush()
        texts = "".join(o.text for o in outputs if o.text)
        assert "Hello" in texts

    def test_flush_incomplete_tool_call(self):
        streamer = ToolCallStreamer()
        streamer.process_token('<tool_call/>{"name": "fn", "arguments": ')
        outputs = streamer.flush()
        # Should not crash
        assert len(outputs) >= 0

    def test_reset_clears_state(self):
        streamer = ToolCallStreamer()
        streamer.process_token("some text")
        streamer.process_token('<tool_call/>{"name": "fn"')
        streamer.reset()
        assert streamer.state == StreamState.TEXT
        assert streamer._buffer == ""
        assert streamer._json_buffer == ""

    def test_empty_token(self):
        streamer = ToolCallStreamer()
        outputs = streamer.process_token("")
        assert isinstance(outputs, list)

    def test_no_tool_call_at_all(self):
        streamer = ToolCallStreamer(flush_threshold=5)
        all_outputs: list[StreamOutput] = []
        for word in ["Hello", " ", "world", "!"]:
            outputs = streamer.process_token(word)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 0
        texts = "".join(o.text for o in all_outputs if o.text)
        assert "Hello world!" in texts


# ── _inject_tool_system_prompt ──


class TestInjectToolSystemPrompt:
    """Test tool system prompt injection with tool_choice support."""

    def _make_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather for a city",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            ),
            ToolDefinition(
                function=ToolFunction(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ),
        ]

    def test_injects_into_new_system_message(self):
        messages = [{"role": "user", "content": "What's the weather?"}]
        result = _inject_tool_system_prompt(messages, self._make_tools())
        assert result[0]["role"] == "system"
        assert "get_weather" in result[0]["content"]
        assert "search" in result[0]["content"]
        assert "tool_call" in result[0]["content"]

    def test_appends_to_existing_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _inject_tool_system_prompt(messages, self._make_tools())
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "get_weather" in result[0]["content"]

    def test_tool_choice_auto(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(messages, self._make_tools(), tool_choice="auto")
        system = result[0]["content"]
        assert "Decide whether to call a tool" in system

    def test_tool_choice_none(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(messages, self._make_tools(), tool_choice="none")
        system = result[0]["content"]
        assert "must NOT call any tools" in system
        assert "tool_call" not in system

    def test_tool_choice_specific_function(self):
        messages = [{"role": "user", "content": "Hi"}]
        forced = ToolChoiceFunction(
            function=ToolFunction(name="get_weather"),
        )
        result = _inject_tool_system_prompt(messages, self._make_tools(), tool_choice=forced)
        system = result[0]["content"]
        assert "MUST call the tool 'get_weather'" in system

    def test_no_tools_returns_unchanged(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(messages, [])
        assert result == messages

    def test_parameters_included(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(messages, self._make_tools())
        system = result[0]["content"]
        assert "Parameters:" in system
        assert "city" in system

    def test_does_not_mutate_original(self):
        messages = [{"role": "user", "content": "Hi"}]
        original = list(messages)
        _inject_tool_system_prompt(messages, self._make_tools())
        assert messages == original

    def test_tool_choice_none_still_creates_system_message(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(messages, self._make_tools(), tool_choice="none")
        assert result[0]["role"] == "system"

    def test_default_tool_choice_is_auto_like(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(messages, self._make_tools(), tool_choice=None)
        system = result[0]["content"]
        assert "Decide whether to call a tool" in system


# ── format_openai_non_stream with tool_calls ──


class TestOpenAIResponseWithToolCalls:
    def test_response_with_tool_calls(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test",
            content=None,
            prompt_tokens=10,
            completion_tokens=20,
            finish_reason="stop",
            tool_calls=[{"name": "get_weather", "arguments": '{"city": "SF"}'}],
        )
        assert resp["choices"][0]["finish_reason"] == "tool_calls"
        tc = resp["choices"][0]["message"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "SF"}'
        assert tc["id"].startswith("call_")

    def test_response_without_tool_calls(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in resp["choices"][0]["message"]

    def test_multiple_tool_calls(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test",
            content=None,
            prompt_tokens=5,
            completion_tokens=10,
            tool_calls=[
                {"name": "fn1", "arguments": '{"a": 1}'},
                {"name": "fn2", "arguments": '{"b": 2}'},
            ],
        )
        tcs = resp["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 2
        assert tcs[0]["function"]["name"] == "fn1"
        assert tcs[1]["function"]["name"] == "fn2"
        assert tcs[0]["id"] != tcs[1]["id"]

    def test_tool_call_ids_format(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-abc",
            model="test",
            content=None,
            prompt_tokens=5,
            completion_tokens=10,
            tool_calls=[{"name": "fn", "arguments": "{}"}],
        )
        tc_id = resp["choices"][0]["message"]["tool_calls"][0]["id"]
        assert tc_id.startswith("call_")

    def test_content_null_with_tool_calls(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test",
            content=None,
            prompt_tokens=5,
            completion_tokens=10,
            tool_calls=[{"name": "fn", "arguments": "{}"}],
        )
        assert resp["choices"][0]["message"]["content"] is None

    def test_usage_correct(self):
        resp = format_openai_non_stream(
            completion_id="chatcmpl-123",
            model="test",
            content="hi",
            prompt_tokens=100,
            completion_tokens=50,
            tool_calls=[{"name": "fn", "arguments": "{}"}],
        )
        assert resp["usage"]["total_tokens"] == 150


# ── ChatCompletionRequest tool_choice parsing ──


class TestChatCompletionRequestToolChoice:
    def test_tool_choice_auto_string(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"

    def test_tool_choice_none_string(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="none",
        )
        assert req.tool_choice == "none"

    def test_tool_choice_function(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice=ToolChoiceFunction(
                function=ToolFunction(name="get_weather"),
            ),
        )
        assert isinstance(req.tool_choice, ToolChoiceFunction)
        assert req.tool_choice.function.name == "get_weather"

    def test_tool_choice_default_none(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.tool_choice is None

    def test_tools_parameter(self):
        tools = [
            ToolDefinition(
                function=ToolFunction(
                    name="fn1",
                    description="A function",
                    parameters={"type": "object", "properties": {"x": {"type": "int"}}},
                )
            )
        ]
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert len(req.tools) == 1
        assert req.tools[0].function.name == "fn1"


# ── ToolCallStreamer edge cases ──


class TestToolCallStreamerEdgeCases:
    def test_tag_appears_in_regular_text(self):
        streamer = ToolCallStreamer(flush_threshold=20)
        all_outputs: list[StreamOutput] = []
        for token in ["The format is ", "<tool_call", " not real"]:
            outputs = streamer.process_token(token)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 0

    def test_rapid_token_by_token(self):
        streamer = ToolCallStreamer(flush_threshold=5)
        all_outputs: list[StreamOutput] = []
        full_text = (
            'Hello <tool_call/>{"name": "fn", "arguments": {"x": 1}}'
            '</tool_call/> World'
        )
        for char in full_text:
            outputs = streamer.process_token(char)
            all_outputs.extend(outputs)
        all_outputs.extend(streamer.flush())
        texts = "".join(o.text for o in all_outputs if o.text)
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.name == "fn"
        assert "Hello" in texts
        assert "World" in texts

    def test_arguments_is_string(self):
        streamer = ToolCallStreamer()
        chunk = '<tool_call/>{"name": "fn", "arguments": "already serialized"}</tool_call/>'
        outputs = streamer.process_token(chunk)
        all_outputs = outputs + streamer.flush()
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1

    def test_empty_arguments(self):
        streamer = ToolCallStreamer()
        chunk = '<tool_call/>{"name": "fn", "arguments": {}}</tool_call/>'
        outputs = streamer.process_token(chunk)
        all_outputs = outputs + streamer.flush()
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.arguments == "{}"

    def test_custom_call_id_prefix(self):
        streamer = ToolCallStreamer(call_id_prefix="tc_")
        chunk = '<tool_call/>{"name": "fn", "arguments": {}}</tool_call/>'
        outputs = streamer.process_token(chunk)
        all_outputs = outputs + streamer.flush()
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call.id.startswith("tc_")

    def test_unicode_in_arguments(self):
        streamer = ToolCallStreamer()
        chunk = '<tool_call/>{"name": "search", "arguments": {"q": "test"}}</tool_call/>'
        outputs = streamer.process_token(chunk)
        all_outputs = outputs + streamer.flush()
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].tool_call.arguments)
        assert args["q"] == "test"

    def test_nested_json_in_arguments(self):
        streamer = ToolCallStreamer()
        chunk = '<tool_call/>{"name": "fn", "arguments": {"a": {"b": [1, 2, 3]}}}</tool_call/>'
        outputs = streamer.process_token(chunk)
        all_outputs = outputs + streamer.flush()
        tool_calls = [o for o in all_outputs if o.tool_call is not None]
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].tool_call.arguments)
        assert args["a"]["b"] == [1, 2, 3]
