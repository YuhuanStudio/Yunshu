"""Tests for tool calling integration.

Covers:
- Tool calling system prompt generation (_inject_tool_system_prompt)
- OpenAI response format with tool_calls (format_openai_non_stream)
- tool_choice parameter handling

Parsing and streaming of tool calls: test_tool_format.py.
"""

from yunshu_gateway.routers.chat import (
    ChatCompletionRequest,
    ToolChoiceFunction,
    ToolDefinition,
    ToolFunction,
    _inject_tool_system_prompt,
)
from yunshu_gateway.streaming import (
    format_openai_non_stream,
)


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
        result = _inject_tool_system_prompt(
            messages, self._make_tools(), tool_choice="auto"
        )
        system = result[0]["content"]
        assert "Decide whether to call a tool" in system

    def test_tool_choice_none(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(
            messages, self._make_tools(), tool_choice="none"
        )
        system = result[0]["content"]
        assert "must NOT call any tools" in system
        assert "tool_call" not in system

    def test_tool_choice_specific_function(self):
        messages = [{"role": "user", "content": "Hi"}]
        forced = ToolChoiceFunction(
            function=ToolFunction(name="get_weather"),
        )
        result = _inject_tool_system_prompt(
            messages, self._make_tools(), tool_choice=forced
        )
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
        result = _inject_tool_system_prompt(
            messages, self._make_tools(), tool_choice="none"
        )
        assert result[0]["role"] == "system"

    def test_default_tool_choice_is_auto_like(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _inject_tool_system_prompt(
            messages, self._make_tools(), tool_choice=None
        )
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
