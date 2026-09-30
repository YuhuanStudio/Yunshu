"""Messages tools reach a template that renders them (Qwen3.x) natively instead of as an injected JSON prompt."""

from __future__ import annotations

from yunshu_gateway.routers import anthropic


def _req(**kw):
    return anthropic.AnthropicMessagesRequest(
        model="m",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "name": "Bash",
                "description": "run a command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
            {"name": "NoSchema"},
        ],
        **kw,
    )


class NativeEngine:
    def supports_native_tools(self):
        return True


class PlainEngine:
    def supports_native_tools(self):
        return False


def test_native_plan_for_an_engine_that_renders_tools():
    req = _req()
    assert anthropic._apply_native_tools(req, NativeEngine()) is True
    tools = req._native_tools
    assert tools[0] == {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
    # a tool without a schema still gets a valid empty object schema
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert anthropic._native_kw(req) == {"tools": tools}


def test_engine_without_native_tools_keeps_the_injected_prompt():
    req = _req()
    assert anthropic._apply_native_tools(req, PlainEngine()) is False
    assert anthropic._native_kw(req) == {}
    msgs = anthropic._inject_tool_prompt(
        [{"role": "user", "content": "hi"}], "\n\nYou have access to tools"
    )
    assert msgs[0]["role"] == "system" and "access to tools" in msgs[0]["content"]
    msgs = anthropic._inject_tool_prompt(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        " +tools",
    )
    assert msgs[0]["content"] == "sys +tools"


def test_only_auto_tool_choice_is_deferred():
    assert anthropic._tool_choice_is_auto(None)
    assert anthropic._tool_choice_is_auto("auto")
    assert anthropic._tool_choice_is_auto({"type": "auto"})
    assert not anthropic._tool_choice_is_auto({"type": "any"})
    assert not anthropic._tool_choice_is_auto({"type": "tool", "name": "Bash"})
    assert not anthropic._tool_choice_is_auto("any")
