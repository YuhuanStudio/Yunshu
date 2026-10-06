"""A Responses forced function choice is flat ({"type": "function", "name": ...}); the injected
tool prompt must still tell the model to call that function."""

from yunshu_gateway.routers.chat import (
    ToolDefinition,
    ToolFunction,
    _inject_tool_system_prompt,
)
from yunshu_gateway.routers.responses import _chat_tool_choice


def _prompt(tool_choice):
    tools = [ToolDefinition(type="function", function=ToolFunction(name="get_weather"))]
    msgs = _inject_tool_system_prompt(
        [{"role": "user", "content": "hi"}],
        tools,
        tool_choice=_chat_tool_choice(tool_choice),
    )
    return "\n".join(str(m.get("content")) for m in msgs)


def test_named_function_is_forced_in_prompt():
    assert "You MUST call the tool 'get_weather'" in _prompt(
        {"type": "function", "name": "get_weather"}
    )


def test_allowed_tools_mode_required():
    text = _prompt({"type": "allowed_tools", "mode": "required", "tools": []})
    assert "You MUST call at least one" in text


def test_strings_pass_through():
    assert _chat_tool_choice("auto") == "auto"
    assert _chat_tool_choice(None) is None
