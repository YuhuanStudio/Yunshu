"""Realtime cascade path must give declared tools / tool_choice to the text model."""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import realtime


def test_with_tools_system_prepends_tool_prompt():
    tools = [{"type": "function", "name": "get_w", "description": "weather"}]
    out = realtime._with_tools_system([{"role": "user", "content": "hi"}], tools)
    assert out[0]["role"] == "system" and "get_w" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_with_tools_system_joins_the_existing_system_turn():
    # chat templates take one system turn, at the start
    tools = [{"type": "function", "name": "get_w"}]
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    out = realtime._with_tools_system(msgs, tools)
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"].startswith("Be brief.") and "get_w" in out[0]["content"]


def test_with_tools_system_honors_tool_choice_and_empty():
    msgs = [{"role": "user", "content": "hi"}]
    assert realtime._with_tools_system(msgs, []) == msgs
    t = [{"type": "function", "name": "f"}]
    out = realtime._with_tools_system(msgs, t, "required")
    assert "MUST call" in out[0]["content"]


def test_native_tool_shapes():
    tools = [
        {
            "type": "function",
            "name": "get_w",
            "description": "d",
            "parameters": {"type": "object"},
        }
    ]
    assert realtime._chat_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "get_w",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert realtime._chat_tool_choice({"type": "function", "name": "get_w"}) == {
        "type": "function",
        "function": {"name": "get_w"},
    }
    assert realtime._chat_tool_choice("required") == "required"


def test_generate_response_gives_tools_to_both_engine_paths():
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    assert src.index("_with_tools_system(") < src.index("engine.stream_chat(")
    stream = src.index("engine.generate_stream(")
    assert "**(_native_tools or {})" in src[stream : stream + 2000]
