"""Realtime cascade path must give declared tools / tool_choice to the text model."""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import realtime


def test_with_tools_system_prepends_tool_prompt():
    tools = [{"type": "function", "name": "get_w", "description": "weather"}]
    out = realtime._with_tools_system([{"role": "user", "content": "hi"}], tools)
    assert out[0]["role"] == "system" and "get_w" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_with_tools_system_honors_tool_choice_and_empty():
    msgs = [{"role": "user", "content": "hi"}]
    assert realtime._with_tools_system(msgs, []) == msgs
    t = [{"type": "function", "name": "f"}]
    out = realtime._with_tools_system(msgs, t, "required")
    assert "MUST call" in out[0]["content"]


def test_generate_response_injects_tools_before_engine_call():
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    inj = src.index("_with_tools_system(")
    assert inj < src.index("engine.stream_chat(")
