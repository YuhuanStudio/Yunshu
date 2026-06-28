"""Anthropic tool_result conversion — is_error honored + tool name carried.

The /v1/messages input conversion dropped the tool_result `is_error` flag → a FAILED tool result
was rendered as a normal (success) tool message, so the model proceeded on a failed/garbage
result (skipped retries, hallucinated). Also the tool message carried no `name`, so templates
that key the tool turn on the function name rendered a nameless response. Fix: prefix error
results with [tool_error], carry name via an id→name map, and json.dumps tool_use input
unconditionally (str(non-dict) produced invalid JSON).
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from yunshu_gateway.routers.anthropic import _convert_anthropic_messages


def _conv(msgs):
    return _convert_anthropic_messages(msgs)[0]


def test_is_error_marked_and_name_carried():
    out = _conv([
        NS(role="user", content="weather?"),
        NS(role="assistant", content=[{"type": "tool_use", "id": "t1", "name": "get_weather",
                                       "input": {"city": "SF"}}]),
        NS(role="user", content=[{"type": "tool_result", "tool_use_id": "t1",
                                  "content": "API timeout", "is_error": True}]),
    ])
    tool = [m for m in out if m.get("role") == "tool"][0]
    assert tool["content"].startswith("[tool_error]")
    assert "API timeout" in tool["content"]
    assert tool.get("name") == "get_weather"          # carried from the originating tool_use
    assert tool.get("tool_call_id") == "t1"
    asst = [m for m in out if m.get("tool_calls")][0]
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"city": "SF"}'  # valid JSON


def test_non_error_result_unmarked():
    out = _conv([
        NS(role="assistant", content=[{"type": "tool_use", "id": "t2", "name": "f", "input": {}}]),
        NS(role="user", content=[{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]),
    ])
    tool = [m for m in out if m.get("role") == "tool"][0]
    assert tool["content"] == "ok"
    assert not tool["content"].startswith("[tool_error]")
    assert tool.get("name") == "f"


def test_empty_error_result_still_marked():
    out = _conv([
        NS(role="assistant", content=[{"type": "tool_use", "id": "t3", "name": "g", "input": {}}]),
        NS(role="user", content=[{"type": "tool_result", "tool_use_id": "t3", "content": "",
                                  "is_error": True}]),
    ])
    tool = [m for m in out if m.get("role") == "tool"][0]
    assert tool["content"] == "[tool_error] (no detail)"
