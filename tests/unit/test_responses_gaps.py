"""Responses API gaps closed: config echo on the Response object and its events,
truncation="auto", max_tool_calls, conversation rejected, new request fields held."""

from __future__ import annotations

import json

import pytest

from yunshu_gateway.routers.responses import (
    ResponsesRequest,
    _auto_truncate,
    _config_echo,
)
from yunshu_gateway.streaming import (
    RESPONSES_ECHO,
    format_responses_completed,
    format_responses_created,
)


def _req(**kw):
    return ResponsesRequest(model="m", input="hi", **kw)


def test_new_fields_are_held_not_dropped():
    r = _req(
        truncation="auto",
        max_tool_calls=2,
        include=["message.output_text.logprobs"],
        service_tier="flex",
        prompt_cache_key="k",
        safety_identifier="u1",
    )
    assert (r.truncation, r.max_tool_calls, r.service_tier) == ("auto", 2, "flex")
    with pytest.raises(ValueError):
        _req(truncation="sometimes")
    with pytest.raises(ValueError):
        _req(max_tool_calls=0)


def test_config_echo_shape():
    r = _req(
        instructions="be brief",
        temperature=0.2,
        max_output_tokens=64,
        tools=[{"type": "function", "name": "f", "parameters": {"type": "object"}}],
        tool_choice="required",
        reasoning={"effort": "low"},
        text={"format": {"type": "json_object"}},
        metadata={"a": "b"},
    )
    e = _config_echo(r)
    assert e["instructions"] == "be brief" and e["temperature"] == 0.2
    assert e["tools"][0]["name"] == "f" and e["tool_choice"] == "required"
    assert e["reasoning"]["effort"] == "low"
    assert e["text"]["format"] == {"type": "json_object"}
    assert e["truncation"] == "disabled" and e["service_tier"] == "default"
    assert e["metadata"] == {"a": "b"} and e["store"] is True
    d = _config_echo(_req())
    assert d["text"] == {"format": {"type": "text"}}
    assert d["tool_choice"] == "auto" and d["tools"] == []


def test_stream_events_carry_the_echo():
    tok = RESPONSES_ECHO.set(_config_echo(_req(instructions="x", max_tool_calls=3)))
    try:
        ev = format_responses_created("resp-1", "m")
        body = json.loads(ev.split("data: ", 1)[1])
        assert body["response"]["instructions"] == "x"
        assert body["response"]["max_tool_calls"] == 3
        done = format_responses_completed("resp-1", "m", [], 1, 2, 3)
        resp = json.loads(done.split("data: ", 1)[1])["response"]
        assert resp["usage"]["total_tokens"] == 3 and resp["truncation"] == "disabled"
    finally:
        RESPONSES_ECHO.reset(tok)
    plain = json.loads(format_responses_created("r", "m").split("data: ", 1)[1])
    assert "instructions" not in plain["response"]


def test_auto_truncate_drops_oldest_and_keeps_system_and_last():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "new q"},
    ]

    def count(ms):
        return 10 * len(ms)

    out, n = _auto_truncate(msgs, count(msgs), 30, count)
    assert [m["role"] for m in out] == ["system", "user"] and out[-1][
        "content"
    ] == "new q"
    assert n == 20
    # a single message that alone overflows is kept (the 400 guard answers)
    out, _ = _auto_truncate(msgs[-1:], 999, 30, count)
    assert len(out) == 1
    # dropping the oldest turn keeps a later call / result pair together
    out, _ = _auto_truncate(msgs, count(msgs), 40, count)
    assert [m["role"] for m in out] == ["system", "assistant", "tool", "user"]
    # a tool result whose call was dropped goes with it
    out, _ = _auto_truncate(msgs[:1] + msgs[2:], 40, 20, count)
    assert [m["role"] for m in out] == ["system", "user"]


def test_max_calls_limits_streamed_tool_calls():
    from yunshu_engine.tool_call_streamer import ToolCallStreamer

    tools = [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
    ]
    text = (
        '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'
        '<tool_call>{"name": "f", "arguments": {"a": 2}}</tool_call>'
    )

    def run(max_calls):
        s = ToolCallStreamer(max_calls=max_calls, tools=tools)
        starts = 0
        for ch in text:
            for o in s.process_token(ch):
                if o.tool_call_start is not None:
                    starts += 1
        for o in s.flush():
            if o.tool_call_start is not None:
                starts += 1
        return starts

    assert run(None) == 2
    assert run(1) == 1
