"""Per-response tool-call parse outcome lands on the request record and in the finished-request ring."""

from __future__ import annotations

import logging

from yunshu_engine.request_tracker import current_request_info
from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.tool_format import parse_tool_output, tool_formats
from yunshu_gateway import x_yunshu

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }
]
GOOD = '<tool_call>\n{"name": "shell", "arguments": {"cmd": "ls"}}\n</tool_call>'
BAD = "<tool_call>\n{not a call at all\n</tool_call>"


def _info():
    info = x_yunshu.RequestInfo(request_id="r", method="POST", path="/v1/responses")
    return info, current_request_info.set(info)


def _fmts():
    return tool_formats(None)


def test_nonstream_counts_calls_and_failed_markup():
    info, tok = _info()
    try:
        calls, _ = parse_tool_output("hi " + GOOD, _fmts(), TOOLS)
        assert len(calls) == 1
        assert (info.tool_calls_parsed, info.tool_markup_failed) == (1, 0)
        calls, _ = parse_tool_output("sure " + BAD, _fmts(), TOOLS)
        assert calls == []
        assert (info.tool_calls_parsed, info.tool_markup_failed) == (1, 1)
    finally:
        current_request_info.reset(tok)


def test_stream_records_on_flush():
    info, tok = _info()
    try:
        s = ToolCallStreamer(_fmts(), tools=TOOLS)
        for ch in "ok " + BAD:
            s.process_token(ch)
        s.flush()
        assert (info.tool_calls_parsed, info.tool_markup_failed) == (0, 1)
    finally:
        current_request_info.reset(tok)


def test_plain_text_records_zero_and_ring_has_fields(caplog):
    info, tok = _info()
    try:
        parse_tool_output("just text", _fmts(), TOOLS)
    finally:
        current_request_info.reset(tok)
    info.tool_calls_parsed, info.tool_markup_failed = 2, 1
    with caplog.at_level(logging.INFO, logger="yunshu.gateway.response"):
        x_yunshu.record_done(info, {})
    row = x_yunshu.registry.recent_cursor(limit=1)[0][0]
    assert row["tool_calls_parsed"] == 2 and row["tool_markup_failed"] == 1
    assert "tool_markup_failed=1" in caplog.text and "finish_reason=" in caplog.text
