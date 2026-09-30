"""Malformed tool calls seen from Qwen3.x under Claude Code are repaired when unambiguous, and a
call that cannot be read is dropped instead of leaking ``<tool_call>`` markup into text."""

from __future__ import annotations

import json

import pytest

from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.tool_format import (
    INJECTED_JSON,
    JSON_MESSAGE,
    _upstream,
    parse_tool_output,
)

pytest.importorskip("mlx_vlm")

QWEN = _upstream("qwen3_coder")
FORMATS = (QWEN, INJECTED_JSON, JSON_MESSAGE)

TOOLS = [
    {
        "name": "Read",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "Bash",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    },
]


def _stream(text: str, chunk: int = 1):
    streamer = ToolCallStreamer(FORMATS, tools=TOOLS)
    out = []
    for i in range(0, len(text), chunk):
        out.extend(streamer.process_token(text[i : i + chunk]))
    out.extend(streamer.flush())
    calls = [
        (o.tool_call.name, json.loads(o.tool_call.arguments))
        for o in out
        if o.tool_call
    ]
    starts = [o.tool_call_start.name for o in out if o.tool_call_start]
    return "".join(o.text for o in out), calls, starts


def _parse(text: str):
    calls, rest = parse_tool_output(text, FORMATS, TOOLS)
    return [(c["name"], json.loads(c["arguments"])) for c in calls], rest


# The shape from the Claude Code capture: the model closes </parameter> then </tool_call>
# without </function>.
NO_FUNCTION_CLOSE = "\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/tmp/a.py\n</parameter>\n</tool_call>"
SLIP = '<tool_call>\n{"function": "Bash": "arguments": {"command": "ls -la"}}\n</tool_call>'
TWO_IN_ONE = (
    "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n"
    "<function=Read>\n<parameter=file_path>\n/b\n</parameter>\n</function>\n</tool_call>"
)


@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_missing_function_close_is_a_call_not_text(chunk):
    text, calls, starts = _stream("Reading.\n" + NO_FUNCTION_CLOSE, chunk)
    assert calls == [("Read", {"file_path": "/tmp/a.py"})]
    assert starts == ["Read"]
    assert "tool_call" not in text and "<function" not in text
    assert text.strip() == "Reading."


def test_missing_function_close_non_streaming():
    calls, rest = _parse(NO_FUNCTION_CLOSE)
    assert calls == [("Read", {"file_path": "/tmp/a.py"})]
    assert rest == ""


@pytest.mark.parametrize("chunk", [1, 5, 1000])
def test_key_value_slip_json(chunk):
    text, calls, _ = _stream(SLIP, chunk)
    assert calls == [("Bash", {"command": "ls -la"})]
    assert text.strip() == ""


def test_two_functions_in_one_tool_call():
    for chunk in (1, 1000):
        text, calls, _ = _stream(TWO_IN_ONE, chunk)
        assert calls == [
            ("Read", {"file_path": "/a"}),
            ("Read", {"file_path": "/b"}),
        ]
        assert text.strip() == ""


def test_untypable_value_still_yields_the_call():
    body = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n<parameter=limit>\nall\n</parameter>\n</function>\n</tool_call>"
    _, calls, _ = _stream(body)
    assert calls[0][0] == "Read"
    assert calls[0][1]["file_path"] == "/a"


def test_typed_values_follow_the_schema():
    body = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>"
    _, calls, _ = _stream(body, 2)
    assert calls == [("Read", {"file_path": "/a", "limit": 20})]


def test_unreadable_call_is_dropped_not_leaked():
    junk = '<tool_call>\n{"oops": [1, 2\n</tool_call>'
    for chunk in (1, 1000):
        text, calls, _ = _stream("before " + junk + " after", chunk)
        assert calls == []
        assert "tool_call" not in text
        assert text.split() == ["before", "after"]
    calls, rest = _parse("before " + junk + " after")
    assert calls == [] and "tool_call" not in rest
    assert rest.split() == ["before", "after"]


def test_prose_mentioning_the_marker_stays_visible():
    text, calls, _ = _stream("Wrap calls in the `<tool_call>` tag, then stop.")
    assert calls == []
    assert "<tool_call>" in text and text.endswith("then stop.")


def test_unterminated_call_at_end_of_output_is_read():
    body = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n"
    text, calls, _ = _stream(body, 4)
    assert calls == [("Read", {"file_path": "/a"})]
    assert text.strip() == ""


def test_calls_after_a_call_keep_parsing():
    body = NO_FUNCTION_CLOSE + "\n" + SLIP + "\n" + TWO_IN_ONE
    text, calls, _ = _stream(body, 7)
    assert [c[0] for c in calls] == ["Read", "Bash", "Read", "Read"]
    assert "tool_call" not in text


def test_tool_named_tag_with_garbled_parameter_syntax():
    # Seen from Claude Code: the model writes <Read> and a garbled parameter, no closing tags.
    body = '<tool_call>\n<Read>\n<parameter=file_path": /w/.docs/instructions.md"}'
    for chunk in (1, 1000):
        text, calls, _ = _stream("\n\n" + body, chunk)
        assert calls == [("Read", {"file_path": "/w/.docs/instructions.md"})]
        assert "tool_call" not in text and "<Read>" not in text
    calls, rest = _parse(body)
    assert calls == [("Read", {"file_path": "/w/.docs/instructions.md"})]


def test_split_json_name_and_arguments_in_two_objects():
    body = '<tool_call>\n{"function": Read}\n{"arguments": {"file_path": "/a.md", "limit": 5}}\n</tool_call>'
    for chunk in (1, 1000):
        text, calls, _ = _stream(body, chunk)
        assert calls == [("Read", {"file_path": "/a.md", "limit": 5})]
        assert "tool_call" not in text


def test_name_only_json_call_keeps_the_name():
    _, calls, _ = _stream('<tool_call>\n{"function": "Read"}\n</tool_call>')
    assert calls == [("Read", {})]


def test_parameter_tag_naming_the_tool():
    body = (
        "<tool_call>\n<parameter=Bash>\n<parameter=command>\nls -la\n</parameter>\n"
        "</tool_call>"
    )
    _, calls, _ = _stream(body, 3)
    assert calls == [("Bash", {"command": "ls -la"})]


def test_unknown_tag_call_is_dropped_not_leaked():
    text, calls, _ = _stream(
        "<tool_call>\n<Nope>\n<parameter=x>1</parameter>\n</tool_call>"
    )
    assert calls == [] and "tool_call" not in text
