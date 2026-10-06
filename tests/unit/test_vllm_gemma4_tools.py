"""Gemma 4 tool-call cases from vLLM tests/tool_parsers/test_gemma4_tool_parser.py (Apache-2.0)."""

from __future__ import annotations

import json

import pytest

from tests.unit.test_tool_format import _formats
from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.tool_format import parse_tool_output

Q = '<|"|>'


def _p(text, tools=None):
    calls, content = parse_tool_output(text, _formats("gemma4"), tools)
    out = []
    for c in calls:
        a = c["arguments"]
        out.append((c["name"], json.loads(a) if isinstance(a, str) else a))
    return out, content


def _call(name, body):
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


CASES = {
    "single": (
        _call("get_weather", f"location:{Q}London{Q}"),
        [("get_weather", {"location": "London"})],
    ),
    "comma_in_string": (
        _call("f", f"location:{Q}Paris, France{Q}"),
        [("f", {"location": "Paris, France"})],
    ),
    "mixed_types": (
        _call("set_status", f"is_active:true,count:42,score:3.14,name:{Q}x{Q},z:null"),
        [
            (
                "set_status",
                {"is_active": True, "count": 42, "score": 3.14, "name": "x", "z": None},
            )
        ],
    ),
    "nested": (
        _call(
            "complex_function", f"nested:{{inner:{Q}value{Q}}},list:[{Q}a{Q},{Q}b{Q}]"
        ),
        [("complex_function", {"nested": {"inner": "value"}, "list": ["a", "b"]})],
    ),
    "delimited_keys": (
        _call("f", f"{Q}name{Q}:{Q}Alice{Q},count:42"),
        [("f", {"name": "Alice", "count": 42})],
    ),
    "hyphen_name": (
        _call("get-weather", f"location:{Q}London{Q}"),
        [("get-weather", {"location": "London"})],
    ),
    "dotted_name": (
        _call("weather.get", f"location:{Q}London{Q}"),
        [("weather.get", {"location": "London"})],
    ),
    "no_args": (_call("get_status", ""), [("get_status", {})]),
    "bare_opener": (
        f"<|tool_call>:get_weather{{location:{Q}London{Q}}}<tool_call|>",
        [("get_weather", {"location": "London"})],
    ),
    "two_calls": (
        _call("a", f"x:{Q}1{Q}") + _call("b", f"y:{Q}2{Q}"),
        [("a", {"x": "1"}), ("b", {"y": "2"})],
    ),
    "incomplete": (
        f"<|tool_call>call:get_weather{{location:{Q}London",
        [("get_weather", {"location": "London"})],
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_parse(name):
    text, want = CASES[name]
    got, content = _p(text)
    assert got == want, (got, content)


def test_text_before_call_kept():
    text = "Let me check. " + CASES["single"][0]
    got, content = _p(text)
    assert got == CASES["single"][1] and content.strip() == "Let me check."


def test_no_tool_calls_untouched():
    t = "Hello, how can I help you today?"
    assert _p(t) == ([], t)


def _stream(chunks):
    s = ToolCallStreamer(_formats("gemma4"))
    outs = []
    for c in chunks:
        outs += s.process_token(c)
    outs += s.flush()
    content = "".join(o.text or "" for o in outs)
    calls = [
        (o.tool_call.name, json.loads(o.tool_call.arguments))
        for o in outs
        if o.tool_call is not None
    ]
    return content, calls


@pytest.mark.parametrize(
    "name",
    ["single", "mixed_types", "nested", "two_calls", "no_args", "delimited_keys"],
)
def test_any_chunking_equals_one_shot(name):
    text = "Hi. " + CASES[name][0]
    want = _stream([text])
    assert want[1] == CASES[name][1]
    for i in range(1, len(text)):
        assert _stream([text[:i], text[i:]]) == want, i
    assert _stream(list(text)) == want


def test_html_argument_not_duplicated():
    text = _call("f", f"html:{Q}<b>x</b> <tool{Q}")
    content, calls = _stream(list(text))
    assert content == "" and len(calls) == 1
