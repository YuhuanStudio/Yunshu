"""Qwen3-Coder tool-call parsing cases ported from vLLM's tests/tool_parsers/test_qwen3coder_tool_parser.py
(Apache-2.0, vllm-project/vllm): typed parameters, anyOf/nullable schemas, the missing-closing-tag and
missing-'>' regressions, multiple calls, and content kept before the call."""

from __future__ import annotations

import json

from tests.unit.test_tool_format import _formats
from yunshu_engine.tool_format import parse_tool_output


def _tool(name, props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": props},
        },
    }


WEATHER = _tool(
    "get_current_weather",
    {
        "city": {"type": "string"},
        "state": {"type": "string"},
        "unit": {"type": "string"},
    },
)


def _args(call):
    a = call["arguments"]
    return json.loads(a) if isinstance(a, str) else a


def _call(text, tools):
    return parse_tool_output(text, _formats("qwen3_coder"), tools)


def _block(name, **params):
    ps = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{ps}</function>\n</tool_call>"


def test_no_tools_text_untouched():
    text = "This is a test response without any tool calls"
    assert _call(text, [WEATHER]) == ([], text)


def test_content_before_call_kept_and_call_parsed():
    out = "Sure! Let me check the weather for you." + _block(
        "get_current_weather", city="Dallas", state="TX", unit="fahrenheit"
    )
    calls, content = _call(out, [WEATHER])
    assert content == "Sure! Let me check the weather for you."
    assert len(calls) == 1 and calls[0]["name"] == "get_current_weather"
    assert _args(calls[0]) == {
        "city": "Dallas",
        "state": "TX",
        "unit": "fahrenheit",
    }


def test_two_calls_in_order():
    out = (
        _block("get_current_weather", city="Dallas", state="TX")
        + "\n"
        + _block("get_current_weather", city="Orlando", state="FL")
    )
    calls, content = _call(out, [WEATHER])
    assert [_args(c)["city"] for c in calls] == ["Dallas", "Orlando"]
    assert content == ""


def test_typed_parameters():
    tool = _tool(
        "test_types",
        {
            "int_param": {"type": "integer"},
            "float_param": {"type": "number"},
            "bool_param": {"type": "boolean"},
            "str_param": {"type": "string"},
            "obj_param": {"type": "object"},
        },
    )
    out = _block(
        "test_types",
        int_param="42",
        float_param="3.14",
        bool_param="true",
        str_param="hello world",
        obj_param='{"key": "value"}',
    )
    (call,), _ = _call(out, [tool])
    assert _args(call) == {
        "int_param": 42,
        "float_param": 3.14,
        "bool_param": True,
        "str_param": "hello world",
        "obj_param": {"key": "value"},
    }


def test_multiline_json_object_parameter():
    tool = _tool("calculate_area", {"dimensions": {"type": "object"}})
    out = _block("calculate_area", dimensions='{"width": 10, \n "height": 20}')
    (call,), _ = _call(out, [tool])
    assert _args(call)["dimensions"] == {"width": 10, "height": 20}


def test_anyof_nullable_types_converted_not_double_encoded():
    tool = _tool(
        "update_record",
        {
            "data": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "tags": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ]
            },
            "t": {"type": ["integer", "null"]},
        },
    )
    out = _block(
        "update_record",
        data='{"key": "value", "count": 42}',
        n="5",
        tags='["a", "b"]',
        t="7",
    )
    (call,), _ = _call(out, [tool])
    assert _args(call) == {
        "data": {"key": "value", "count": 42},
        "n": 5,
        "tags": ["a", "b"],
        "t": 7,
    }


def test_missing_closing_parameter_tag_still_parses():
    out = (
        "Let me check the weather for you:\n<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nDallas\n<parameter=state>\nTX\n</parameter>\n"
        "<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>"
    )
    calls, content = _call(out, [WEATHER])
    assert len(calls) == 1
    assert _args(calls[0]) == {
        "city": "Dallas",
        "state": "TX",
        "unit": "fahrenheit",
    }
    assert "Let me check the weather for you:" in content


def test_function_tag_without_gt_does_not_crash_and_good_call_survives():
    out = "<tool_call>\n<function=bad_func_no_gt\n</function>\n</tool_call>\n" + _block(
        "get_current_weather", city="Dallas", state="TX"
    )
    calls, _ = _call(out, [WEATHER])
    assert [c["name"] for c in calls] == ["get_current_weather"]
    assert _args(calls[0])["city"] == "Dallas"
    # a lone malformed call is a no-op, not an exception
    _call(
        "<tool_call>\n<function=get_current_weather\n<parameter=city>Dallas</parameter>\n</function>\n</tool_call>",
        [WEATHER],
    )


def test_string_argument_not_double_serialized():
    tool = _tool("echo", {"text": {"type": "string"}})
    (call,), _ = _call(_block("echo", text='{"a": 1}'), [tool])
    assert _args(call)["text"] == '{"a": 1}'
