"""Native tool wire formats read by tool_family_parsers: pure text in, calls out."""

from __future__ import annotations

import json

import pytest

from yunshu_engine import tool_family_parsers as tfp

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    }
]


def _args(call):
    return json.loads(call["arguments"])


def test_pythonic_reads_a_call_list_and_json_literals():
    calls = tfp.pythonic(
        '[get_weather(city="Paris", days=3), ping(verbose=true, note=null)]', TOOLS
    )
    assert [c["name"] for c in calls] == ["get_weather", "ping"]
    assert _args(calls[0]) == {"city": "Paris", "days": 3}
    assert _args(calls[1]) == {"verbose": True, "note": None}


def test_pythonic_strips_python_tag_prefix():
    (call,) = tfp.pythonic('<|python_tag|>get_weather(city="Oslo")', TOOLS)
    assert call["name"] == "get_weather"
    assert _args(call) == {"city": "Oslo"}


@pytest.mark.parametrize(
    "body",
    [
        'get_weather("Paris")',  # positional argument
        "get_weather(**kwargs)",  # expansion
        "get_weather(city=os.getcwd)",  # not a JSON literal: never evaluated
        "obj.method(city='x')",  # attribute call, not a plain name
        "[]",  # nothing to call
        "get_weather(city=1, city=2)",  # duplicate keyword
    ],
)
def test_pythonic_rejects_anything_that_is_not_a_literal_call(body):
    with pytest.raises((ValueError, SyntaxError)):
        tfp.pythonic(body, TOOLS)


def test_glm_key_value_pairs_are_typed_by_the_tool_schema():
    body = (
        "get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
        "<arg_key>days</arg_key>\n<arg_value>3</arg_value>"
    )
    (call,) = tfp.glm(body, TOOLS)
    assert call["name"] == "get_weather"
    assert _args(call) == {"city": "Paris", "days": 3}


def test_glm_name_only_and_json_body():
    assert _args(tfp.glm("ping", TOOLS)[0]) == {}
    (call,) = tfp.glm('get_weather {"city": "Rome"}', TOOLS)
    assert _args(call) == {"city": "Rome"}


@pytest.mark.parametrize(
    "body",
    [
        "",  # no function name
        "get_weather garbage",  # text that is not an arg pair
        "get_weather<arg_key>a</arg_key><arg_value>1</arg_value> trailing",
        "get_weather<arg_key>a</arg_key><arg_value>1</arg_value>"
        "<arg_key>a</arg_key><arg_value>2</arg_value>",  # duplicate key
    ],
)
def test_glm_rejects_malformed_bodies(body):
    with pytest.raises(ValueError):
        tfp.glm(body, TOOLS)


def _dsml(name, params):
    inner = "".join(
        f'<｜DSML｜parameter name="{k}" string="{s}">{v}</｜DSML｜parameter>'
        for k, s, v in params
    )
    return f'<｜DSML｜invoke name="{name}">{inner}</｜DSML｜invoke>'


def test_dsml_string_flag_decides_between_text_and_json():
    body = _dsml("get_weather", [("city", "true", "Paris"), ("days", "false", "3")])
    body += _dsml("ping", [])
    calls = tfp.dsml(body, TOOLS)
    assert [c["name"] for c in calls] == ["get_weather", "ping"]
    assert _args(calls[0]) == {"city": "Paris", "days": 3}
    assert _args(calls[1]) == {}


def test_dsml_rejects_duplicates_stray_text_and_empty_bodies():
    dup = _dsml("f", [("a", "true", "1"), ("a", "true", "2")])
    with pytest.raises(ValueError, match="duplicate"):
        tfp.dsml(dup, TOOLS)
    stray = _dsml("f", [("a", "true", "1")]).replace(
        "</｜DSML｜parameter>", "</｜DSML｜parameter> junk"
    )
    with pytest.raises(ValueError, match="malformed"):
        tfp.dsml(stray, TOOLS)
    with pytest.raises(ValueError, match="missing"):
        tfp.dsml("no invoke here", TOOLS)


def _kimi(name, idx, args):
    return (
        f"<|tool_call_begin|>functions.{name}:{idx} "
        f"<|tool_call_argument_begin|>{args}<|tool_call_end|>"
    )


def test_kimi_reads_consecutive_calls_with_json_arguments():
    body = _kimi("get_weather", 0, '{"city": "Paris"}') + "\n" + _kimi("ping", 1, "{}")
    calls = tfp.kimi(body, TOOLS)
    assert [c["name"] for c in calls] == ["get_weather", "ping"]
    assert _args(calls[0]) == {"city": "Paris"}


def test_kimi_rejects_a_missing_end_marker_header_or_body():
    with pytest.raises(ValueError, match="end"):
        tfp.kimi(_kimi("f", 0, "{}").replace("<|tool_call_end|>", ""), TOOLS)
    with pytest.raises(ValueError, match="header"):
        tfp.kimi("functions.f:0 {}", TOOLS)
    with pytest.raises(ValueError, match="missing Kimi call"):
        tfp.kimi("   ", TOOLS)


def test_harmony_takes_the_recipient_and_stops_at_the_call_marker():
    body = (
        'commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Oslo"}'
        "<|call|>"
    )
    (call,) = tfp.harmony(body, TOOLS)
    assert call["name"] == "get_weather"
    assert _args(call) == {"city": "Oslo"}


def test_harmony_requires_a_recipient_and_a_message():
    with pytest.raises(ValueError):
        tfp.harmony('to=functions.f {"a": 1}', TOOLS)  # no <|message|>
    with pytest.raises(ValueError):
        tfp.harmony('commentary<|message|>{"a": 1}', TOOLS)  # no to=


def test_mistral_json_list_args_marker_and_bad_input():
    calls = tfp.mistral(
        '[{"name": "get_weather", "arguments": {"city": "Paris"}},'
        ' {"name": "ping", "arguments": {}}]',
        TOOLS,
    )
    assert [c["name"] for c in calls] == ["get_weather", "ping"]
    (call,) = tfp.mistral('get_weather[ARGS]{"city": "Paris"}', TOOLS)
    assert call["name"] == "get_weather"
    assert _args(call) == {"city": "Paris"}
    with pytest.raises(ValueError):
        tfp.mistral("[]", TOOLS)
    with pytest.raises(ValueError):
        tfp.mistral('[{"arguments": {}}]', TOOLS)  # no name
