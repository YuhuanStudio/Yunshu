"""Claude Code wire details: per-turn token counter, integer arguments, undeclared markup."""

import json

from tests.unit.test_anthropic_mid_system import (
    _captured_messages,
)  # noqa: F401
from yunshu_engine.tool_arguments import coerce_tool_arguments, tool_schemas

COUNTER = "<total_tokens>14982239 tokens left</total_tokens>"


def _body(counter):
    return {
        "model": "claude-x",
        "max_tokens": 10,
        "system": "TOP_SYSTEM",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "more"},
            {"role": "system", "content": counter},
        ],
    }


def test_token_counter_system_message_never_reaches_the_prompt(_engine, monkeypatch):
    engine, set_engine = _engine
    a = _captured_messages(
        engine, set_engine, monkeypatch, "Qwen3.8-27B", _body(COUNTER)
    )
    b = _captured_messages(
        engine,
        set_engine,
        monkeypatch,
        "Qwen3.8-27B",
        _body(COUNTER.replace("14982239", "14900000")),
    )
    assert "tokens left" not in json.dumps(a)
    assert a == b  # prefix identical across turns


def test_real_system_note_is_still_kept(_engine, monkeypatch):
    engine, set_engine = _engine
    msgs = _captured_messages(
        engine, set_engine, monkeypatch, "Qwen3.8-27B", _body("be terse")
    )
    assert "be terse" in msgs[0]["content"]


def _tools(kind):
    return [
        {
            "name": "f",
            "input_schema": {"type": "object", "properties": {"n": {"type": kind}}},
        }
    ]


def test_integer_text_under_number_schema_stays_integer():
    schemas = tool_schemas(_tools("number"))
    out = json.loads(coerce_tool_arguments("f", '{"n": "10000"}', schemas))
    assert out == {"n": 10000} and isinstance(out["n"], int)
    out = json.loads(coerce_tool_arguments("f", '{"n": 10000}', schemas))
    assert isinstance(out["n"], int)
    assert json.loads(coerce_tool_arguments("f", '{"n": "1.5"}', schemas)) == {"n": 1.5}


def test_typed_arguments_are_not_decoded_twice():
    """Typed JSON values (already dict/list/str) must survive exactly."""
    schemas = {
        "f": {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "a": {"type": "array"},
                "o": {"type": "object"},
                "u": {},
            },
        }
    }
    args = {
        "s": '{"x": 1}',  # a string that merely looks like JSON
        "a": [1, "2", '{"k": 1}'],
        "o": {"inner": "[1, 2]", "n": "3"},
        "u": '"quoted"',  # unknown type: a JSON string literal stays as sent
    }
    out = json.loads(coerce_tool_arguments("f", json.dumps(args), schemas))
    assert out == args


UNDECLARED = (
    "<tool_call>\n<function=read>\n<parameter=path>file.txt</parameter>\n"
    "</function>\n</tool_call>"
)


def test_undeclared_tool_markup_never_reaches_content():
    from tests.unit.test_tool_format import _formats
    from yunshu_engine.tool_format import parse_tool_output

    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    calls, content = parse_tool_output(UNDECLARED, _formats("qwen3_coder"), tools)
    assert "<tool_call>" not in content and "<function=" not in content
    assert calls == [] and content == ""
    mixed = UNDECLARED + "\n" + UNDECLARED.replace("read", "get_weather")
    calls, content = parse_tool_output(mixed, _formats("qwen3_coder"), tools)
    assert [c["name"] for c in calls] == ["get_weather"] and content == ""
