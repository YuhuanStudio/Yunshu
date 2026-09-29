"""Small-model malformations of the injected JSON tool call are repaired when unambiguous,
and the injected tool prompt never renders literal doubled braces."""

import json

from yunshu_engine.tool_format import INJECTED_JSON, _unwrap_doubled_braces

GOOD = {"name": "get_weather", "arguments": {"city": "Oslo", "days": 3}}


def test_doubled_braces_and_stray_paren():
    body = '{{"name": "get_weather", "arguments": {"city": "Oslo", "days": 3}})\n'
    assert json.loads(_unwrap_doubled_braces(body)) == GOOD
    (call,) = INJECTED_JSON.parse(body, None)
    assert call is not None


def test_doubled_braces_only():
    body = '{{"name": "get_weather", "arguments": {"city": "Oslo", "days": 3}}}'
    assert json.loads(_unwrap_doubled_braces(body)) == GOOD


def test_doubled_open_brace_missing_close():
    body = '{{"name": "get_weather", "arguments": {"city": "Oslo", "days": 3}}'
    assert json.loads(_unwrap_doubled_braces(body)) == GOOD


def test_valid_json_untouched_and_hopeless_unchanged():
    ok = '{"name": "f", "arguments": {}}'
    assert _unwrap_doubled_braces(ok) == ok
    bad = '{{"name": "f", "arguments": {oops}})'
    assert _unwrap_doubled_braces(bad) == bad


def test_tool_prompt_has_no_doubled_braces():
    from yunshu_gateway.routers.chat import ToolDefinition, _inject_tool_system_prompt

    tool = ToolDefinition(
        type="function",
        function={
            "name": "get_weather",
            "description": "Weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    )
    msgs = _inject_tool_system_prompt([{"role": "user", "content": "hi"}], [tool])
    assert "{{" not in msgs[0]["content"]
