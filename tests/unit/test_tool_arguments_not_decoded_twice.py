"""Tool-call arguments that are already typed (dict) must not be JSON-decoded again."""

import json

from yunshu_gateway.routers.chat import ToolCallFunction
from yunshu_gateway.routers.ollama import _tool_calls_out


def test_chat_request_accepts_dict_arguments():
    fn = ToolCallFunction(name="add", arguments={"a": 2, "b": 3})
    assert json.loads(fn.arguments) == {"a": 2, "b": 3}


def test_chat_request_string_arguments_unchanged():
    fn = ToolCallFunction(name="add", arguments='{"a": 2}')
    assert fn.arguments == '{"a": 2}'


def test_ollama_output_keeps_dict_arguments():
    out = _tool_calls_out([{"function": {"name": "add", "arguments": {"a": 2}}}])
    assert out[0]["function"]["arguments"] == {"a": 2}


def test_ollama_output_decodes_string_once():
    inner = json.dumps({"s": '{"x": 1}'})
    out = _tool_calls_out([{"function": {"name": "f", "arguments": inner}}])
    assert out[0]["function"]["arguments"] == {"s": '{"x": 1}'}
