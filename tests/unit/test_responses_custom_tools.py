"""Freeform Responses tools use native tools internally and custom items outward."""

import json

import pytest
from openai import OpenAI

from yunshu_gateway.routers import responses
from yunshu_gateway.server_tools.responses_loop import function_tools

from .wire_harness import Script, install

INPUT = "*** Begin Patch\n*** Add File: hello.txt\n+你好\n*** End Patch\n"
TOOLS = [
    {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch",
        "format": {"type": "text"},
    }
]


@pytest.mark.parametrize("stream", [False, True])
def test_custom_tool_typed_sdk_and_history(monkeypatch, stream):
    text = (
        "<tool_call>"
        + json.dumps({"name": "apply_patch", "arguments": {"input": INPUT}})
        + "</tool_call>"
    )
    client, engine = install(monkeypatch, Script(pieces=[text]))
    sdk = OpenAI(api_key="x", base_url="http://testserver/v1", http_client=client)
    result = sdk.responses.create(
        model="scripted", input="Apply this patch", tools=TOOLS, stream=stream
    )
    if stream:
        events = list(result)
        delta = [
            e.delta for e in events if e.type == "response.custom_tool_call_input.delta"
        ]
        done = [
            e.input for e in events if e.type == "response.custom_tool_call_input.done"
        ]
        assert "".join(delta) == INPUT and done == [INPUT]
        assert not any(
            e.type == "response.function_call_arguments.done" for e in events
        )
        seq = [e.sequence_number for e in events]
        assert seq == sorted(set(seq))
        result = events[-1].response
    call = next(i for i in result.output if i.type == "custom_tool_call")
    assert call.name == "apply_patch" and call.input == INPUT
    assert result.tools[0].type == "custom"
    assert engine.calls and "apply_patch" in str(engine.calls)
    # A chained response must replay the custom call as a native tool call.
    follow = sdk.responses.create(
        model="scripted",
        previous_response_id=result.id,
        input=[
            {"type": "custom_tool_call_output", "call_id": call.call_id, "output": "ok"}
        ],
        tools=TOOLS,
    )
    assert follow.output
    assert any(m.get("tool_calls") for m in engine.calls[-1]["messages"])


def test_namespace_custom_is_not_dropped():
    req = responses.ResponsesRequest(
        model="m",
        input="x",
        tools=[{"type": "namespace", "name": "editing", "tools": TOOLS}],
    )
    fn = function_tools(req.tools)
    assert fn[0].name == "apply_patch"
    assert fn[0].parameters["required"] == ["input"]


def test_custom_grammar_is_a_clear_400(monkeypatch):
    client, engine = install(monkeypatch, Script(pieces=["should not generate"]))
    r = client.post(
        "/v1/responses",
        json={
            "model": "scripted",
            "input": "x",
            "tools": [
                {
                    "type": "custom",
                    "name": "patch",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": 'start: "x"',
                    },
                }
            ],
        },
    )
    assert r.status_code == 400 and "cannot be guaranteed" in r.text
    assert not engine.calls
