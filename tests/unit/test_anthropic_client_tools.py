"""bash_* / text_editor_* / memory_* are declared by type only; the gateway supplies their schema."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.anthropic_client_tools import fill_client_tool_schemas, schema_for
from yunshu_gateway.routers import anthropic

from .server_tools_helpers import ScriptedInner


@pytest.mark.parametrize(
    "t,expect",
    [
        ("bash_20250124", "command"),
        ("bash_20241022", "command"),
        ("text_editor_20250728", "path"),
        ("text_editor_20241022", "path"),
        ("memory_20250818", "command"),
    ],
)
def test_known_types_have_schemas(t, expect):
    spec = schema_for(t)
    assert spec and expect in spec["input_schema"]["properties"]


@pytest.mark.parametrize("t", [None, "custom", "web_search_20250305", "bash"])
def test_unknown_types_are_left_alone(t):
    assert schema_for(t) is None


def _req(tools):
    return anthropic.AnthropicMessagesRequest(
        model="m",
        max_tokens=10,
        messages=[{"role": "user", "content": "x"}],
        tools=tools,
    )


def test_fill_only_when_the_client_sent_no_schema():
    req = _req(
        [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
            {
                "type": "bash_20250124",
                "name": "mine",
                "input_schema": {"type": "object", "properties": {}},
            },
            {"name": "plain", "input_schema": {"type": "object"}},
        ]
    )
    assert fill_client_tool_schemas(req.tools) is True
    bash, editor, mine, plain = req.tools
    assert "command" in bash.input_schema["properties"] and bash.description
    assert editor.input_schema["required"] == ["command", "path"]
    assert mine.input_schema == {
        "type": "object",
        "properties": {},
    }  # the client's own schema wins
    assert plain.description is None
    assert fill_client_tool_schemas(req.tools) is False  # idempotent


def test_the_model_sees_the_schema_through_the_messages_route(monkeypatch):
    inner = ScriptedInner([([{"type": "text", "text": "ok"}], "end_turn")])
    monkeypatch.setattr(anthropic, "create_message", inner)
    app = FastAPI()
    app.include_router(anthropic.router, prefix="/v1")
    # a server tool in the request sends the call through the loop, which hands the inner
    # handler the converted tools (the plain path would need an engine)
    r = TestClient(app).post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    names = {t.name: t for t in inner.requests[0].tools}
    assert "command" in names["bash"].input_schema["properties"]
