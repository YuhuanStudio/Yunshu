"""POST /v1/responses/input_tokens and POST /v1/messages/count_tokens must equal the usage of the
real call. Found by the real-server route checks: input_tokens counted raw text only (no chat
template, no tools, no previous_response_id chain, no conversation) -- 14 against a usage of 28 --
and count_tokens injected a tool prompt where generation hands the tools to the template natively
(101 against 272). These tests render with a fake template that, like Qwen's, writes tools itself."""

from __future__ import annotations

import json
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import conversations_store as cs
from yunshu_gateway.routers import anthropic as A  # noqa: N812
from yunshu_gateway.routers import responses as R  # noqa: N812
from yunshu_gateway.routers import tokenize as T  # noqa: N812


class FakeTok:
    bos_token = None

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw
    ):
        out = []
        if tools:
            out.append("<tools>" + json.dumps(tools) + "</tools>")
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            extra = json.dumps(m["tool_calls"]) if m.get("tool_calls") else ""
            out.append(f"<{m['role']}>{c or ''}{extra}</>")
        if add_generation_prompt:
            out.append("<assistant>")
        return "".join(out)

    def encode(self, text, add_special_tokens=True):
        return list(text)  # one token per character


class NativeEngine:
    """Like VLMEngine: its template renders tools itself."""

    _tokenizer = FakeTok()

    def supports_native_tools(self):
        return True


def render(messages, tools=None):
    return len(
        FakeTok().apply_chat_template(messages, tools=tools, add_generation_prompt=True)
    )


WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}
WEATHER_CHAT = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_check_permission", lambda request, perm: None)
    monkeypatch.setattr(T, "_resolve_tokenizer", lambda model: FakeTok())
    monkeypatch.setattr(R, "get_engine", lambda: NativeEngine())
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "convs"))
    cs.reset_store()
    R._response_store.clear()
    app = FastAPI()
    app.include_router(R.router, prefix="/v1")
    yield TestClient(app)
    cs.reset_store()
    R._response_store.clear()


def count(client, **body):
    r = client.post("/v1/responses/input_tokens", json={"model": "m", **body})
    assert r.status_code == 200, r.text
    assert r.json()["object"] == "response.input_tokens"
    return r.json()["input_tokens"]


def test_counts_the_rendered_template_not_the_raw_text(client):
    n = count(client, instructions="Be terse.", input="Hello there")
    want = render(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello there"},
        ]
    )
    assert n == want
    assert n > len("Be terse.\nHello there")  # the old raw-text count


def test_function_tools_count_natively(client):
    base = count(client, input="Hi")
    n = count(client, input="Hi", tools=[WEATHER])
    assert n == render([{"role": "user", "content": "Hi"}], tools=[WEATHER_CHAT])
    assert n > base


def test_previous_response_chain_is_counted(client):
    R._store_response(
        "resp-p",
        {
            "id": "resp-p",
            "object": "response",
            "_input_messages": [{"role": "user", "content": "first"}],
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer one"}],
                }
            ],
        },
    )
    n = count(client, input="second", previous_response_id="resp-p")
    want = render(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
        ]
    )
    assert n == want


def test_conversation_items_are_counted(client):
    conv = cs.get_store().create(
        items=[{"type": "message", "role": "user", "content": "I am Ada."}]
    )
    n = count(client, input="Who am I?", conversation=conv["id"])
    assert n == render(
        [
            {"role": "user", "content": "I am Ada."},
            {"role": "user", "content": "Who am I?"},
        ]
    )
    r = client.post(
        "/v1/responses/input_tokens",
        json={"model": "m", "input": "x", "conversation": "conv_nope"},
    )
    assert r.status_code == 404


def test_bad_bodies_are_400(client):
    assert client.post("/v1/responses/input_tokens", content=b"{no").status_code == 400
    assert client.post("/v1/responses/input_tokens", json=[1]).status_code == 400
    r = client.post("/v1/responses/input_tokens", json={"model": "m", "input": 5})
    assert r.status_code == 400


# ── Anthropic count_tokens ───────────────────────────────────────────────────────────────


@pytest.fixture
def aclient(monkeypatch):
    monkeypatch.setattr(A, "_check_permission", lambda request, perm: None)

    async def resolve(model):
        return NativeEngine(), False

    monkeypatch.setattr(A, "_resolve_engine", resolve)
    app = FastAPI()
    app.include_router(A.router, prefix="/v1")
    return TestClient(app)


def acount(c, **body):
    r = c.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], **body},
    )
    assert r.status_code == 200, r.text
    return r.json()["input_tokens"]


def test_count_tokens_renders_tools_natively(aclient):
    tool = {
        "name": "get_weather",
        "description": "weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    base = acount(aclient)
    n = acount(aclient, tools=[tool])
    assert base == render([{"role": "user", "content": "Hi"}])
    assert n == render(
        [{"role": "user", "content": "Hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": tool["input_schema"],
                },
            }
        ],
    )


def test_count_tokens_tool_choice_none_counts_no_tools(aclient):
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    assert acount(aclient, tools=[tool], tool_choice={"type": "none"}) == acount(
        aclient
    )
    assert types.SimpleNamespace  # keep the import used
