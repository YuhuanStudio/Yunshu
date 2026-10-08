"""Ollama-compatible /api layer: translation to and from the OpenAI routes."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from yunshu_gateway.routers import ollama


def _upstream() -> tuple[FastAPI, list[dict]]:
    app = FastAPI()
    seen: list[dict] = []

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "demo-4bit", "created": 1700000000}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        seen.append(body)
        if body["model"] == "boom":
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "bad request here", "type": "x"}},
            )
        usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        if body.get("tools"):
            call = {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
            if body["stream"]:

                async def sse_tools():
                    d = {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                **{"function": call["function"]},
                            }
                        ]
                    }
                    yield "data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n"
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "choices": [
                                    {"delta": {}, "finish_reason": "tool_calls"}
                                ],
                                "usage": usage,
                            }
                        )
                        + "\n\n"
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(sse_tools(), media_type="text/event-stream")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [call],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": usage,
            }
        if body["stream"]:

            async def sse():
                for piece in ("Hel", "lo"):
                    yield (
                        "data: "
                        + json.dumps({"choices": [{"delta": {"content": piece}}]})
                        + "\n\n"
                    )
                yield (
                    "data: "
                    + json.dumps({"choices": [{"delta": {"reasoning_content": "hmm"}}]})
                    + "\n\n"
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [{"delta": {}, "finish_reason": "length"}],
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "reasoning_content": "hmm",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    @app.post("/v1/embeddings")
    async def emb(request: Request):
        body = await request.json()
        n = len(body["input"]) if isinstance(body["input"], list) else 1
        return {
            "data": [{"embedding": [0.1, 0.2], "index": i} for i in range(n)],
            "usage": {"prompt_tokens": 4},
        }

    return app, seen


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: None)
    up, seen = _upstream()

    def fake_client(request):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=up), base_url="http://up"
        )

    monkeypatch.setattr(ollama, "_client", fake_client)
    app = FastAPI()
    app.include_router(ollama.router)
    c = TestClient(app)
    c.seen = seen  # type: ignore[attr-defined]
    return c


def test_version_tags_ps_show(client):
    assert "version" in client.get("/api/version").json()
    tags = client.get("/api/tags").json()["models"]
    assert (
        tags[0]["name"] == "demo-4bit"
        and tags[0]["details"]["quantization_level"] == "4BIT"
    )
    assert client.get("/api/ps").json()["models"][0]["model"] == "demo-4bit"
    assert (
        client.post("/api/show", json={"model": "demo-4bit:latest"}).status_code == 200
    )
    r = client.post("/api/show", json={"model": "nope"})
    assert r.status_code == 404 and "error" in r.json()


def test_chat_non_stream_maps_options_and_usage(client):
    r = client.post(
        "/api/chat",
        json={
            "model": "demo-4bit",
            "stream": False,
            "messages": [{"role": "user", "content": "hi", "images": ["QUJD"]}],
            "options": {
                "temperature": 0.2,
                "num_predict": 9,
                "stop": "x",
                "repeat_penalty": 1.1,
            },
            "format": "json",
            "think": False,
        },
    )
    j = r.json()
    assert j["done"] and j["done_reason"] == "stop"
    assert j["message"]["content"] == "Hello" and j["message"]["thinking"] == "hmm"
    assert j["prompt_eval_count"] == 7 and j["eval_count"] == 3
    sent = client.seen[-1]
    assert sent["max_tokens"] == 9 and sent["stop"] == ["x"]
    assert sent["repetition_penalty"] == 1.1 and sent["temperature"] == 0.2
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["enable_thinking"] is False
    part = sent["messages"][0]["content"]
    assert part[1]["image_url"]["url"].endswith("QUJD")


def test_chat_stream_is_ndjson(client):
    r = client.post(
        "/api/chat",
        json={"model": "demo-4bit", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.headers["content-type"].startswith("application/x-ndjson")
    chunks = [json.loads(line) for line in r.text.splitlines()]
    assert [c["done"] for c in chunks] == [False, False, False, True]
    assert "".join(c["message"]["content"] for c in chunks) == "Hello"
    assert chunks[2]["message"]["thinking"] == "hmm"
    assert chunks[-1]["done_reason"] == "length" and chunks[-1]["eval_count"] == 3


def test_tool_calls_have_object_arguments(client):
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    for stream in (False, True):
        r = client.post(
            "/api/chat",
            json={
                "model": "demo-4bit",
                "stream": stream,
                "tools": tools,
                "messages": [{"role": "user", "content": "w?"}],
            },
        )
        last = [json.loads(line) for line in r.text.splitlines()][-1]
        tc = last["message"]["tool_calls"][0]["function"]
        assert tc == {"name": "get_weather", "arguments": {"city": "Paris"}}


def test_tool_result_messages_are_translated(client):
    client.post(
        "/api/chat",
        json={
            "model": "demo-4bit",
            "stream": False,
            "messages": [
                {"role": "user", "content": "w?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
                },
                {"role": "tool", "content": "sunny", "tool_name": "f"},
            ],
        },
    )
    msgs = client.seen[-1]["messages"]
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert msgs[2]["role"] == "tool" and msgs[2]["tool_call_id"]


def test_generate_and_empty_prompt(client):
    r = client.post(
        "/api/generate",
        json={"model": "demo-4bit", "prompt": "p", "system": "s", "stream": False},
    )
    assert r.json()["response"] == "Hello" and r.json()["done"]
    assert [m["role"] for m in client.seen[-1]["messages"]] == ["system", "user"]
    r = client.post("/api/generate", json={"model": "demo-4bit", "prompt": ""})
    assert r.json()["done_reason"] == "load"


def test_embed_and_legacy_embeddings(client):
    j = client.post("/api/embed", json={"model": "m", "input": ["a", "b"]}).json()
    assert len(j["embeddings"]) == 2 and j["prompt_eval_count"] == 4
    assert client.post(
        "/api/embeddings", json={"model": "m", "prompt": "a"}
    ).json() == {"embedding": [0.1, 0.2]}


def test_errors_use_ollama_shape(client):
    r = client.post(
        "/api/chat",
        json={
            "model": "boom",
            "stream": False,
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 400 and r.json() == {"error": "bad request here"}
    r = client.post(
        "/api/chat",
        json={"model": "boom", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 400 and r.json()["error"] == "bad request here"
    assert client.post("/api/chat", content=b"{").status_code == 400
    assert client.post("/api/chat", json={"messages": []}).status_code == 400
    assert client.post("/api/pull", json={"name": "x"}).status_code == 400
    assert client.delete("/api/delete").status_code == 400


def test_ps_filters_unloaded_models_without_private_list_fields(client, monkeypatch):
    from types import SimpleNamespace

    manager = SimpleNamespace(
        list_entries=lambda: [SimpleNamespace(model_id="demo-4bit", is_loaded=False)]
    )
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: manager)
    assert client.get("/api/ps").json() == {"models": []}


def test_latest_tag_routes_to_native_id_but_echoes_requested_model(client, monkeypatch):
    from types import SimpleNamespace

    entry = SimpleNamespace(model_id="demo-4bit")
    manager = SimpleNamespace(
        get_entry=lambda name: entry if name == "demo-4bit" else None
    )
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: manager)
    r = client.post(
        "/api/chat",
        json={
            "model": "demo-4bit:latest",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200 and r.json()["model"] == "demo-4bit:latest"
    assert client.seen[-1]["model"] == "demo-4bit"
