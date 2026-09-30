"""`conversation` on POST /v1/responses: input prepended, turn appended once, stream and non-stream."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from yunshu_gateway import conversations_store as cs
from yunshu_gateway.routers import conversations as conv_router
from yunshu_gateway.routers import responses as resp_mod


def _response_obj(rid: str, text: str, status: str = "completed") -> dict:
    return {
        "id": rid,
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": status,
        "output": [
            {
                "id": f"msg_{rid}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


class Inner:
    """Stands in for create_response once the wrapper re-enters it."""

    def __init__(self):
        self.calls: list = []
        self.stream = False

    async def __call__(self, req, request):
        self.calls.append(req)
        rid = f"resp_{len(self.calls)}"
        obj = _response_obj(rid, "the answer")
        if not self.stream:
            return JSONResponse(obj)

        async def gen():
            base = {"response": {**obj, "status": "in_progress", "output": []}}
            yield _sse("response.created", {"type": "response.created", **base})
            yield _sse("response.output_text.delta", {"type": "x", "delta": "the"})
            yield _sse(
                "response.completed", {"type": "response.completed", "response": obj}
            )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "convs"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    cs.reset_store()
    inner = Inner()
    monkeypatch.setattr(resp_mod, "create_response", inner)
    app = FastAPI()
    app.include_router(resp_mod.router, prefix="/v1")
    app.include_router(conv_router.router, prefix="/v1")
    client = TestClient(app)
    yield client, inner
    cs.reset_store()


def _texts(client, cid):
    data = client.get(f"/v1/conversations/{cid}/items", params={"order": "asc"}).json()[
        "data"
    ]
    return [
        (i.get("role"), i["content"][0]["text"])
        if i["type"] == "message"
        else i["type"]
        for i in data
    ]


def _new_conv(client, items=None):
    return client.post("/v1/conversations", json={"items": items or []}).json()["id"]


def test_prepended_and_appended_once(env):
    client, inner = env
    cid = _new_conv(client, [{"role": "user", "content": "earlier question"}])
    r = client.post(
        "/v1/responses", json={"model": "m", "input": "next", "conversation": cid}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["conversation"] == {"id": cid}
    assert len(inner.calls) == 1
    seen = inner.calls[0]
    assert seen.conversation is None and seen.context_management is None
    msgs = resp_mod._convert_to_messages(seen)
    assert [m["content"] for m in msgs] == ["earlier question", "next"]
    assert _texts(client, cid) == [
        ("user", "earlier question"),
        ("user", "next"),
        ("assistant", "the answer"),
    ]
    # second turn sees all three items before its own input
    client.post(
        "/v1/responses",
        json={"model": "m", "input": "more", "conversation": {"id": cid}},
    )
    msgs = resp_mod._convert_to_messages(inner.calls[1])
    assert [m["content"] for m in msgs] == [
        "earlier question",
        "next",
        "the answer",
        "more",
    ]
    assert len(_texts(client, cid)) == 5


def test_store_false_still_appends(env):
    client, _ = env
    cid = _new_conv(client)
    client.post(
        "/v1/responses",
        json={"model": "m", "input": "q", "conversation": cid, "store": False},
    )
    assert _texts(client, cid) == [("user", "q"), ("assistant", "the answer")]


def test_stream(env):
    client, inner = env
    inner.stream = True
    cid = _new_conv(client, [{"role": "user", "content": "before"}])
    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "m", "input": "q", "conversation": cid, "stream": True},
    ) as r:
        raw = b"".join(r.iter_bytes()).decode()
    events = [json.loads(ln[6:]) for ln in raw.split("\n") if ln.startswith("data: {")]
    responses = [e["response"] for e in events if "response" in e]
    assert responses and all(x["conversation"] == {"id": cid} for x in responses)
    assert "data: [DONE]" in raw
    assert _texts(client, cid) == [
        ("user", "before"),
        ("user", "q"),
        ("assistant", "the answer"),
    ]


def test_function_call_items_round_trip(env):
    client, inner = env
    cid = _new_conv(
        client,
        [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ],
    )
    client.post(
        "/v1/responses", json={"model": "m", "input": "and?", "conversation": cid}
    )
    msgs = resp_mod._convert_to_messages(inner.calls[0])
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "f"
    assert msgs[1] == {"role": "tool", "tool_call_id": "c1", "content": "42"}
    assert msgs[2]["content"] == "and?"


def test_errors(env):
    client, inner = env
    cid = _new_conv(client)
    r = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": "q",
            "conversation": cid,
            "previous_response_id": "resp_x",
        },
    )
    assert r.status_code == 400
    assert "previous_response_id" in r.json()["error"]["message"]
    r = client.post(
        "/v1/responses",
        json={"model": "m", "input": "q", "conversation": "conv_" + "a" * 24},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "conversation_not_found"
    assert inner.calls == []
    assert _texts(client, cid) == []


def test_openai_sdk(env):
    openai = pytest.importorskip("openai")
    client, _ = env
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="x", http_client=client
    )
    conv = sdk.conversations.create()
    resp = sdk.responses.create(model="m", input="hello", conversation=conv.id)
    assert resp.conversation.id == conv.id
    items = sdk.conversations.items.list(conv.id, order="asc")
    assert [i.type for i in items] == ["message", "message"]
