"""Stored chat completions (store=true) through the official openai SDK against a TestClient."""

from __future__ import annotations

import json

import openai
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from yunshu_gateway import chat_store
from yunshu_gateway.routers import chat as chat_mod
from yunshu_gateway.routers import chat_stored


def _completion(cid: str, text: str, model="m1", created=1000) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": None},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CHAT_COMPLETIONS_DIR", str(tmp_path / "cc"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    chat_store.reset_store()
    counter = {"n": 0}

    async def fake(req, request):
        counter["n"] += 1
        cid = f"chatcmpl-{counter['n']:024d}"
        if req.stream:

            def chunk(delta, finish=None, usage=None):
                d = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": 1000 + counter["n"],
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if usage:
                    d["usage"] = usage
                    d["choices"] = []
                return f"data: {json.dumps(d)}\n\n"

            async def gen():
                yield chunk({"role": "assistant", "content": ""})
                yield chunk({"content": "Hel"})
                yield chunk({"content": "lo"})
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            _completion(cid, f"answer {counter['n']}", req.model, 1000 + counter["n"])
        )

    monkeypatch.setattr(chat_mod, "_create_chat_completion", fake)
    app = FastAPI()
    app.include_router(chat_mod.router, prefix="/v1")
    app.include_router(chat_stored.router, prefix="/v1")
    tc = TestClient(app)
    yield openai.OpenAI(base_url="http://testserver/v1", api_key="x", http_client=tc)
    chat_store.reset_store()


def test_store_retrieve_update_delete(sdk):
    msgs = [{"role": "user", "content": "hi"}]
    r = sdk.chat.completions.create(
        model="m1", messages=msgs, store=True, metadata={"k": "v"}
    )
    got = sdk.chat.completions.retrieve(r.id)
    assert (
        got.id == r.id
        and got.choices[0].message.content == r.choices[0].message.content
    )
    assert got.metadata == {"k": "v"}
    upd = sdk.chat.completions.update(r.id, metadata={"k": "w", "a": "b"})
    assert upd.metadata == {"k": "w", "a": "b"}
    dele = sdk.chat.completions.delete(r.id)
    assert dele.deleted and dele.object == "chat.completion.deleted"
    with pytest.raises(openai.NotFoundError):
        sdk.chat.completions.retrieve(r.id)


def test_not_stored_without_store_flag(sdk):
    r = sdk.chat.completions.create(
        model="m1", messages=[{"role": "user", "content": "x"}]
    )
    with pytest.raises(openai.NotFoundError):
        sdk.chat.completions.retrieve(r.id)
    assert list(sdk.chat.completions.list()) == []


def test_list_filters_order_pagination(sdk):
    ids = []
    for i in range(5):
        r = sdk.chat.completions.create(
            model="m1" if i % 2 == 0 else "m2",
            messages=[{"role": "user", "content": f"q{i}"}],
            store=True,
            metadata={"parity": str(i % 2)},
        )
        ids.append(r.id)
    assert [c.id for c in sdk.chat.completions.list()] == ids
    assert [c.id for c in sdk.chat.completions.list(order="desc")] == ids[::-1]
    assert [c.id for c in sdk.chat.completions.list(model="m2")] == [ids[1], ids[3]]
    assert [c.id for c in sdk.chat.completions.list(metadata={"parity": "0"})] == [
        ids[0],
        ids[2],
        ids[4],
    ]
    page = sdk.chat.completions.list(limit=2)
    assert [c.id for c in page.data] == ids[:2] and page.has_next_page()
    nxt = sdk.chat.completions.list(limit=2, after=ids[1])
    assert [c.id for c in nxt.data] == ids[2:4]


def test_messages_keep_the_input_with_parts(sdk):
    parts = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    r = sdk.chat.completions.create(
        model="m1",
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": parts},
        ],
        store=True,
    )
    msgs = list(sdk.chat.completions.messages.list(r.id))
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == "be brief" and msgs[0].id.startswith("msg_")
    assert msgs[1].content_parts is not None and len(msgs[1].content_parts) == 2
    assert [m.role for m in sdk.chat.completions.messages.list(r.id, order="desc")] == [
        "user",
        "system",
    ]


def test_stream_is_folded_and_stored(sdk):
    stream = sdk.chat.completions.create(
        model="m1",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        store=True,
    )
    cid = None
    for ch in stream:
        cid = ch.id
    got = sdk.chat.completions.retrieve(cid)
    assert got.choices[0].message.content == "Hello"
    assert got.choices[0].finish_reason == "stop"


def test_bad_inputs_are_400(sdk):
    r = sdk.chat.completions.create(
        model="m1", messages=[{"role": "user", "content": "x"}], store=True
    )
    with pytest.raises(openai.BadRequestError):
        sdk.chat.completions.update(r.id, metadata={"k": 1})
    with pytest.raises(openai.BadRequestError):
        sdk.chat.completions.create(
            model="m1",
            messages=[{"role": "user", "content": "x"}],
            store=True,
            metadata={"k": "v" * 513},
        )
    with pytest.raises(openai.BadRequestError):
        sdk.chat.completions.list(limit=1000)
    with pytest.raises(openai.NotFoundError):
        sdk.chat.completions.delete("chatcmpl-nope")


def test_eviction_keeps_the_newest(sdk, monkeypatch):
    monkeypatch.setenv("YUNSHU_CHAT_COMPLETIONS_MAX", "2")
    ids = [
        sdk.chat.completions.create(
            model="m1", messages=[{"role": "user", "content": "x"}], store=True
        ).id
        for _ in range(3)
    ]
    assert [c.id for c in sdk.chat.completions.list()] == ids[1:]
