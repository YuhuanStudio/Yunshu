"""n>1 with stream=true on chat completions and completions: choices stream with their `index`."""

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    engine = BatchedEngine()
    engine._model = object()
    engine._loaded = True
    engine._running = True
    engine.model_name = "m"
    seeds: list = []

    async def _stream(*args, **kwargs):
        seeds.append(kwargs.get("seed"))
        tag = f"s{len(seeds)}"
        for i, piece in enumerate((f"{tag}a", f"{tag}b")):
            last = i == 1
            yield GenerationOutput(
                text=piece,
                new_text=piece,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
            )

    engine.stream_chat = _stream
    engine.stream_generate = _stream
    set_engine(engine)
    yield TestClient(create_app(), raise_server_exceptions=False), seeds
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _events(text):
    out = []
    for block in text.split("\n\n"):
        if block.startswith("data:") and block[5:].strip() != "[DONE]":
            out.append(json.loads(block[5:]))
    return out


def _by_index(events, key):
    got: dict[int, str] = {}
    for e in events:
        for c in e["choices"]:
            piece = c["delta"].get("content") if key == "chat" else c.get("text")
            got[c["index"]] = got.get(c["index"], "") + (piece or "")
    return got


def test_chat_stream_n3(server):
    client, seeds = server
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "n": 3,
            "stream": True,
            "seed": 7,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    ev = _events(r.text)
    assert len({e["id"] for e in ev}) == 1
    content = [e for e in ev if e["choices"]]
    assert sorted(_by_index(content, "chat")) == [0, 1, 2]
    assert all(v for v in _by_index(content, "chat").values())
    assert sorted(set(seeds)) == [7, 8, 9]  # the non-stream path's per-choice seeds
    finishes = [c["index"] for e in content for c in e["choices"] if c["finish_reason"]]
    assert sorted(finishes) == [0, 1, 2]
    usage = [e for e in ev if e.get("usage")]
    assert len(usage) == 1 and usage[0]["usage"]["completion_tokens"] == 6
    assert r.text.rstrip().endswith("[DONE]")


def test_completions_stream_n2(server):
    client, _ = server
    r = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "n": 2, "stream": True, "max_tokens": 4},
    )
    assert r.status_code == 200, r.text
    ev = [e for e in _events(r.text) if e.get("choices")]
    assert sorted(_by_index(ev, "completions")) == [0, 1]


def test_vlm_stream_n3_interleaves_by_index(monkeypatch):
    import asyncio

    from yunshu_gateway.routers import chat
    from yunshu_gateway.routers.chat import ChatCompletionRequest, StreamOptions
    from yunshu_gateway.streaming import format_openai_chunk, format_openai_usage_chunk

    seeds = []

    async def one(vlm, messages, req, cid, request, **kw):
        seeds.append((cid, req.seed, req.n))
        for k in range(2):
            await asyncio.sleep(0.01)
            yield format_openai_chunk(
                completion_id=cid, model="m", delta_content=f"{cid[-1]}{k}"
            ).encode()
        yield format_openai_usage_chunk(cid, "m", 5, 2).encode()
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(chat, "_stream_vlm_response", one)
    req = ChatCompletionRequest(
        model="m",
        n=3,
        stream=True,
        seed=10,
        stream_options=StreamOptions(include_usage=True),
        messages=[{"role": "user", "content": "hi"}],
    )

    async def run():
        return [
            f
            async for f in chat._stream_vlm_response_multi(
                None, [], req, "chatcmpl-x", None
            )
        ]

    frames = asyncio.run(run())
    ev = _events("".join(f.decode() for f in frames))
    content = [e for e in ev if e["choices"]]
    assert all(e["id"] == "chatcmpl-x" for e in ev)
    assert _by_index(content, "chat") == {0: "0001", 1: "1011", 2: "2021"}
    assert sorted(i for _, s, n in seeds for i in [s]) == [10, 11, 12]
    assert all(n == 1 for *_, n in seeds)
    usage = [e for e in ev if e.get("usage")]
    assert len(usage) == 1
    assert usage[0]["usage"]["prompt_tokens"] == 5
    assert usage[0]["usage"]["completion_tokens"] == 6
    assert frames[-1] == b"data: [DONE]\n\n"
