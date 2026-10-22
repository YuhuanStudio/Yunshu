"""Claude Code's display=updates and the native thinking stream envelope."""

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from yunshu_gateway.reasoning_token import seal_reasoning, unseal_reasoning
from yunshu_gateway.routers import anthropic as a


def message():
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [
            {"type": "thinking", "thinking": "私密 reasoning", "signature": ""},
            {"type": "text", "text": "42"},
        ],
        "usage": {"input_tokens": 7, "output_tokens": 13},
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }


def frames(signed=False):
    return [
        {"type": "message_start", "message": {**message(), "content": []}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "私密 "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "reasoning"},
        },
        *(
            [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "existing"},
                }
            ]
            if signed
            else []
        ),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "42"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": message()["usage"],
        },
        {"type": "message_stop"},
    ]


def client(monkeypatch, signed=False):
    async def inner(req, request, temp_files):
        if not req.stream:
            return JSONResponse(message(), headers={"x-probe": "preserved"})

        async def body():
            yield b": ping\n\n"
            # Deliberately split UTF-8 and SSE framing across bytes.
            for event in frames(signed):
                raw = f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                for i in range(0, len(raw), 3):
                    yield raw[i : i + 3]

        return StreamingResponse(body(), media_type="text/event-stream")

    monkeypatch.setattr(a, "_create_message", inner)
    app = FastAPI()
    app.include_router(a.router, prefix="/v1")
    return TestClient(app)


def request(display=None, stream=False):
    return {
        "model": "m",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "thinking": {
            "type": "adaptive",
            **({"display": display} if display is not None else {}),
        },
    }


@pytest.mark.parametrize("display", ["omitted", "updates"])
def test_nonstream_display_hides_reasoning_preserving_usage(monkeypatch, display):
    r = client(monkeypatch).post("/v1/messages", json=request(display))
    assert r.status_code == 200
    assert r.json()["content"] == [
        {
            **message()["content"][0],
            "thinking": "",
            "signature": seal_reasoning("私密 reasoning"),
        },
        message()["content"][1],
    ]
    assert r.json()["usage"] == message()["usage"]
    assert r.headers["x-probe"] == "preserved"
    assert int(r.headers["content-length"]) == len(r.content)


@pytest.mark.parametrize("display", [None, "summarized", "omitted", "updates"])
@pytest.mark.parametrize("signed", [False, True])
def test_stream_display_and_signature_before_stop(monkeypatch, display, signed):
    r = client(monkeypatch, signed).post("/v1/messages", json=request(display, True))
    assert r.status_code == 200
    assert ": ping\n\n" in r.text
    ev = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    thinking = [
        e["delta"]["thinking"]
        for e in ev
        if e.get("delta", {}).get("type") == "thinking_delta"
    ]
    assert thinking == (
        [""] if display in ("omitted", "updates") else ["私密 ", "reasoning"]
    )
    signatures = [e for e in ev if e.get("delta", {}).get("type") == "signature_delta"]
    assert len(signatures) == 1
    signature = signatures[0]["delta"]["signature"]
    if signed:
        assert signature == "existing"
    elif display in ("omitted", "updates"):
        assert unseal_reasoning(signature) == "私密 reasoning"
    else:
        assert signature == ""
    stop = next(i for i, e in enumerate(ev) if e["type"] == "content_block_stop")
    assert ev[stop - 1] == signatures[0]
    assert ev[-2:] == frames()[-2:]
    assert [e for e in ev if e.get("index") == 1] == [
        e for e in frames() if e.get("index") == 1
    ]


def test_unknown_display_rejected(monkeypatch):
    r = client(monkeypatch).post("/v1/messages", json=request("typo"))
    assert r.status_code in (400, 422)


def test_hidden_reasoning_roundtrips_through_message_conversion(monkeypatch):
    c = client(monkeypatch)
    response = c.post("/v1/messages", json=request("omitted")).json()
    converted, _ = a._convert_anthropic_messages(
        [a.AnthropicMessage(role="assistant", content=response["content"])]
    )
    assert converted == [
        {"role": "assistant", "content": "42", "reasoning_content": "私密 reasoning"}
    ]


def test_foreign_or_malformed_signature_does_not_become_prompt_text():
    for signature in ("hosted-signature", "yunshu1:%%%%"):
        keep, thinking = a._split_thinking(
            [
                {"type": "thinking", "thinking": "", "signature": signature},
                {"type": "text", "text": "42"},
            ]
        )
        assert keep == [{"type": "text", "text": "42"}]
        assert thinking == ""
