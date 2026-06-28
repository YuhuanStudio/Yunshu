"""a multimodal (image/audio) request to /v1/responses used to return a
Chat-Completions object (object="chat.completion", choices[].message) for non-stream
and chat.completion.chunk SSE for stream — wrong object type, wrong output shape, wrong
event family — and silently dropped `store`. _vlm_to_responses now re-wraps the VLM
text into the proper Responses object / response.* event stream and honors store."""
from __future__ import annotations

import asyncio
import json
import types

from fastapi.responses import JSONResponse

from yunshu_gateway.routers.responses import (
    ResponsesRequest,
    _get_stored_response,
    _vlm_to_responses,
)


def _fake_request():
    st = types.SimpleNamespace(role="", rbac_key=None, _forced_response_id=None)
    return types.SimpleNamespace(state=st)


def _patch_vlm(monkeypatch, content="a photo of a cat", reasoning=None, finish="stop"):
    async def _fake(chat_req, messages, request, json_schema=None):
        # assert the wrapper forced non-stream / n=1
        assert chat_req.stream is False
        assert chat_req.n == 1
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return JSONResponse({
            "object": "chat.completion",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        })
    from yunshu_gateway.routers import chat as chat_mod
    monkeypatch.setattr(chat_mod, "_handle_vlm_chat", _fake)


def test_nonstream_returns_response_object_and_stores(monkeypatch):
    _patch_vlm(monkeypatch)
    req = ResponsesRequest(model="vlm", input="describe", store=True, stream=False)
    msgs = [{"role": "user", "content": "describe"}]
    resp = asyncio.run(_vlm_to_responses(req, msgs, _fake_request(), None, msgs))
    assert isinstance(resp, JSONResponse)
    data = json.loads(bytes(resp.body))
    # correct Responses shape — NOT chat.completion
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["type"] == "output_text"
    assert data["output"][0]["content"][0]["text"] == "a photo of a cat"
    assert data["usage"]["input_tokens"] == 12
    assert data["usage"]["output_tokens"] == 7
    # store honored → retrievable
    stored = _get_stored_response(data["id"])
    assert stored is not None
    assert stored["_input_messages"] == msgs


def test_nonstream_no_store_not_retrievable(monkeypatch):
    _patch_vlm(monkeypatch)
    req = ResponsesRequest(model="vlm", input="x", store=False, stream=False)
    msgs = [{"role": "user", "content": "x"}]
    resp = asyncio.run(_vlm_to_responses(req, msgs, _fake_request(), None, msgs))
    data = json.loads(bytes(resp.body))
    assert _get_stored_response(data["id"]) is None


def test_reasoning_is_separate_output_item(monkeypatch):
    # reasoning is a SEPARATE output item (type=reasoning) preceding the
    # message, NOT a content part of the message (OpenAI Responses spec shape).
    _patch_vlm(monkeypatch, content="answer", reasoning="let me think")
    req = ResponsesRequest(model="vlm", input="x", store=False, stream=False)
    msgs = [{"role": "user", "content": "x"}]
    resp = asyncio.run(_vlm_to_responses(req, msgs, _fake_request(), None, msgs))
    output = json.loads(bytes(resp.body))["output"]
    assert output[0]["type"] == "reasoning"
    assert output[0]["summary"][0]["text"] == "let me think"
    assert output[1]["type"] == "message"
    # the message content is ONLY output_text now (no nested reasoning part)
    assert [p["type"] for p in output[1]["content"]] == ["output_text"]


def test_length_finish_maps_incomplete(monkeypatch):
    _patch_vlm(monkeypatch, finish="length")
    req = ResponsesRequest(model="vlm", input="x", store=False, stream=False)
    msgs = [{"role": "user", "content": "x"}]
    resp = asyncio.run(_vlm_to_responses(req, msgs, _fake_request(), None, msgs))
    assert json.loads(bytes(resp.body))["status"] == "incomplete"


def test_streaming_emits_response_events(monkeypatch):
    _patch_vlm(monkeypatch, content="streamed text")
    req = ResponsesRequest(model="vlm", input="x", store=True, stream=True)
    msgs = [{"role": "user", "content": "x"}]

    async def _collect():
        resp = await _vlm_to_responses(req, msgs, _fake_request(), None, msgs)
        chunks = []
        async for c in resp.body_iterator:
            chunks.append(c if isinstance(c, str) else c.decode())
        return "".join(chunks)

    sse = asyncio.run(_collect())
    # must be the response.* event family, NOT chat.completion.chunk
    assert "chat.completion.chunk" not in sse
    for ev in ("response.created", "response.in_progress", "response.output_item.added",
               "response.content_part.added", "response.output_text.delta",
               "response.output_text.done", "response.completed"):
        assert ev in sse, f"missing {ev}"
    assert "streamed text" in sse
    # sequence_number monotonic from 0
    seqs = [json.loads(line[5:])["sequence_number"]
            for line in sse.splitlines() if line.startswith("data:")]
    assert seqs == sorted(seqs)
    assert seqs[0] == 0
