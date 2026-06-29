"""Responses reasoning is now a SEPARATE output item (OpenAI spec: type=
reasoning, rs_ id, output_index 0) PRECEDING the message (output_index 1 when reasoning
present) — across non-stream, the VLM path, AND the main streaming path — instead of a
reasoning content part nested in the message. Stream and non-stream stay consistent
(the invariant): the message's output_item.added is emitted lazily so a leading
reasoning item can take output_index 0."""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import types

from yunshu_gateway.routers import (
    responses as R,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.responses import ResponsesRequest, _vlm_to_responses


def _fake_request():
    st = types.SimpleNamespace(role="", rbac_key=None, _forced_response_id=None)
    return types.SimpleNamespace(state=st)


def _patch_vlm(monkeypatch, content, reasoning):
    from fastapi.responses import JSONResponse
    async def _fake(chat_req, messages, request, json_schema=None):
        return JSONResponse({
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                                                 "reasoning_content": reasoning},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        })
    from yunshu_gateway.routers import chat as chat_mod
    monkeypatch.setattr(chat_mod, "_handle_vlm_chat", _fake)


def _collect_sse(monkeypatch, content="the answer", reasoning="thinking..."):
    _patch_vlm(monkeypatch, content, reasoning)
    req = ResponsesRequest(model="vlm", input="x", store=False, stream=True)
    msgs = [{"role": "user", "content": "x"}]

    async def _run():
        resp = await _vlm_to_responses(req, msgs, _fake_request(), None, msgs)
        out = []
        async for c in resp.body_iterator:
            out.append(c if isinstance(c, str) else c.decode())
        return "".join(out)
    return asyncio.run(_run())


def _events(sse):
    evs = []
    for line in sse.splitlines():
        if line.startswith("data: ") and line[6:].strip() not in ("[DONE]", ""):
            with contextlib.suppress(Exception):
                evs.append(json.loads(line[6:]))
    return evs


def test_streaming_reasoning_item_precedes_message(monkeypatch):
    evs = _events(_collect_sse(monkeypatch))
    added = [e for e in evs if e.get("type") == "response.output_item.added"]
    # first added item is the reasoning item at output_index 0; second is the message at 1
    assert added[0]["item"]["type"] == "reasoning"
    assert added[0]["output_index"] == 0
    assert added[1]["item"]["type"] == "message"
    assert added[1]["output_index"] == 1
    # reasoning summary events carry the rs_ id, not the message id
    rs_id = added[0]["item"]["id"]
    assert rs_id.startswith("rs-")
    summary = [e for e in evs if e.get("type", "").startswith("response.reasoning_summary")]
    assert summary and all(e["item_id"] == rs_id for e in summary)


def test_streaming_completed_output_is_reasoning_then_message(monkeypatch):
    evs = _events(_collect_sse(monkeypatch))
    completed = [e for e in evs if e.get("type") == "response.completed"][-1]
    out = completed["response"]["output"]
    assert [o["type"] for o in out] == ["reasoning", "message"]


def test_no_reasoning_message_at_index_0(monkeypatch):
    evs = _events(_collect_sse(monkeypatch, reasoning=None))
    added = [e for e in evs if e.get("type") == "response.output_item.added"]
    assert added[0]["item"]["type"] == "message"
    assert added[0]["output_index"] == 0


def test_main_streaming_path_uses_lazy_reasoning_item():
    # the main text streaming generator must use the lazy separate-item helpers
    src = inspect.getsource(R)
    assert "_ensure_reasoning_item" in src
    assert "_ensure_msg_item" in src
    assert "_emit_token" in src
    # message close + final_output keyed on _msg_idx / separate reasoning item
    assert "output_index=_msg_idx" in src
