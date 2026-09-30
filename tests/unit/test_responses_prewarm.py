"""`generate: false` (Codex's Responses WebSocket prewarm): prefill the prefix, return an id, no output."""

from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse
from starlette.requests import Request

from yunshu_gateway.routers import responses


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "query_string": b"",
        }
    )


@pytest.fixture
def inner(monkeypatch):
    seen = []

    async def fake(req, request):
        seen.append(req)
        return JSONResponse(
            {
                "id": "resp-inner",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "x"}],
                    }
                ],
                "usage": {
                    "input_tokens": 4200,
                    "output_tokens": 1,
                    "total_tokens": 4201,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    monkeypatch.setattr(responses, "create_response", fake)
    return seen


def _req(**kw):
    base = {
        "model": "m",
        "instructions": "be a coding agent",
        "input": [],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {"type": "object"},
            }
        ],
        "generate": False,
        "store": False,
    }
    return responses.ResponsesRequest(**{**base, **kw})


async def test_prewarm_prefills_and_returns_an_empty_response(inner):
    r = await responses._prewarm_response(_req(), _request())
    body = json.loads(r.body)
    assert body["status"] == "completed" and body["output"] == []
    assert body["usage"]["input_tokens"] == 4200 and body["usage"]["output_tokens"] == 0
    assert body["id"].startswith("resp-") and body["x_yunshu"] == {"prewarm": True}
    (sent,) = inner
    assert (
        sent.max_output_tokens == 1 and sent.stream is False and sent.generate is None
    )
    # an empty input gets a placeholder user turn so chat templates can render
    assert sent.input[0].role == "user"


async def test_prewarm_keeps_a_real_input(inner):
    await responses._prewarm_response(
        _req(input=[{"type": "message", "role": "user", "content": "hello"}]),
        _request(),
    )
    assert inner[0].input[0].content == "hello"


async def test_prewarm_response_is_chainable_and_replays_nothing(inner):
    r = await responses._prewarm_response(_req(), _request())
    rid = json.loads(r.body)["id"]
    stored = responses._get_stored_response(rid)
    assert (
        stored is not None
        and stored["_input_messages"] == []
        and stored["output"] == []
    )


async def test_prewarm_streams_created_and_completed(inner):
    r = await responses._prewarm_response(_req(stream=True), _request())
    raw = b"".join([c async for c in r.body_iterator]).decode()
    names = [ln[7:] for ln in raw.splitlines() if ln.startswith("event: ")]
    assert names == ["response.created", "response.in_progress", "response.completed"]
    last = json.loads(
        [ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")][-1]
    )
    assert last["response"]["status"] == "completed" and last["sequence_number"] == 2


async def test_prewarm_passes_errors_through(monkeypatch):
    async def boom(req, request):
        return JSONResponse({"error": {"message": "bad"}}, status_code=400)

    monkeypatch.setattr(responses, "create_response", boom)
    r = await responses._prewarm_response(_req(), _request())
    assert r.status_code == 400
