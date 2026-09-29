"""x_yunshu stats on the Anthropic Messages and OpenAI Responses routes: inside the
``usage`` object (JSON body and terminal SSE event), parsed by the official SDKs."""

import json

import httpx
import openai
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from yunshu_gateway import x_yunshu
from yunshu_gateway.x_yunshu import YunshuExtensionsMiddleware, normalize_usage

anthropic = pytest.importorskip("anthropic")


@pytest.fixture(autouse=True)
def _clean():
    x_yunshu.registry.clear()
    yield
    x_yunshu.registry.clear()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(body: dict):
        msg = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 5},
        }
        if not body.get("stream"):
            return JSONResponse(msg)

        async def events():
            start = {
                **msg,
                "content": [],
                "usage": {"input_tokens": 12, "output_tokens": 0},
            }
            yield _sse("message_start", {"type": "message_start", "message": start})
            block = {"type": "text", "text": ""}
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": block},
            )
            delta = {"type": "text_delta", "text": "hi"}
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": delta},
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/responses")
    async def responses(body: dict):
        resp = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 17,
            },
        }
        if not body.get("stream"):
            return JSONResponse(resp)

        async def events():
            yield _sse(
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**resp, "status": "in_progress", "usage": None},
                },
            )
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "i",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hi",
                    "logprobs": [],
                },
            )
            yield _sse(
                "response.completed",
                {"type": "response.completed", "sequence_number": 2, "response": resp},
            )

        return StreamingResponse(events(), media_type="text/event-stream")

    app.add_middleware(YunshuExtensionsMiddleware)
    return app


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://t"
    )


def _anthropic_client(http: httpx.AsyncClient):
    """Newer anthropic SDKs are built on httpx2 and reject an httpx client."""
    try:
        import httpx2
    except ImportError:
        httpx2 = None
    if httpx2 is not None and hasattr(anthropic._base_client, "httpx2"):
        http2 = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=_app()), base_url="http://t"
        )
        return anthropic.AsyncAnthropic(api_key="k", base_url="http://t", http_client=http2)
    return anthropic.AsyncAnthropic(api_key="k", base_url="http://t", http_client=http)


def test_normalize_usage():
    assert normalize_usage({"input_tokens": 3, "output_tokens": 2}) == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
    }
    got = normalize_usage({"output_tokens": 2, "cache_read_input_tokens": 9})
    assert got["prompt_tokens_details"] == {"cached_tokens": 9}
    assert normalize_usage(None) == {}


@pytest.mark.asyncio
async def test_anthropic_json_and_sdk():
    async with _http() as http:
        raw = (
            await http.post("/v1/messages", json={"model": "m", "messages": []})
        ).json()
        assert raw["usage"]["x_yunshu"]["completion_tokens"] == 5
        assert "x_yunshu" not in raw
        client = _anthropic_client(http)
        msg = await client.messages.create(
            model="m", max_tokens=8, messages=[{"role": "user", "content": "x"}]
        )
        assert msg.content[0].text == "hi" and msg.usage.output_tokens == 5
        assert msg.usage.model_extra["x_yunshu"]["request_id"]


@pytest.mark.asyncio
async def test_anthropic_stream_sdk():
    async with _http() as http:
        client = _anthropic_client(http)
        async with client.messages.stream(
            model="m", max_tokens=8, messages=[{"role": "user", "content": "x"}]
        ) as stream:
            final = await stream.get_final_message()
        assert final.content[0].text == "hi"
        seen = []
        async with http.stream(
            "POST", "/v1/messages", json={"model": "m", "stream": True}
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:") and "message_delta" in line:
                    seen.append(json.loads(line[5:]))
        xy = seen[0]["usage"]["x_yunshu"]
        assert xy["completion_tokens"] == 5 and xy["ttft_ms"] is not None


@pytest.mark.asyncio
async def test_responses_json_and_stream_sdk():
    async with _http() as http:
        client = openai.AsyncOpenAI(
            api_key="k", base_url="http://t/v1", http_client=http
        )
        resp = await client.responses.create(model="m", input="x")
        xy = resp.usage.model_extra["x_yunshu"]
        assert xy["cached_tokens"] == 4 and xy["completion_tokens"] == 5
        stream = await client.responses.create(model="m", input="x", stream=True)
        events = [e async for e in stream]
        done = events[-1]
        assert done.type == "response.completed"
        assert done.response.usage.model_extra["x_yunshu"]["ttft_ms"] is not None
