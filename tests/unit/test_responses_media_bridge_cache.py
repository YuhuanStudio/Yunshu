"""The Responses media bridge forwards prompt_cache_* to the chat request."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi.responses import JSONResponse

from yunshu_gateway.routers import chat
from yunshu_gateway.routers import responses as resp_mod


def test_bridge_forwards_prompt_cache_fields(monkeypatch):
    seen = {}

    async def fake(chat_req, messages, request, json_schema=None):
        seen["req"] = chat_req
        return JSONResponse(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
                "usage": {},
            }
        )

    monkeypatch.setattr(chat, "_handle_vlm_chat", fake)
    req = resp_mod.ResponsesRequest(
        model="m",
        input="hi",
        store=False,
        prompt_cache_key="k1",
        prompt_cache_retention="24h",
        prompt_cache_options={"a": 1},
    )
    # only the request handed to the chat handler matters here
    with contextlib.suppress(Exception):
        asyncio.run(
            resp_mod._vlm_to_responses(
                req, [{"role": "user", "content": "hi"}], None, None, []
            )
        )
    got = seen["req"]
    assert got.prompt_cache_key == "k1"
    assert got.prompt_cache_retention == "24h"
    assert got.prompt_cache_options == {"a": 1}
