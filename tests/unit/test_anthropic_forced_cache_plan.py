"""Forced-tool assistant prefill must also appear in the cache diagnostic render."""

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.prompt_caching import strip_markers
from yunshu_gateway.engine import set_engine
from yunshu_gateway.routers import anthropic


@pytest.mark.parametrize("choice", [{"type": "any"}, {"type": "tool", "name": "shell"}])
@pytest.mark.parametrize("prefill", [False, True])
def test_forced_tool_plan_matches_generated_messages(monkeypatch, choice, prefill):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    engine = BatchedEngine()
    # Generation is replaced below; this fixture tests prompt construction only.
    monkeypatch.setattr(engine, "validate_forced_tools", lambda *a: None)
    engine._model = object()
    engine._loaded = True
    engine._running = True
    engine.model_name = "forced-cache-test"
    seen = []

    async def capture(engine, messages, req, stop, **kwargs):
        plan = req._prompt_cache_plan
        assert plan["markers"]
        assert strip_markers(plan["messages"], plan["markers"]) == messages
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].endswith("<tool_call>\n")
        seen.append(messages)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(anthropic, "_non_stream_batched", capture)
    from yunshu_gateway.main import create_app

    messages = [{"role": "user", "content": "Call shell"}]
    if prefill:
        messages.append({"role": "assistant", "content": "I will call it. "})
    set_engine(engine)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": engine.model_name,
                    "max_tokens": 64,
                    "system": [
                        {
                            "type": "text",
                            "text": "Use tools.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": messages,
                    "tool_choice": choice,
                    "tools": [
                        {
                            "name": "shell",
                            "input_schema": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                            },
                        }
                    ],
                },
            )
        assert response.status_code == 200, response.text
        assert seen
    finally:
        set_engine(None)
