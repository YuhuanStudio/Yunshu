"""Tool schemas the chat template renders itself (Qwen3.x native tools) sit outside `messages`;
the context guard / token budget must count them (an agent's 15-25K tokens of schemas decide
how much room the answer really has), and a request without max_tokens must not pay a full
prompt encode just because the default output limit equals the window."""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_gateway import streaming
from yunshu_gateway.routers import anthropic, chat, responses  # noqa: F401
from yunshu_gateway.token_budget import estimate_tools_tokens

BIG = "x" * 4000  # ~1000 tokens of schema text per tool
ANTHROPIC_TOOLS = [
    {
        "name": "big",
        "description": BIG,
        "input_schema": {"type": "object", "properties": {}},
    }
]
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "big",
            "description": BIG,
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "big",
        "description": BIG,
        "parameters": {"type": "object", "properties": {}},
    }
]


class _Tok:
    """Counts characters; records encode calls."""

    bos_token = None

    def __init__(self):
        self.calls = 0

    def encode(self, text, *a, **k):
        self.calls += 1
        return list(range(len(text)))


@pytest.fixture
def engine():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_gateway.engine import set_engine

    eng = BatchedEngine()
    eng._model = object()
    eng._loaded = True
    eng.model_name = "claude-test"
    eng._running = True
    eng._tokenizer = _Tok()
    set_engine(eng)
    yield eng
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


@pytest.fixture
def seen(monkeypatch):
    """Record the prompt-token estimate handed to validate_context_window, then stop the
    request there (418) so no generation is needed."""
    got: list[int] = []

    def stop(est, model, engine, *a, **k):
        got.append(est)
        raise HTTPException(status_code=418, detail="stop")

    monkeypatch.setattr(streaming, "validate_context_window", stop)
    monkeypatch.setattr(chat, "validate_context_window", stop)
    monkeypatch.setattr(streaming, "get_max_context_window", lambda *a, **k: 2000)
    return got


def _client():
    from yunshu_gateway.main import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def test_estimate_tools_tokens_is_cheap_and_scales_with_schema():
    assert estimate_tools_tokens(None) == 0
    assert estimate_tools_tokens([]) == 0
    assert 900 <= estimate_tools_tokens(OPENAI_TOOLS) <= 1200


def test_anthropic_guard_counts_native_tools(engine, seen, monkeypatch):
    def native(req, eng):
        req._native_active = True
        return True

    monkeypatch.setattr(anthropic, "_apply_native_tools", native)
    base = {
        "model": "claude-test",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    _client().post("/v1/messages", json=base)
    _client().post("/v1/messages", json={**base, "tools": ANTHROPIC_TOOLS})
    assert len(seen) == 2 and seen[1] - seen[0] >= 900


def test_responses_guard_counts_tools(engine, seen):
    base = {"model": "claude-test", "input": "hi", "max_output_tokens": 64}
    _client().post("/v1/responses", json=base)
    _client().post("/v1/responses", json={**base, "tools": RESPONSES_TOOLS})
    assert len(seen) == 2 and seen[1] - seen[0] >= 900


def test_chat_text_path_guard_counts_native_tools(engine, seen, monkeypatch):
    # native rendering: the tools stay out of `messages` (the template adds them)
    monkeypatch.setattr(chat, "_inject_tool_system_prompt", lambda m, *a, **k: m)
    monkeypatch.setattr(chat, "_native_tools_active", lambda: True)
    base = {
        "model": "claude-test",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    _client().post("/v1/chat/completions", json={**base, "tools": OPENAI_TOOLS})
    assert len(seen) == 1 and seen[0] >= 900


@pytest.mark.asyncio
async def test_chat_vlm_path_guard_counts_native_tools(monkeypatch):
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine.__new__(VLMEngine)
    engine._tokenizer = _Tok()
    engine._processor = None
    engine._config = {"max_position_embeddings": 100_000}
    monkeypatch.setattr(chat, "get_engine", lambda: engine)
    monkeypatch.setattr(chat, "get_model_manager", lambda: None)
    monkeypatch.setattr(chat, "_vlm_renders_tools", lambda e: True)
    budgets: list[int] = []
    monkeypatch.setattr(
        chat, "_apply_token_budget", lambda req, count, e: budgets.append(count)
    )
    for tools in (None, OPENAI_TOOLS):
        req = chat.ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            max_tokens=8,
            tools=tools,
        )
        response = await chat._handle_vlm_chat(
            req, [{"role": "user", "content": "hello"}], None
        )
        await response.body_iterator.aclose()
    assert budgets[1] - budgets[0] >= 900
