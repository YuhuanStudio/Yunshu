"""A chat request without max_tokens must not pay a full synchronous prompt encode: the default
output limit equals the window, so the old `chars + max_tokens < 0.9 * window` shortcut never
held. The prompt provably fits (tokens <= chars) and the room it leaves is >= window - chars."""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_gateway import streaming
from yunshu_gateway.routers import chat


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


def _client():
    from yunshu_gateway.main import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def test_default_max_tokens_does_not_encode_the_prompt(engine, monkeypatch):
    """No max_tokens: the default is as large as the window, so the old `chars + max_tokens <
    0.9 * window` shortcut never held and every request paid a full synchronous encode."""
    monkeypatch.setattr(streaming, "get_max_context_window", lambda *a, **k: 262144)
    counted: list[int] = []

    def stop(req, prompt_tokens, eng, basis="exact"):
        counted.append((prompt_tokens, basis))
        raise HTTPException(status_code=418, detail="stop")

    monkeypatch.setattr(chat, "_apply_token_budget", stop)
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hello world " * 200}],
        },
    )
    assert r.status_code == 418, r.text
    assert engine._tokenizer.calls == 0
    # the byte upper bound, not an encode, and the budget says so
    assert counted == [(len("hello world " * 200), "upper_bound")]


def test_small_window_does_not_over_clamp_from_the_bound(engine, monkeypatch):
    """32K window, ~5K-token prompt of 20K chars: the bound would leave ~12K for the answer; the
    exact count decides instead, so the grant is not cut to 12K."""
    monkeypatch.setattr(streaming, "get_max_context_window", lambda *a, **k: 32768)
    seen: list[tuple[int, str]] = []

    def stop(req, prompt_tokens, eng, basis="exact"):
        seen.append((prompt_tokens, basis))
        raise HTTPException(status_code=418, detail="stop")

    monkeypatch.setattr(chat, "_apply_token_budget", stop)
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "claude-test",
            "messages": [{"role": "user", "content": "x" * 20000}],
        },
    )
    assert r.status_code == 418, r.text
    assert engine._tokenizer.calls >= 1 and seen[0][1] == "exact"


def test_budget_report_names_its_basis():
    from yunshu_gateway.token_budget import plan

    assert "prompt_tokens_basis" not in plan(1000, 10, 50).report()
    assert (
        plan(1000, 10, 50, basis="upper_bound").report()["prompt_tokens_basis"]
        == "upper_bound"
    )


def test_explicit_max_tokens_near_the_window_still_counts_exactly(engine, monkeypatch):
    monkeypatch.setattr(streaming, "get_max_context_window", lambda *a, **k: 262144)
    seen_exact: list[int] = []

    def stop(req, prompt_tokens, eng, basis="exact"):
        seen_exact.append(prompt_tokens)
        raise HTTPException(status_code=418, detail="stop")

    monkeypatch.setattr(chat, "_apply_token_budget", stop)
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "claude-test",
            "max_tokens": 262144,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 418, r.text
    assert engine._tokenizer.calls >= 1 and seen_exact[0] > 0


@pytest.mark.asyncio
async def test_vlm_non_stream_does_not_re_encode_when_engine_reports_prompt_tokens(
    monkeypatch,
):
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine.__new__(VLMEngine)
    engine._tokenizer = _Tok()
    engine._processor = None
    engine._config = {"max_position_embeddings": 100_000}
    formatted: list[int] = []
    engine._format_prompt = lambda messages: formatted.append(1) or "prompt"

    async def generate(**kwargs):
        return {
            "text": "ok",
            "prompt_tokens": 7,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    engine.generate = generate
    monkeypatch.setattr(chat, "get_engine", lambda: engine)
    monkeypatch.setattr(chat, "get_model_manager", lambda: None)
    req = chat.ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
    )
    response = await chat._handle_vlm_chat(
        req, [{"role": "user", "content": "hello"}], None
    )
    assert response.status_code == 200
    assert b'"prompt_tokens":7' in response.body.replace(b" ", b"")
    assert formatted == []  # no template render for a fallback nobody needed
