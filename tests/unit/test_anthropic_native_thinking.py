"""thinking={enabled} without tools: a natively reasoning model gets no <think> prompt
instruction and no forced regex grammar; a model without native reasoning still gets both."""

import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.routers import anthropic as anth


@pytest.fixture
def engine(monkeypatch):
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine

    eng = BatchedEngine()
    eng._model = object()
    eng._loaded = True
    eng.model_name = "claude-3"
    eng._running = True
    set_engine(eng)
    seen: dict = {}

    async def fake(*args, **kwargs):
        seen["kwargs"] = kwargs
        seen["args"] = args
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    monkeypatch.setattr(eng, "generate", fake)
    monkeypatch.setattr(eng, "chat", fake)
    yield seen
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _post():
    from yunshu_gateway.main import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    return client.post(
        "/v1/messages",
        json={
            "model": "claude-3",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )


def _sent(seen) -> str:
    return repr(seen.get("kwargs")) + repr(seen.get("args"))


def test_native_reasoning_model_gets_no_injection_or_grammar(engine, monkeypatch):
    monkeypatch.setattr(anth, "_model_reasons_natively", lambda model: True)
    r = _post()
    assert r.status_code == 200
    sent = _sent(engine)
    assert "IMPORTANT FORMAT REQUIREMENT" not in sent
    assert "[\\\\s\\\\S]+?</think>" not in sent and "</think>[" not in sent
    # the thinking block is still returned (empty placeholder) per the Anthropic contract
    assert r.json()["content"][0]["type"] == "thinking"


def test_non_reasoning_model_keeps_instruction_and_grammar(engine, monkeypatch):
    monkeypatch.setattr(anth, "_model_reasons_natively", lambda model: False)
    r = _post()
    assert r.status_code == 200
    sent = _sent(engine)
    assert "IMPORTANT FORMAT REQUIREMENT" in sent
    assert "</think>" in sent and "[\\\\s\\\\S]+?" in sent


def test_card_drives_the_native_check(monkeypatch):
    from yunshu_gateway import model_cards

    card = lambda supported: SimpleNamespace(reasoning={"supported": supported})  # noqa: E731
    monkeypatch.setattr(model_cards, "find_card", lambda m: card(True))
    assert anth._model_reasons_natively("x") is True
    monkeypatch.setattr(model_cards, "find_card", lambda m: card(False))
    assert anth._model_reasons_natively("x") is False
    # unknown placeholder name in single-model mode: fall back to the loaded engine's card
    monkeypatch.setattr(model_cards, "find_card", lambda m: None)
    monkeypatch.setattr(model_cards, "engine_card", lambda m=None: card(True))
    assert anth._model_reasons_natively("claude-sonnet-x") is True
    monkeypatch.setattr(model_cards, "engine_card", lambda m=None: None)
    assert anth._model_reasons_natively("claude-sonnet-x") is False
