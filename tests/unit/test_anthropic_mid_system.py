"""Instruction priority is preserved by canonical system hoisting."""

import os

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.message_adapter import keeps_mid_conversation_system


def test_instruction_messages_are_hoisted_for_all_families():
    assert not keeps_mid_conversation_system("Qwen3.8-27B-oQ4e-mtp")
    assert not keeps_mid_conversation_system("qwen2.5-3b")
    assert not keeps_mid_conversation_system("gemma-4-e4b-it")
    assert not keeps_mid_conversation_system("claude-3")
    assert not keeps_mid_conversation_system(None)


@pytest.fixture
def _engine():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.engine import set_engine

    engine = BatchedEngine()
    engine._model = object()
    engine._loaded = True
    engine._running = True
    yield engine, set_engine
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _captured_messages(engine, set_engine, monkeypatch, model_name, body):
    from yunshu_engine.batched_engine import GenerationOutput
    from yunshu_gateway.main import create_app

    engine.model_name = model_name
    set_engine(engine)
    captured = {}

    async def _fake(*args, **kwargs):
        captured["messages"] = kwargs.get("messages", kwargs.get("prompt"))
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    monkeypatch.setattr(engine, "generate", _fake)
    monkeypatch.setattr(engine, "chat", _fake)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    return captured["messages"]


BODY = {
    "model": "claude-x",
    "max_tokens": 10,
    "system": "TOP_SYSTEM",
    "messages": [
        {"role": "system", "content": "LEADING_NOTE"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "more"},
        {"role": "system", "content": "TURN_NOTE"},
    ],
}


def test_qwen_hoists_trailing_system_note(_engine, monkeypatch):
    engine, set_engine = _engine
    msgs = _captured_messages(engine, set_engine, monkeypatch, "Qwen3.8-27B", BODY)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert "TOP_SYSTEM" in msgs[0]["content"] and "LEADING_NOTE" in msgs[0]["content"]
    assert "TURN_NOTE" in msgs[0]["content"]


def test_other_families_still_hoist(_engine, monkeypatch):
    engine, set_engine = _engine
    msgs = _captured_messages(engine, set_engine, monkeypatch, "gemma-4-e4b", BODY)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "TURN_NOTE" in msgs[0]["content"]


def test_changing_note_preserves_conversation_turns(_engine, monkeypatch):
    engine, set_engine = _engine
    a = dict(
        BODY, messages=BODY["messages"][:-1] + [{"role": "system", "content": "N1"}]
    )
    b = dict(
        BODY, messages=BODY["messages"][:-1] + [{"role": "system", "content": "N2"}]
    )
    ma = _captured_messages(engine, set_engine, monkeypatch, "Qwen3.8-27B", a)
    mb = _captured_messages(engine, set_engine, monkeypatch, "Qwen3.8-27B", b)
    assert ma[1:] == mb[1:]
    assert "N1" in ma[0]["content"] and "N2" in mb[0]["content"]
