"""/v1/messages/count_tokens renders with the thinking switch generation uses: no ``thinking`` means
off (0.1.6), and a Qwen template then adds an empty think block that the count left out (release
M3 sweep: count 28 vs usage 30)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


class _Tok:
    bos_token = None

    def __init__(self):
        self.kwargs = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kw
    ):
        self.kwargs.append(kw)
        think = "" if kw.get("enable_thinking", True) else "<think>\n\n</think>\n\n"
        return "".join(m["content"] for m in messages) + think

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


@pytest.fixture
def client():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    engine = BatchedEngine()
    engine._model = object()
    engine._loaded = True
    engine.model_name = "m"
    engine._running = True
    engine._tokenizer = _Tok()
    set_engine(engine)
    yield TestClient(create_app(), raise_server_exceptions=False), engine._tokenizer
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


@pytest.mark.parametrize(
    "thinking, expect",
    [
        (None, False),
        ({"type": "disabled"}, False),
        ({"type": "enabled", "budget_tokens": 2048}, True),
    ],
)
def test_count_tokens_passes_the_generation_thinking_switch(client, thinking, expect):
    c, tok = client
    body = {"model": "m", "messages": [{"role": "user", "content": "hi there"}]}
    if thinking:
        body["thinking"] = thinking
    r = c.post("/v1/messages/count_tokens", json=body)
    assert r.status_code == 200, r.text
    assert tok.kwargs and tok.kwargs[-1].get("enable_thinking") is expect


def test_adaptive_keeps_the_template_default(client):
    c, tok = client
    r = c.post(
        "/v1/messages/count_tokens",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
        },
    )
    assert r.status_code == 200, r.text
    assert "enable_thinking" not in tok.kwargs[-1]
