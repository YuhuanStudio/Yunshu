"""A background response holds its key's concurrency slot until generation ends and its
final tokens are counted to the key (the queued reply used to release both early)."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway import api_keys


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    store = api_keys.configure(tmp_path / "keys.json")
    from yunshu_engine.vlm_engine import VLMEngine
    from yunshu_gateway.engine import set_engine

    engine = VLMEngine("/models/Qwen3.8-27B")
    engine._model = object()
    engine._running = True

    class _Tok:
        chat_template = "x"

    engine._tokenizer = _Tok()
    gate = {"open": False}

    async def _generate(prompt=None, **kw):
        while not gate["open"]:
            await asyncio.sleep(0.01)
        return {
            "text": "done",
            "finish_reason": "stop",
            "prompt_tokens": 11,
            "completion_tokens": 5,
        }

    monkeypatch.setattr(engine, "generate", _generate)
    set_engine(engine)
    yield store, gate
    set_engine(None)
    api_keys._store = None


def test_background_response_keeps_the_slot_and_counts_its_tokens(setup):
    from yunshu_gateway.main import create_app

    store, gate = setup
    rec, sec = store.create("bg", quotas={"max_concurrent": 1})
    h = {"Authorization": f"Bearer {sec}"}
    body = {"model": "any", "input": "hi", "background": True}
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        r = client.post("/v1/responses", json=body, headers=h)
        assert r.status_code == 200 and r.json()["status"] == "queued", r.text
        rid = r.json()["id"]
        # still generating: the key's one slot is taken
        assert client.post("/v1/responses", json=body, headers=h).status_code == 429
        gate["open"] = True
        for _ in range(200):
            if client.get(f"/v1/responses/{rid}", headers=h).json().get("status") == (
                "completed"
            ):
                break
            time.sleep(0.02)
        time.sleep(0.1)
        st = store._state[rec.id]
        assert st.inflight == 0
        assert st.tok_sum >= 16, st.tok_sum
