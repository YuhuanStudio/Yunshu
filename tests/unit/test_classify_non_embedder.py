"""/v1/classify on a chat model answered a bare 500 "Classification failed"; /v1/pooling, /v1/score
and /v1/rerank answered a 400 that says the model cannot be an embedder (found by the real-server
route checks on Qwen3.5-0.8B)."""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app
from yunshu_gateway.routers import scoring as S  # noqa: N812

MSG = "This model architecture cannot be used as a text embedder; load a dedicated embedding model"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)

    async def resolve(model):
        return types.SimpleNamespace(_tokenizer=None)

    async def cannot(engine, texts, normalize=True):
        raise ValueError(MSG)

    monkeypatch.setattr(S, "_resolve_engine", resolve)
    monkeypatch.setattr(S, "_get_embeddings", cannot)
    return TestClient(create_app())


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/classify", {"model": "m", "input": "hi", "labels": ["a", "b"]}),
        ("/v1/score", {"model": "m", "text_1": "a", "text_2": "b"}),
        ("/v1/rerank", {"model": "m", "query": "q", "documents": ["a", "b"]}),
    ],
)
def test_non_embedder_is_a_400_with_the_reason(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 400, r.text
    msg = r.json()["error"]["message"]
    assert "cannot be used as a text embedder" in msg and "Error)" not in msg
