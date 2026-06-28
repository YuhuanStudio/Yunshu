"""the empty-embedding zero-vector fallback hardcoded 768 when the WHOLE
batch was empty. A 768-dim zero vector silently mismatches a 384-dim (e5-small) or
1024-dim (bge-large) model — the OpenAI contract requires every data[] entry in one
response to be equal-length, and a wrong-dim zero vector poisons the client's vector
store. Fallback now asks the engine for its real hidden_size."""
from __future__ import annotations

import asyncio

from yunshu_gateway.routers import embeddings as emb_mod
from yunshu_gateway.routers.embeddings import EmbeddingRequest, _embed_and_format


class _FakeEngine:
    """Reports a non-768 hidden_size; has no real tokenizer."""
    _tokenizer = None
    is_loaded = True

    def __init__(self, hidden):
        self._hidden = hidden

    def _get_hidden_size(self):
        return self._hidden


def _run(monkeypatch, hidden, n_inputs):
    eng = _FakeEngine(hidden)

    async def _fake_resolve(model_id):
        return eng

    async def _fake_gen(engine, texts, **kw):
        # degenerate path: engine returns an EMPTY vector for every input
        return [[] for _ in texts]

    monkeypatch.setattr(emb_mod, "_resolve_embedding_engine", _fake_resolve)
    monkeypatch.setattr(emb_mod, "_generate_embeddings", _fake_gen)
    req = EmbeddingRequest(model="m", input=["a"] * n_inputs)
    return asyncio.run(_embed_and_format(req))


def test_whole_batch_empty_uses_engine_hidden_size(monkeypatch):
    out = _run(monkeypatch, 384, 2)
    dims = {len(d["embedding"]) for d in out["data"]}
    assert dims == {384}, f"expected all 384-dim, got {dims}"


def test_large_model_hidden_size(monkeypatch):
    out = _run(monkeypatch, 1024, 3)
    assert all(len(d["embedding"]) == 1024 for d in out["data"])


def test_source_consults_get_hidden_size():
    import inspect
    src = inspect.getsource(emb_mod)
    assert "engine._get_hidden_size()" in src
