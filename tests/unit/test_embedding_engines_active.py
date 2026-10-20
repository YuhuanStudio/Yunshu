"""Multimodal embedding engines must expose has_active_requests so the
ModelManager can prove them idle and evict them."""

from __future__ import annotations

import pytest

from yunshu_engine.gemma_embedding_engine import GemmaEmbeddingEngine
from yunshu_engine.vl_embedding_engine import VLEmbeddingEngine


@pytest.mark.parametrize("cls", [VLEmbeddingEngine, GemmaEmbeddingEngine])
def test_idle_engine_reports_no_active_requests(cls):
    eng = cls("/models/some-embedding")
    assert callable(eng.has_active_requests)
    assert eng.has_active_requests() is False


@pytest.mark.asyncio
async def test_embed_counts_as_active_while_running():
    eng = VLEmbeddingEngine("/models/x")
    eng._loaded = True
    seen = []

    class _Model:
        def process(self, items, proc):
            seen.append(eng.has_active_requests())
            raise RuntimeError("stop here")

    eng._model = _Model()
    eng._executor = None  # default executor runs _run in a thread
    with pytest.raises(RuntimeError):
        await eng.embed(["hi"])
    assert seen == [True]
    assert eng.has_active_requests() is False
