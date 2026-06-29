"""a grounded /v1/embeddings hunt found the endpoint fundamentally correct
(input polymorphism, base64 little-endian float32, Matryoshka truncate-then-renormalize-
before-base64, batch indexing, usage, isolation all verified). Two robustness gaps fixed:

1. NaN/Inf guard (the keystone, un-propagated to embeddings): a non-finite component
   serialized as a bare NaN/Infinity literal in float mode → INVALID JSON (RFC 8259, strict
   parsers reject the whole response) and garbage in base64. Now sanitized → 0.0.
2. token-id input + missing model returned 400 "tokenizer unavailable" where string input
   correctly returns 404 "model not found". Now both return 404.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from yunshu_gateway.routers import (
    embeddings as E,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.embeddings import EmbeddingRequest, _embed_and_format


class _StubEngine:
    _tokenizer = None  # → char/4 token count, no decode needed for str input


def test_nan_inf_sanitized_to_zero_and_valid_json(monkeypatch):
    async def _fake_gen(engine, texts, **kw):
        return [[float("nan"), float("inf"), -float("inf"), 0.5]]

    monkeypatch.setattr(E, "_generate_embeddings", _fake_gen)

    async def _fake_resolve(model):
        return _StubEngine()

    monkeypatch.setattr(E, "_resolve_embedding_engine", _fake_resolve)

    req = EmbeddingRequest(model="m", input="hi", encoding_format="float")
    out = asyncio.run(_embed_and_format(req))
    emb = out["data"][0]["embedding"]
    assert emb == [0.0, 0.0, 0.0, 0.5], emb
    # the whole response must be STRICT-valid JSON (no NaN/Infinity literals)
    json.dumps(out, allow_nan=False)


def test_token_id_input_missing_model_404_not_400(monkeypatch):
    async def _fake_resolve(model):
        return None  # model not found

    monkeypatch.setattr(E, "_resolve_embedding_engine", _fake_resolve)
    req = EmbeddingRequest(model="nope", input=[1, 2, 3])
    with pytest.raises(Exception) as ei:
        asyncio.run(_embed_and_format(req))
    # FastAPI HTTPException with 404 (was 400 "tokenizer unavailable")
    assert getattr(ei.value, "status_code", None) == 404


def test_base64_path_also_sanitized(monkeypatch):
    import base64
    import struct

    async def _fake_gen(engine, texts, **kw):
        return [[float("nan"), 1.0]]

    monkeypatch.setattr(E, "_generate_embeddings", _fake_gen)

    async def _fake_resolve(model):
        return _StubEngine()

    monkeypatch.setattr(E, "_resolve_embedding_engine", _fake_resolve)
    req = EmbeddingRequest(model="m", input="hi", encoding_format="base64")
    out = asyncio.run(_embed_and_format(req))
    raw = base64.b64decode(out["data"][0]["embedding"])
    vals = list(struct.unpack(f"{len(raw)//4}f", raw))
    assert vals == [0.0, 1.0]  # NaN packed as 0.0, not a garbage float


def test_nan_with_dimensions_still_renormalizes_to_unit(monkeypatch):
    """a NaN component must be sanitized BEFORE the Matryoshka truncation+renorm.
    With the old order norm=sqrt(...+NaN)=NaN → `norm>0` False → renorm SKIPPED → the NaN was
    zeroed last, leaving a NON-UNIT truncated vector ([3,4,NaN]→[3,4,0], L2=5). The NaN test
    above used no `dimensions`, and the Matryoshka renorm test did its math locally — so this
    ordering slipped through. Now it must renorm to a true unit vector."""
    import math

    async def _fake_gen(engine, texts, **kw):
        return [[3.0, 4.0, float("nan"), 9.0, 9.0]]   # native dim 5, NaN inside the kept slice

    monkeypatch.setattr(E, "_generate_embeddings", _fake_gen)

    async def _fake_resolve(model):
        return _StubEngine()

    monkeypatch.setattr(E, "_resolve_embedding_engine", _fake_resolve)
    req = EmbeddingRequest(model="m", input="hi", dimensions=3, encoding_format="float")
    out = asyncio.run(_embed_and_format(req))
    emb = out["data"][0]["embedding"]
    # [3,4,NaN] → NaN→0 → [3,4,0] → renorm → [0.6,0.8,0.0]
    assert emb == pytest.approx([0.6, 0.8, 0.0])
    assert math.isclose(math.sqrt(sum(x * x for x in emb)), 1.0, abs_tol=1e-9)  # UNIT
