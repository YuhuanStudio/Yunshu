"""Qwen3-VL multimodal embedding + reranker wiring.

Model-free tests (no weights): detection, input normalization, and the router
helpers driven against a stub engine. The real-model cross-modal correctness
(text↔image cosine, cross-encoder relevance) is verified separately same-process
— not in CI, which has no model weights.
"""

import json

import pytest

from yunshu_engine.model_manager import ModelType, _detect_model_type
from yunshu_engine.vl_embedding_engine import VLEmbeddingEngine, _normalize_item


def _write_config(tmp_path, name, cfg):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


# ── detection ──────────────────────────────────────────────────────


def test_detect_vl_embedding(tmp_path):
    p = _write_config(
        tmp_path,
        "Qwen3-VL-Embedding-2B-8bit",
        {"model_type": "qwen3_vl", "vision_config": {"x": 1}},
    )
    assert _detect_model_type(p) == ModelType.EMBEDDING


def test_detect_vl_reranker(tmp_path):
    p = _write_config(
        tmp_path,
        "Qwen3-VL-Reranker-2B-8bit",
        {"model_type": "qwen3_vl", "vision_config": {"x": 1}},
    )
    assert _detect_model_type(p) == ModelType.RERANKER


def test_plain_qwen3_vl_still_vlm(tmp_path):
    # a normal vision model (no embedding/reranker in the name) must NOT be hijacked
    p = _write_config(
        tmp_path,
        "Qwen3-VL-7B-Instruct",
        {"model_type": "qwen3_vl", "vision_config": {"x": 1}},
    )
    assert _detect_model_type(p) == ModelType.VLM


def test_text_embedding_name_without_vision_not_hijacked(tmp_path):
    # "embedding" in the name but no vision → NOT a VL embedder (text embedder path)
    p = _write_config(tmp_path, "bge-embedding-base", {"model_type": "bert"})
    assert _detect_model_type(p) != ModelType.EMBEDDING


# ── input normalization ────────────────────────────────────────────


def test_normalize_item_str_to_dict():
    assert _normalize_item("hello") == {"text": "hello"}


def test_normalize_item_dict_passthrough():
    d = {"image": "x.png", "instruction": "find"}
    assert _normalize_item(d) is d


def test_normalize_item_rejects_other():
    with pytest.raises(ValueError):
        _normalize_item(123)


def test_is_reranker_from_name():
    assert VLEmbeddingEngine("mlx-community/Qwen3-VL-Reranker-2B-8bit").is_reranker
    assert not VLEmbeddingEngine("mlx-community/Qwen3-VL-Embedding-2B-8bit").is_reranker


# ── router helpers against a stub engine ───────────────────────────


class _StubEmbed:
    is_reranker = False

    async def embed(self, inputs, instruction=None):
        # echo back deterministic vectors; record what it received
        self.seen = (inputs, instruction)
        return [[float(i + 1), 0.0, 0.0] for i in range(len(inputs))]


class _StubRerank:
    async def rerank(self, query, documents, instruction=None):
        self.seen = (query, documents, instruction)
        # higher score for later docs so sort order is observable
        return [0.1 * (i + 1) for i in range(len(documents))]


@pytest.mark.asyncio
async def test_embed_multimodal_formats_and_validates():
    from yunshu_gateway.routers.embeddings import EmbeddingRequest, _embed_multimodal

    eng = _StubEmbed()
    req = EmbeddingRequest(
        model="vl",
        input=[{"text": "q", "instruction": "find"}, {"image": "a.png"}],
        instruction="default-instr",
    )
    resp = await _embed_multimodal(req, eng)
    assert resp["object"] == "list"
    assert len(resp["data"]) == 2
    assert resp["data"][0]["embedding"] == [1.0, 0.0, 0.0]
    assert resp["usage"]["prompt_tokens"] == 2
    # the default instruction is threaded through to the engine
    assert eng.seen[1] == "default-instr"


@pytest.mark.asyncio
async def test_embed_multimodal_base64_encoding():
    from yunshu_gateway.routers.embeddings import EmbeddingRequest, _embed_multimodal

    req = EmbeddingRequest(
        model="vl", input=[{"image": "a.png"}], encoding_format="base64"
    )
    resp = await _embed_multimodal(req, _StubEmbed())
    assert isinstance(resp["data"][0]["embedding"], str)


@pytest.mark.asyncio
async def test_embed_multimodal_rejects_objectless_item():
    from fastapi import HTTPException

    from yunshu_gateway.routers.embeddings import EmbeddingRequest, _embed_multimodal

    req = EmbeddingRequest(model="vl", input=[{"foo": "bar"}])
    with pytest.raises(HTTPException) as ei:
        await _embed_multimodal(req, _StubEmbed())
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_rerank_cross_encoder_sorts_and_top_n():
    import json as _json

    from yunshu_gateway.routers.scoring import RerankRequest, _rerank_cross_encoder

    docs = ["a", "b", "c"]
    req = RerankRequest(model="vl-rr", query="q", documents=docs, top_n=2)
    resp = await _rerank_cross_encoder(req, _StubRerank(), docs)
    body = _json.loads(resp.body)
    # scores 0.1,0.2,0.3 → sorted desc → indices [2,1], top_n=2
    assert [r["index"] for r in body["results"]] == [2, 1]
    assert (
        body["results"][0]["relevance_score"] >= body["results"][1]["relevance_score"]
    )


# ── multimodal rerank schema + fallback guard ──────────────────────


def test_rerank_request_multimodal_parses():
    from yunshu_gateway.routers.scoring import RerankRequest

    r = RerankRequest(
        model="m", query={"text": "q"}, documents=[{"image": "a.png"}, "plain text"]
    )
    assert r.is_multimodal is True


def test_rerank_request_text_only_not_multimodal():
    from yunshu_gateway.routers.scoring import RerankRequest

    r = RerankRequest(model="m", query="q", documents=["a", "b"])
    assert r.is_multimodal is False


def test_rerank_request_rejects_objectless_doc():
    from pydantic import ValidationError

    from yunshu_gateway.routers.scoring import RerankRequest

    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="q", documents=[{"foo": "bar"}])


@pytest.mark.asyncio
async def test_rerank_cross_encoder_echoes_image_doc():
    import json as _json

    from yunshu_gateway.routers.scoring import RerankRequest, _rerank_cross_encoder

    docs = [{"image": "a.png"}, "plain text"]
    req = RerankRequest(
        model="vl-rr", query={"text": "q"}, documents=docs, return_documents=True
    )
    resp = await _rerank_cross_encoder(req, _StubRerank(), docs)
    body = _json.loads(resp.body)
    by_idx = {r["index"]: r["document"] for r in body["results"]}
    # image doc echoed as the object; string doc wrapped as {"text": ...}
    assert by_idx[0] == {"image": "a.png"}
    assert by_idx[1] == {"text": "plain text"}


@pytest.mark.asyncio
async def test_rerank_multimodal_rejected_on_cosine_fallback(monkeypatch):
    # a non-VL (text bi-encoder) engine must reject image inputs with 400
    from fastapi import HTTPException

    from yunshu_gateway.routers import scoring

    class _PlainEngine:
        is_loaded = True

    async def _fake_resolve(_model):
        return _PlainEngine()

    monkeypatch.setattr(scoring, "_resolve_engine", _fake_resolve)
    req = scoring.RerankRequest(
        model="text-embedder", query="q", documents=[{"image": "a.png"}]
    )

    class _Req:
        # minimal stand-in for fastapi Request used by the permission checks
        headers: dict = {}
        state = type("S", (), {})()

    monkeypatch.setattr(scoring, "_check_permission", lambda *a, **k: None)
    monkeypatch.setattr(scoring, "_check_model_access", lambda *a, **k: None)
    with pytest.raises(HTTPException) as ei:
        await scoring.create_rerank(req, _Req())
    assert ei.value.status_code == 400
    assert "cross-encoder" in ei.value.detail
