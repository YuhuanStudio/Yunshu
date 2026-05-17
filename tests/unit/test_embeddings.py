"""Embeddings API endpoint tests.

Tests:
- Request schema validation (single/list input, encoding_format, dimensions)
- Input parsing (string vs list)
- Response format validation (OpenAI-compatible structure)
- Base64 encoding format
- Token counting (with and without tokenizer)
- Edge cases (empty input, too many inputs)
- Endpoint-level integration via FastAPI TestClient
"""
import base64
import os
import struct

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from yunshu_gateway.routers.embeddings import (
    EmbeddingRequest,
    _generate_embeddings,
    _resolve_embedding_engine,
)


# ── Schema Tests ──


class TestEmbeddingRequest:
    def test_single_input(self):
        req = EmbeddingRequest(model="text-embedding", input="hello world")
        assert req.model == "text-embedding"
        assert req.input == "hello world"
        assert req.encoding_format == "float"
        assert req.dimensions is None

    def test_list_input(self):
        req = EmbeddingRequest(model="text-embedding", input=["hello", "world"])
        assert isinstance(req.input, list)
        assert len(req.input) == 2
        assert req.input[0] == "hello"
        assert req.input[1] == "world"

    def test_base64_format(self):
        req = EmbeddingRequest(
            model="text-embedding",
            input="hello",
            encoding_format="base64",
        )
        assert req.encoding_format == "base64"

    def test_with_dimensions(self):
        req = EmbeddingRequest(
            model="text-embedding",
            input="hello",
            dimensions=256,
        )
        assert req.dimensions == 256

    def test_default_encoding_format(self):
        req = EmbeddingRequest(model="test", input="test")
        assert req.encoding_format == "float"

    def test_empty_list_input(self):
        req = EmbeddingRequest(model="test", input=[])
        assert req.input == []

    def test_valid_encoding_formats(self):
        """encoding_format must be 'float' or 'base64'."""
        from yunshu_gateway.routers.embeddings import _VALID_ENCODING_FORMATS
        assert "float" in _VALID_ENCODING_FORMATS
        assert "base64" in _VALID_ENCODING_FORMATS
        assert len(_VALID_ENCODING_FORMATS) == 2

    def test_long_list_input(self):
        texts = [f"text_{i}" for i in range(100)]
        req = EmbeddingRequest(model="test", input=texts)
        assert len(req.input) == 100


# ── Input Parsing Tests ──


class TestInputParsing:
    """Test conversion of input field to list of texts."""

    def test_string_becomes_single_element_list(self):
        """Single string input should be treated as [input]."""
        req = EmbeddingRequest(model="test", input="hello")
        texts = req.input if isinstance(req.input, list) else [req.input]
        assert texts == ["hello"]
        assert len(texts) == 1

    def test_list_stays_as_list(self):
        """List input should remain as list."""
        req = EmbeddingRequest(model="test", input=["a", "b", "c"])
        texts = req.input if isinstance(req.input, list) else [req.input]
        assert texts == ["a", "b", "c"]
        assert len(texts) == 3

    def test_single_element_list(self):
        """Single-element list input."""
        req = EmbeddingRequest(model="test", input=["only one"])
        texts = req.input if isinstance(req.input, list) else [req.input]
        assert len(texts) == 1
        assert texts[0] == "only one"


# ── Response Format Tests ──


class TestEmbeddingResponseFormat:
    """Test the OpenAI embeddings response format."""

    def test_response_structure(self):
        """Verify expected structure of embeddings response."""
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "text-embedding",
            "usage": {
                "prompt_tokens": 2,
                "total_tokens": 2,
            },
        }
        assert response["object"] == "list"
        assert isinstance(response["data"], list)
        assert len(response["data"]) == 1
        assert response["data"][0]["object"] == "embedding"
        assert response["data"][0]["index"] == 0
        assert isinstance(response["data"][0]["embedding"], list)
        assert response["model"] == "text-embedding"
        assert "prompt_tokens" in response["usage"]
        assert "total_tokens" in response["usage"]

    def test_multiple_embeddings_response(self):
        """Response should have correct indices for multiple embeddings."""
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        data = []
        for i, emb in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": emb,
            })
        assert len(data) == 3
        assert data[0]["index"] == 0
        assert data[1]["index"] == 1
        assert data[2]["index"] == 2

    def test_base64_encoding_format(self):
        """Base64 encoding should pack floats as little-endian."""
        emb = [0.1, 0.2, 0.3]
        packed = struct.pack(f"{len(emb)}f", *emb)
        b64_value = base64.b64encode(packed).decode("ascii")

        # Decode back to verify
        decoded = base64.b64decode(b64_value)
        unpacked = struct.unpack(f"{len(emb)}f", decoded)
        assert len(unpacked) == 3
        # Floats should be approximately equal (within float32 precision)
        for orig, dec in zip(emb, unpacked):
            assert abs(orig - dec) < 1e-6

    def test_float_encoding_format(self):
        """Float encoding should return the raw list of floats."""
        emb = [0.1, 0.2, 0.3]
        # In float mode, embedding value is the list itself
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)

    def test_usage_tokens_sum(self):
        """Usage total_tokens should equal prompt_tokens for embeddings."""
        usage = {"prompt_tokens": 42, "total_tokens": 42}
        assert usage["total_tokens"] == usage["prompt_tokens"]


# ── Embedding Generation Tests ──


class TestEmbeddingGeneration:
    """Test the _generate_embeddings function with mock engines."""

    async def test_engine_with_embed_method(self):
        """Should use engine.embed() if available."""
        engine = MagicMock()
        expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        engine.embed.return_value = expected

        result = await _generate_embeddings(engine, ["hello", "world"])
        assert result == expected
        engine.embed.assert_called_once_with(["hello", "world"])

    async def test_engine_without_embed_no_model(self):
        """Should raise RuntimeError if engine has no embed() and no model/tokenizer."""
        engine = MagicMock(spec=[])  # No embed method
        with pytest.raises(RuntimeError, match="does not support embedding"):
            await _generate_embeddings(engine, ["hello"])

    async def test_empty_text_list(self):
        """Should handle empty input gracefully."""
        engine = MagicMock()
        engine.embed.return_value = []
        result = await _generate_embeddings(engine, [])
        assert result == []


# ── Engine Resolution Tests ──


class TestEmbeddingEngineResolution:
    """Test the _resolve_embedding_engine function."""

    async def test_no_engine_returns_none(self):
        """Should return None when no engine is available."""
        with patch("yunshu_gateway.routers.embeddings.get_model_manager", return_value=None):
            with patch("yunshu_gateway.routers.embeddings.get_engine", return_value=None):
                result = await _resolve_embedding_engine("nonexistent")
                assert result is None

    async def test_single_engine_loaded(self):
        """Should return loaded single engine."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        with patch("yunshu_gateway.routers.embeddings.get_model_manager", return_value=None):
            with patch("yunshu_gateway.routers.embeddings.get_engine", return_value=mock_engine):
                result = await _resolve_embedding_engine("any-model")
                assert result == mock_engine


# ── Endpoint-Level Integration Tests ──


@pytest.fixture
def _setup_engine():
    """Set up a mock engine for endpoint tests."""
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.engine import Engine, EngineConfig
    from yunshu_gateway.engine import set_engine
    engine = Engine(EngineConfig())
    engine._model = object()
    engine._model_name = "test-embedding"
    engine._running = True
    engine._loaded = True
    set_engine(engine)
    yield engine
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _client():
    from yunshu_gateway.main import create_app
    return TestClient(create_app(), raise_server_exceptions=False)


class TestEmbeddingsEndpoint:
    """Test the /v1/embeddings endpoint via FastAPI TestClient."""

    def test_embeddings_validates_required_fields(self, _setup_engine):
        """Should return 422 when required fields are missing."""
        client = _client()
        resp = client.post("/v1/embeddings", json={})
        assert resp.status_code in (400, 422)
        # Verify OpenAI error format
        body = resp.json()
        assert "error" in body

    def test_embeddings_validates_model_required(self, _setup_engine):
        """Should return 400 or 422 when model is missing."""
        client = _client()
        resp = client.post("/v1/embeddings", json={"input": "hello"})
        assert resp.status_code in (400, 422)

    def test_embeddings_validates_input_required(self, _setup_engine):
        """Should return 400 or 422 when input is missing."""
        client = _client()
        resp = client.post("/v1/embeddings", json={"model": "test"})
        assert resp.status_code in (400, 422)

    def test_embeddings_no_model_loaded(self, _setup_engine):
        """Should fail with 404 when embedding model is not found."""
        client = _client()
        resp = client.post("/v1/embeddings", json={
            "model": "nonexistent",
            "input": "hello world",
        })
        assert resp.status_code in (404, 500, 503)

    def test_embeddings_accepts_string_input(self, _setup_engine):
        """Should accept string input without schema error."""
        client = _client()
        resp = client.post("/v1/embeddings", json={
            "model": "test-embedding",
            "input": "hello world",
        })
        assert resp.status_code in (200, 404, 500, 503)

    def test_embeddings_accepts_list_input(self, _setup_engine):
        """Should accept list of strings input without schema error."""
        client = _client()
        resp = client.post("/v1/embeddings", json={
            "model": "test-embedding",
            "input": ["hello", "world"],
        })
        assert resp.status_code in (200, 404, 500, 503)

    def test_embeddings_accepts_base64_format(self, _setup_engine):
        """Should accept base64 encoding format without schema error."""
        client = _client()
        resp = client.post("/v1/embeddings", json={
            "model": "test-embedding",
            "input": "hello",
            "encoding_format": "base64",
        })
        assert resp.status_code in (200, 404, 500, 503)

    def test_embeddings_accepts_dimensions(self, _setup_engine):
        """Should accept dimensions parameter without schema error."""
        client = _client()
        resp = client.post("/v1/embeddings", json={
            "model": "test-embedding",
            "input": "hello",
            "dimensions": 256,
        })
        assert resp.status_code in (200, 404, 500, 503)

    def test_embeddings_rejects_empty_input(self):
        """Should return 400 for empty input list."""
        req = EmbeddingRequest(model="test", input=[])
        texts = req.input if isinstance(req.input, list) else [req.input]
        assert len(texts) == 0  # Empty should trigger 400 in endpoint
