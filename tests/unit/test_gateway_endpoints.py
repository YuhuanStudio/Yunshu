"""Tests for remaining gateway endpoints: models, images, audio, batches."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _setup_engine():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.engine import Engine, EngineConfig
    from yunshu_gateway.engine import set_engine

    engine = Engine(EngineConfig())
    engine._model = object()
    engine._model_name = "test-model"
    engine._running = True
    set_engine(engine)
    yield
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _client():
    from yunshu_gateway.main import create_app

    return TestClient(create_app())


class TestModelsEndpoint:
    def test_list_models(self):
        client = _client()
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_models_has_object_field(self):
        client = _client()
        resp = client.get("/v1/models")
        data = resp.json()
        for m in data["data"]:
            assert "id" in m
            assert "object" in m


class TestTokenizeEndpoint:
    def test_tokenize_no_model(self):
        client = _client()
        resp = client.post(
            "/v1/tokenize", json={"model": "nonexistent", "text": "hello"}
        )
        # Should fail with 404 since no real tokenizer
        assert resp.status_code in (404, 500)

    def test_token_count_no_model(self):
        client = _client()
        resp = client.post(
            "/v1/token_count", json={"model": "nonexistent", "prompt": "hello"}
        )
        assert resp.status_code in (404, 500)


class TestImagesEndpoint:
    def test_images_generations_no_engine(self):
        client = _client()
        resp = client.post(
            "/v1/images/generations",
            json={
                "model": "test",
                "prompt": "a cat",
                "n": 1,
                "size": "256x256",
            },
        )
        # Will fail since image engine not loaded
        assert resp.status_code in (200, 404, 500, 503)


class TestAudioEndpoint:
    def test_tts_no_engine(self):
        client = _client()
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "test",
                "input": "Hello world",
                "voice": "default",
            },
        )
        assert resp.status_code in (200, 404, 500, 503)

    def test_transcription_no_engine(self):
        client = _client()
        resp = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": "test",
            },
        )
        # Might fail with missing file or no engine (400 for validation error from custom handler)
        assert resp.status_code in (200, 400, 404, 422, 500, 503)

    def test_list_voices_uses_engine_catalogue(self):
        client = _client()
        with patch(
            "yunshu_gateway.routers.audio._list_tts_voices",
            return_value=["voice_a", "voice_b"],
        ):
            resp = client.get("/v1/audio/voices")
        assert resp.status_code == 200
        assert resp.json()["data"] == [
            {"id": "voice_a", "object": "voice"},
            {"id": "voice_b", "object": "voice"},
        ]


class TestCancelEndpoint:
    def test_cancel_routes_are_mounted_once_under_v1(self):
        client = _client()
        resp = client.get("/v1/active-generations")
        assert resp.status_code == 200
        # response now uses OpenAI list envelope (object/data) +
        # preserves legacy fields (active/count) for backwards-compat.
        body = resp.json()
        assert body["active"] == []
        assert body["count"] == 0
        assert body["object"] == "list"
        assert body["data"] == []

        resp = client.post("/v1/cancel", json={"cancel_all": True})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_legacy_double_v1_cancel_routes_are_not_mounted(self):
        client = _client()
        assert client.get("/v1/v1/active-generations").status_code == 404
        assert (
            client.post("/v1/v1/cancel", json={"cancel_all": True}).status_code == 404
        )


class TestBatchEndpoint:
    def test_batch_status_no_batch(self):
        client = _client()
        resp = client.post(
            "/v1/batches",
            json={
                "input_file_id": "test",
                "endpoint": "/v1/chat/completions",
            },
        )
        assert resp.status_code in (200, 404, 500, 503)


class TestEmbeddingsEndpoint:
    def test_embeddings_no_model(self):
        client = _client()
        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "nonexistent",
                "input": "hello world",
            },
        )
        assert resp.status_code in (404, 500, 503)


class TestChatLogprobs:
    """Test logprobs and new parameters in chat completions schema."""

    def test_chat_request_accepts_logprobs(self):
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        # Engine has no tokenizer so this 500s — schema parsing is what we test
        assert resp.status_code == 500

    def test_chat_request_accepts_n(self):
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 3,
            },
        )
        assert resp.status_code == 500

    def test_chat_request_accepts_user(self):
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "user-123",
            },
        )
        assert resp.status_code == 500

    def test_chat_rejects_invalid_logprobs_type(self):
        client = _client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": "not_a_bool",
            },
        )
        assert resp.status_code in (400, 422)


class TestChatLoRAAdapter:
    """Test LoRA adapter passthrough in chat completions."""

    def test_chat_accepts_lora_adapter_field(self):
        """lora_adapter field should be accepted without 422."""
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "lora_adapter": "my-lora-v1",
            },
        )
        # Schema accepts the field (not 422). A requested adapter that can't be
        # served is now a hard error instead of silently running the base model:
        # 400 if the engine has no LoRA manager (legacy Engine here), 404 if the
        # adapter id is unknown; absent a real engine it may also 500.
        assert resp.status_code in (400, 404, 500)

    def test_chat_accepts_null_lora_adapter(self):
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "lora_adapter": None,
            },
        )
        assert resp.status_code == 500

    def test_chat_without_lora_adapter_still_works(self):
        """Backward compatibility: requests without lora_adapter work."""
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 500

    def test_chat_streaming_accepts_lora_adapter(self):
        """Streaming path also accepts lora_adapter."""
        client = TestClient(_client().app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "lora_adapter": "test-adapter",
                "stream": True,
            },
        )
        # Streaming may 500 due to no real engine, but not 422
        assert resp.status_code in (200, 500)
