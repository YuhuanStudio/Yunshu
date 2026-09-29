"""Small conformance checks against the OpenAI / Anthropic references (no model needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app
from yunshu_gateway.routers.chat import ChatCompletionRequest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    return TestClient(create_app())


def test_openai_error_object_has_param_and_code(client):
    r = client.post(
        "/v1/chat/completions", json={"model": "m", "messages": [], "temperature": 9}
    )
    err = r.json()["error"]
    assert r.status_code == 400
    assert set(err) >= {"message", "type", "param", "code"}
    assert err["type"] == "invalid_request_error"


def test_malformed_json_is_openai_400(client):
    r = client.post(
        "/v1/chat/completions",
        content=b"{bad",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_anthropic_error_object(client):
    r = client.post("/v1/messages", json={"model": "m", "messages": []})
    body = r.json()
    assert r.status_code == 400
    assert body["type"] == "error" and body["error"]["type"] == "invalid_request_error"


def test_stop_accepts_string_or_list():
    msgs = [{"role": "user", "content": "x"}]
    assert ChatCompletionRequest(model="m", messages=msgs, stop="END").stop == ["END"]
    assert ChatCompletionRequest(model="m", messages=msgs, stop=["a", "b"]).stop == [
        "a",
        "b",
    ]


def test_x_api_key_is_accepted_when_token_is_set(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "sekret")
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    c = TestClient(create_app())
    h = {"x-api-key": "sekret"}
    assert c.get("/v1/models/nope", headers=h).status_code == 404
    assert c.get("/v1/models/nope", headers={"x-api-key": "wrong"}).status_code == 401
    assert c.get("/v1/models/nope").status_code == 401


def test_models_list_carries_anthropic_fields(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list" and body["has_more"] is False
    assert "first_id" in body and "last_id" in body


def test_responses_input_tokens(client, monkeypatch):
    import yunshu_gateway.routers.tokenize as tok

    class T:
        def encode(self, text):
            return text.split()

    monkeypatch.setattr(tok, "_resolve_tokenizer", lambda model: T())
    r = client.post(
        "/v1/responses/input_tokens",
        json={
            "model": "m",
            "instructions": "be brief",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "one two three"}],
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json() == {"object": "response.input_tokens", "input_tokens": 5}


def test_removed_surfaces_are_gone(client):
    for method, path in [
        ("post", "/v1/video/generations"),
        ("post", "/v1/batch"),
        ("post", "/sleep"),
        ("post", "/v1/start_profile"),
        ("get", "/api/v1/bench/status"),
        ("get", "/v1/cachedContents"),
        ("post", "/v1/images/inpaint"),
        ("post", "/v1/token_count"),
    ]:
        assert getattr(client, method)(path).status_code in (404, 405), path


def test_staged_media_lives_under_media_dir(tmp_path, monkeypatch):
    """Anthropic base64 images are staged to disk; the VLM path only reads under the media dir."""
    from yunshu_engine.paths import stage_media_file
    from yunshu_engine.vlm_engine import _VALIDATE_LOCAL_PATH

    monkeypatch.setenv("YUNSHU_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.delenv("YUNSHU_ALLOW_LOCAL_FILES", raising=False)
    f = stage_media_file(".png")
    f.write(b"x")
    f.close()
    assert _VALIDATE_LOCAL_PATH(f.name)
