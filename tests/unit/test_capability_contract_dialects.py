"""The capability contract gates every dialect, each answering in its own native error shape.

Same policy as /v1/chat/completions: a media part the model cannot take is a 400 naming the
modality; reasoning_effort / thinking on a model without a reasoning mode is accepted with no
effect; a model that does not generate text rejects generation-only fields.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.model_manager import ModelManager, ModelType
from yunshu_gateway import model_cards
from yunshu_gateway.main import create_app
from yunshu_gateway.routers import chat as chat_router
from yunshu_gateway.routers import models as models_router

from .test_capability_contract import TEXT_CFG, VLM_CFG, _write_model

IMG = {"type": "image/png", "data": "QUJD"}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("models")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
        _write_model(tmp_path, "plain", TEXT_CFG)
        _write_model(tmp_path, "vision", VLM_CFG, "{{ tools }}")
        _write_model(
            tmp_path,
            "emb",
            {**TEXT_CFG, "architectures": ["BertModel"], "model_type": "bert"},
        )
        mgr = ModelManager(max_memory_bytes=None)
        mgr.register_model("plain", str(tmp_path / "plain"))
        mgr.register_model("vision", str(tmp_path / "vision"))
        mgr.register_model("emb", str(tmp_path / "emb"), model_type=ModelType.EMBEDDING)
        for mod in (models_router, model_cards, chat_router):
            monkeypatch.setattr(mod, "get_model_manager", lambda: mgr, raising=False)
        yield TestClient(create_app())


def anthropic_image(name="image"):
    return {
        "type": name,
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def anth(c, model, content, **extra):
    return c.post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": content}],
            **extra,
        },
    )


def assert_anthropic_400(r, word):
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert word in body["error"]["message"]


def assert_openai_400(r, word):
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and word in err["message"]
    assert err["param"] is None and err["code"] == "bad_request"


# ── /v1/messages ─────────────────────────────────────────────────────────────────
def test_messages_image_block_on_text_model(served):
    r = anth(served, "plain", [{"type": "text", "text": "x"}, anthropic_image()])
    assert_anthropic_400(r, "image")


def test_messages_image_inside_tool_result_on_text_model(served):
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [anthropic_image()],
        }
    ]
    assert_anthropic_400(anth(served, "plain", content), "image")


def test_messages_image_block_on_vision_model_passes_the_gate(served):
    r = anth(served, "vision", [anthropic_image()])
    assert r.status_code != 400 or "does not support" not in r.text


def test_messages_thinking_on_non_reasoning_model_is_accepted(served):
    r = anth(
        served,
        "plain",
        "hi",
        thinking={"type": "enabled", "budget_tokens": 1024},
        reasoning_effort="high",
    )
    assert "does not support" not in r.text


def test_messages_tools_on_embedding_model(served):
    r = anth(
        served,
        "emb",
        "hi",
        tools=[{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
    )
    assert_anthropic_400(r, "tools")


def test_anthropic_sdk_raises_bad_request(served):
    anthropic = pytest.importorskip("anthropic")
    cl = anthropic.Anthropic(
        base_url="http://testserver", api_key="x", http_client=served, max_retries=0
    )
    with pytest.raises(anthropic.BadRequestError) as e:
        cl.messages.create(
            model="plain",
            max_tokens=8,
            messages=[{"role": "user", "content": [anthropic_image()]}],
        )
    assert "image" in str(e.value)


# ── /v1/responses ────────────────────────────────────────────────────────────────
def resp(c, model, content, **extra):
    return c.post(
        "/v1/responses",
        json={
            "model": model,
            "input": [{"role": "user", "content": content}],
            **extra,
        },
    )


def test_responses_input_image_on_text_model(served):
    r = resp(
        served,
        "plain",
        [
            {"type": "input_text", "text": "x"},
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
        ],
    )
    assert_openai_400(r, "image")


def test_responses_input_audio_on_vision_model(served):
    r = resp(
        served,
        "vision",
        [{"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}}],
    )
    assert_openai_400(r, "audio")


def test_responses_function_output_with_image_on_text_model(served):
    r = served.post(
        "/v1/responses",
        json={
            "model": "plain",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,QUJD",
                        }
                    ],
                }
            ],
        },
    )
    assert_openai_400(r, "image")


def test_responses_reasoning_on_non_reasoning_model_is_accepted(served):
    r = resp(
        served,
        "plain",
        [{"type": "input_text", "text": "hi"}],
        reasoning={"effort": "high"},
    )
    assert "does not support" not in r.text


def test_responses_tools_on_embedding_model(served):
    r = resp(
        served,
        "emb",
        [{"type": "input_text", "text": "hi"}],
        tools=[{"type": "function", "name": "f", "parameters": {"type": "object"}}],
    )
    assert_openai_400(r, "tools")


def test_openai_sdk_responses_bad_request(served):
    openai = pytest.importorskip("openai")
    cl = openai.OpenAI(
        base_url="http://testserver/v1", api_key="x", http_client=served, max_retries=0
    )
    with pytest.raises(openai.BadRequestError) as e:
        cl.responses.create(
            model="plain",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,QUJD",
                        }
                    ],
                }
            ],
        )
    assert "image" in str(e.value)


# ── /v1/completions ──────────────────────────────────────────────────────────────
def test_completions_on_embedding_model_rejects_generation_fields(served):
    r = served.post(
        "/v1/completions",
        json={"model": "emb", "prompt": "hi", "logprobs": 2},
    )
    assert_openai_400(r, "logprobs")


def test_completions_response_format_on_embedding_model(served):
    r = served.post(
        "/v1/completions",
        json={
            "model": "emb",
            "prompt": "hi",
            "response_format": {"type": "json_object"},
        },
    )
    assert_openai_400(r, "response_format")


def test_completions_ordinary_request_is_not_gated(served):
    r = served.post(
        "/v1/completions",
        json={
            "model": "plain",
            "prompt": "hi",
            "logprobs": 2,
            "reasoning_effort": "high",
        },
    )
    assert "does not support" not in r.text


# ── Ollama ───────────────────────────────────────────────────────────────────────
def ollama_loopback(served, monkeypatch):
    import httpx

    from yunshu_gateway.routers import ollama as ollama_router

    monkeypatch.setattr(
        ollama_router,
        "_client",
        lambda request: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=served.app),
            base_url="http://testserver",
            timeout=None,
        ),
    )


def test_ollama_chat_images_on_text_model(served, monkeypatch):
    ollama_loopback(served, monkeypatch)
    r = served.post(
        "/api/chat",
        json={
            "model": "plain",
            "messages": [{"role": "user", "content": "x", "images": ["QUJD"]}],
            "stream": False,
        },
    )
    assert r.status_code == 400, r.text
    assert set(r.json()) == {"error"} and "image" in r.json()["error"]


def test_ollama_generate_images_on_text_model_streams_nothing(served, monkeypatch):
    ollama_loopback(served, monkeypatch)
    r = served.post(
        "/api/generate",
        json={"model": "plain", "prompt": "x", "images": ["QUJD"]},
    )
    assert r.status_code == 400
    assert "image" in json.loads(r.text.splitlines()[0])["error"]


def test_ollama_think_on_non_reasoning_model_is_accepted(served, monkeypatch):
    ollama_loopback(served, monkeypatch)
    r = served.post(
        "/api/chat",
        json={
            "model": "plain",
            "messages": [{"role": "user", "content": "hi"}],
            "think": True,
            "stream": False,
        },
    )
    assert "does not support" not in r.text
