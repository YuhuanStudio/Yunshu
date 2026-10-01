"""F01: the per-model capability contract is served and enforced (explicit 400, no silent ignore)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from yunshu_engine import capability_contract
from yunshu_engine.model_card import build_model_card
from yunshu_engine.model_manager import ModelManager
from yunshu_gateway import model_cards
from yunshu_gateway.main import create_app
from yunshu_gateway.routers import chat as chat_router
from yunshu_gateway.routers import models as models_router


def _write_model(root, name, config, template=""):
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    if template:
        (d / "chat_template.jinja").write_text(template)
    return d


TEXT_CFG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "max_position_embeddings": 8192,
    "hidden_size": 64,
}
VLM_CFG = {
    "model_type": "qwen2_vl",
    "architectures": ["Qwen2VLForConditionalGeneration"],
    "max_position_embeddings": 8192,
    "vision_config": {"hidden_size": 8},
}


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    _write_model(tmp_path, "plain", TEXT_CFG)
    _write_model(tmp_path, "vision", VLM_CFG, "{{ tools }}")
    mgr = ModelManager(max_memory_bytes=None)
    mgr.register_model("plain", str(tmp_path / "plain"))
    mgr.register_model("vision", str(tmp_path / "vision"))
    for mod in (models_router, model_cards, chat_router):
        monkeypatch.setattr(mod, "get_model_manager", lambda: mgr, raising=False)
    return TestClient(create_app())


def _chat(c, model, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    return c.post("/v1/chat/completions", json=body)


def test_card_states_the_contract(tmp_path):
    d = _write_model(tmp_path, "m", TEXT_CFG)
    c = build_model_card(d).contract
    assert c["media"]["input"] == [] and c["context"]["length"] == 8192
    assert c["tools"]["supported"] and c["tools"]["mode"] == "prompt"
    so = c["structured_output"]
    assert so["json_schema"]["engines"][0] == "in-house"
    assert so["grammar"]["engine"] in ("llguidance", None)
    assert c["speculative"]["mode"] == "none" and c["unsupported_fields"] == "400"
    assert "cache_tiers" in c and c["logprobs"]["supported"]


def test_contract_is_served_on_v1_models(served):
    item = served.get("/v1/models/plain").json()
    assert item["yunshu"]["contract"]["media"]["input"] == []
    vision = served.get("/v1/models/vision").json()["yunshu"]["contract"]
    assert "image" in vision["media"]["input"]


def test_reasoning_effort_on_non_reasoning_model_is_accepted(served):
    # Coding agents send an effort whatever model they target; it has no effect here.
    r = _chat(served, "plain", reasoning_effort="high")
    assert r.status_code != 400, r.text


def test_image_part_on_text_model_is_400(served):
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        }
    ]
    r = served.post("/v1/chat/completions", json={"model": "plain", "messages": msg})
    assert r.status_code == 400 and "image" in r.text


def test_audio_part_on_vision_model_is_400(served):
    msg = [
        {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": "QQ=="}}],
        }
    ]
    r = served.post("/v1/chat/completions", json={"model": "vision", "messages": msg})
    assert r.status_code == 400 and "audio" in r.text


def test_unknown_checkpoint_is_not_gated(tmp_path):
    card = build_model_card(tmp_path / "nowhere")
    assert capability_contract.unsupported(card, {"reasoning_effort": "high"}) == []


def test_embedding_model_rejects_text_generation_fields(tmp_path):
    d = _write_model(
        tmp_path,
        "emb",
        {**TEXT_CFG, "architectures": ["BertModel"], "model_type": "bert"},
    )
    card = build_model_card(d, model_type_name="EMBEDDING")
    reasons = capability_contract.unsupported(
        card, {"tools": [{"type": "function"}], "logprobs": True}
    )
    assert len(reasons) == 2


def test_openai_sdk_sees_the_400(served):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(
        base_url="http://testserver/v1", api_key="x", http_client=served
    )
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(
            model="plain",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        }
                    ],
                }
            ],
        )
    assert "image" in str(e.value)
