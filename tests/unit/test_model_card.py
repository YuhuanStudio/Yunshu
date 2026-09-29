"""ModelCard derivation on real checkpoints (skipped when a model is not on this machine),
plus the wire formats built from it."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from yunshu_engine.model_card import (
    API_MAX_OUTPUT_TOKENS,
    TEXT_PARAMETERS,
    build_model_card,
    normalize_effort,
    reasoning_levels,
)
from yunshu_gateway.model_card_formats import ollama_show, openai_model

MODELS = Path("/Volumes/P5Plus/models")


def _card(name: str, **kw):
    path = MODELS / name
    if not path.exists():
        pytest.skip(f"{path} not available")
    return build_model_card(path, **kw)


# ── real checkpoints ─────────────────────────────────────────────────────────


def test_qwen38_27b_card():
    c = _card("Jundot/Qwen3.8-27B-oQ4e-mtp")
    assert c.kind == "vlm" and c.family == "qwen3_5_text"
    assert c.architecture == "Qwen3_5ForConditionalGeneration"
    assert 26e9 < c.parameters < 31e9  # 27B language model + vision tower + MTP head
    assert c.quantization["bits"] == 4 and c.quantization["group_size"] == 64
    assert c.quantization["layer_groups"] == {"5": 166}  # oQ keeps 166 layers at 5 bits
    assert c.input_modalities == ["text", "image", "video"]
    assert c.output_modalities == ["text"]
    assert (
        c.context["length"] == 262144 and c.max_output_tokens == API_MAX_OUTPUT_TOKENS
    )
    r = c.reasoning
    assert r["supported"] and r["toggle"] == "enable_thinking"
    assert (
        r["effort_levels"] == ["xhigh", "medium", "low"]
        and r["default_effort"] == "xhigh"
    )
    assert r["effort_aliases"]["high"] == "xhigh"
    assert c.tools["supported"] and c.tools["parallel"]
    assert c.structured_output["json_schema"] and c.structured_output["grammar"]
    assert c.speculative["available"] and c.speculative["lossless"] is True
    assert c.speculative["method"] in ("mtp", "dflash2") and c.speculative["mtp_head"]
    assert c.prefix_cache["supported"] and c.prefix_cache["hybrid_checkpoints"]
    assert c.embeddings["dimensions"] == 5120
    assert "/v1/chat/completions" in c.api["endpoints"]
    assert {"chat_completions", "responses", "messages"} <= set(c.api["formats"])
    assert c.memory["weights_bytes"] > 15e9
    assert "reasoning_effort" in c.supported_parameters


def test_qwen35_08b_card():
    c = _card("Qwen3.5-0.8B-MLX-bf16")
    assert c.kind == "vlm" and c.quantization is None
    assert 0.7e9 < c.parameters < 1.0e9
    assert c.reasoning["supported"] and c.reasoning["effort_levels"] == []
    assert (
        "reasoning_effort" not in c.supported_parameters
    )  # template has no effort levels
    assert "enable_thinking" in c.supported_parameters
    assert c.speculative["available"] is False  # no MTP head in this checkpoint


def test_qwen25_3b_text_card():
    c = _card("Qwen2.5-3B-Instruct-4bit")
    assert c.kind == "chat" and c.family == "qwen2"
    assert c.input_modalities == ["text"] and c.output_modalities == ["text"]
    assert c.quantization == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "layer_groups": {},
    }
    assert 2.9e9 < c.parameters < 3.3e9  # quantized words are unpacked, scales skipped
    assert c.context["length"] == 32768 and c.max_output_tokens == 32768
    assert c.tools["supported"] and not c.reasoning["supported"]
    assert "enable_thinking" not in c.supported_parameters
    assert c.speculative["available"] is False


def test_whisper_card():
    c = _card("whisper-large-v3-mlx")
    assert (
        c.kind == "asr"
        and c.input_modalities == ["audio"]
        and c.output_modalities == ["text"]
    )
    assert (
        c.audio["multilingual"]
        and c.audio["translation"]
        and c.audio["audio_context_seconds"] == 30
    )
    assert "/v1/audio/translations" in c.api["endpoints"]
    assert c.supported_parameters == [] and c.max_output_tokens is None
    assert c.memory["weights_bytes"] > 1e9  # weights.npz counted


def test_qwen3_asr_card():
    c = _card("Qwen3-ASR-1.7B-bf16")
    assert c.kind == "asr" and "English" in c.audio["languages"]
    assert (
        "/v1/audio/translations" not in c.api["endpoints"]
    )  # translation needs Whisper


def test_qwen3_tts_card():
    c = _card("Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    assert (
        c.kind == "tts"
        and c.input_modalities == ["text"]
        and c.output_modalities == ["audio"]
    )
    assert c.audio["voice_design"] and "english" in c.audio["languages"]
    assert "/v1/audio/speech" in c.api["endpoints"]


def test_zimage_card():
    c = _card("Z-Image-Turbo-MLX-4bit")
    assert c.kind == "image" and c.output_modalities == ["image"]
    assert c.image["pipeline"] == "ZImagePipeline"
    assert {"transformer", "vae", "text_encoder"} <= set(c.image["components"])
    assert c.quantization["bits"] == 4 and c.quantization["skip_components"] == ["vae"]
    assert c.api["endpoints"] == ["/v1/images/generations"]


def test_glm_ocr_card():
    c = _card("GLM-OCR-bf16")
    assert (
        c.kind == "ocr"
        and c.input_modalities == ["image"]
        and c.output_modalities == ["text"]
    )
    assert c.api["endpoints"] == ["/v1/ocr"]
    assert c.tools["supported"] is False


def _fake_embedding(tmp_path: Path) -> Path:
    """Qwen3-Embedding-0.6B's published config, with a tiny safetensors header."""
    d = tmp_path / "Qwen3-Embedding-0.6B"
    (d / "1_Pooling").mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 1024,
                "max_position_embeddings": 32768,
                "num_hidden_layers": 28,
                "vocab_size": 151669,
            }
        )
    )
    (d / "1_Pooling" / "config.json").write_text(
        json.dumps({"pooling_mode_lasttoken": True})
    )
    header = json.dumps(
        {"w": {"dtype": "BF16", "shape": [1000, 1000], "data_offsets": [0, 2_000_000]}}
    ).encode()
    (d / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 2_000_000
    )
    return d


def test_qwen3_embedding_real_config():
    snaps = sorted(
        (
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"
        ).glob("*")
    )
    if not snaps:
        pytest.skip("Qwen3-Embedding-0.6B not in the HF cache")
    c = build_model_card(snaps[0], model_id="Qwen3-Embedding-0.6B")
    assert c.kind == "embedding" and c.embeddings["dimensions"] == 1024
    assert c.embeddings["pooling"] == "last" and c.context["length"] == 32768
    assert 0.5e9 < c.parameters < 0.7e9 and c.api["endpoints"] == [
        "/v1/embeddings",
        "/pooling",
    ]


def test_qwen3_embedding_card(tmp_path):
    c = build_model_card(_fake_embedding(tmp_path))
    assert c.kind == "embedding" and c.output_modalities == ["embedding"]
    assert c.embeddings == {"dimensions": 1024, "pooling": "last", "normalized": True}
    assert c.parameters == 1_000_000 and c.context["length"] == 32768
    assert c.api["endpoints"] == ["/v1/embeddings", "/pooling"]
    assert c.supported_parameters == [] and not c.reasoning["supported"]
    assert openai_model(c)["capabilities"]["embedding"] is True
    assert openai_model(c)["task"] == "embed"


def test_missing_checkpoint_still_yields_a_card():
    c = build_model_card("/nonexistent/some-model", model_id="some-model")
    assert c.id == "some-model" and c.created > 0 and c.state["status"] == "not-loaded"


def test_load_state_is_per_call():
    c = _card("Qwen2.5-3B-Instruct-4bit", loaded=True, pinned=True, estimated_bytes=123)
    assert (
        c.state["loaded"] and c.state["pinned"] and c.memory["estimated_bytes"] == 123
    )
    c2 = _card("Qwen2.5-3B-Instruct-4bit")
    assert not c2.state["loaded"]


# ── template + parameter contracts ───────────────────────────────────────────


def test_effort_normalization():
    lv = ["xhigh", "medium", "low"]
    assert normalize_effort("high", lv) == "xhigh"
    assert normalize_effort("medium", lv) == "medium"
    assert normalize_effort("minimal", lv) == "low"
    assert normalize_effort("high", ["low", "medium", "high"]) == "high"
    assert normalize_effort("high", []) == "high"


def test_reasoning_levels_parse():
    tpl = "{%- set reasoning_effort = reasoning_effort|default('xhigh') %}{% if reasoning_effort not in ('xhigh', 'medium', 'low') %}"
    assert reasoning_levels(tpl) == (["xhigh", "medium", "low"], "xhigh")
    assert reasoning_levels("no effort here") == ([], None)


def test_supported_parameters_are_real_chat_request_fields():
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    fields = set(ChatCompletionRequest.model_fields)
    assert set(TEXT_PARAMETERS) <= fields, set(TEXT_PARAMETERS) - fields
    assert (
        ChatCompletionRequest.model_fields["max_tokens"].metadata[-1].le
        == API_MAX_OUTPUT_TOKENS
    )


# ── wire formats ─────────────────────────────────────────────────────────────


def test_openai_payload_has_every_dialect():
    c = _card("Jundot/Qwen3.8-27B-oQ4e-mtp")
    m = openai_model(c)
    # OpenAI
    assert (
        m["object"] == "model"
        and m["owned_by"] == "yunshu"
        and isinstance(m["created"], int)
    )
    # Anthropic
    assert m["type"] == "model" and m["display_name"] and m["created_at"].endswith("Z")
    assert m["max_input_tokens"] == 262144 and m["max_tokens"] == API_MAX_OUTPUT_TOKENS
    caps = m["capabilities"]
    assert caps["image_input"]["supported"] and caps["thinking"]["supported"]
    assert caps["effort"]["low"]["supported"] and caps["effort"]["xhigh"]["supported"]
    assert (
        caps["effort"]["high"]["supported"] and not caps["effort"]["max"]["supported"]
    )
    # OpenRouter
    assert m["context_length"] == 262144
    assert m["architecture"]["input_modalities"] == ["text", "image", "video"]
    assert m["architecture"]["modality"] == "text+image+video->text"
    assert m["top_provider"]["max_completion_tokens"] == API_MAX_OUTPUT_TOKENS
    assert (
        "tools" in m["supported_parameters"]
        and "structured_outputs" in m["supported_parameters"]
    )
    assert "reasoning" in m["supported_parameters"]
    # vLLM / LM Studio (the keys Yunxin's adapters read)
    assert m["max_model_len"] == 262144 and m["max_context_length"] == 262144
    assert m["task"] == "generate" and m["model_type"] == "vlm"
    assert (
        caps["vision"] is True
        and caps["trained_for_tool_use"] is True
        and caps["reasoning"] is True
    )
    assert m["quantization"] == "mixed-4/5bit" and m["state"] == "not-loaded"
    # the full card, without the filesystem path
    assert m["yunshu"]["kind"] == "vlm" and "path" not in m["yunshu"]
    assert "chat_completions" in m["yunshu"]["api"]["formats"]
    json.dumps(m)


def test_path_only_when_detailed():
    c = _card("Qwen2.5-3B-Instruct-4bit")
    assert "path" not in openai_model(c)["yunshu"]
    assert openai_model(c, detailed=True)["yunshu"]["path"].endswith(
        "Qwen2.5-3B-Instruct-4bit"
    )


def test_ollama_show_capabilities():
    show = ollama_show(_card("Jundot/Qwen3.8-27B-oQ4e-mtp"))
    assert set(show["capabilities"]) == {"completion", "tools", "vision", "thinking"}
    assert show["model_info"]["general.architecture"] == "qwen3_5_text"
    assert show["model_info"]["qwen3_5_text.context_length"] == 262144
    assert show["details"]["parameter_size"].endswith("B")
    # the loopback form (wire dict) gives the same answer
    assert (
        ollama_show(openai_model(_card("Jundot/Qwen3.8-27B-oQ4e-mtp"))["yunshu"])
        == show
    )
    assert ollama_show(_card("GLM-OCR-bf16"))["capabilities"] == ["ocr"]
