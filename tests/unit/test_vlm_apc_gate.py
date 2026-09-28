"""Runner routing and the Qwen APC capacity gate."""

import contextlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine.vlm_engine import VLMEngine


def _engine(runner=True, vision=True, processor=True):
    engine = object.__new__(VLMEngine)
    engine._batch_runner = object() if runner else None
    engine._has_vision = vision
    engine._processor = object() if processor else None
    engine._model_path = "/models/m"
    return engine


def test_runner_serves_every_request_knob():
    engine = _engine()
    for kwargs in (
        {},
        {"json_schema": {"type": "object"}},
        {"reasoning_effort": "low"},
        {"xtc_probability": 0.5},
        {"min_tokens": 3},
        {"ignore_eos": True},
        {"suppress_tokens": [1]},
        {"top_n_sigma": 1.0},
        {"spec_decode": True},
    ):
        engine._check_request_supported([], [], kwargs)


def test_unimplemented_knobs_are_rejected_not_ignored():
    engine = _engine()
    for key in ("lora_adapter", "logits_processors"):
        with pytest.raises(ValueError, match=key):
            engine._check_request_supported([], [], {key: 1})


def test_media_requirements_fail_loudly():
    with pytest.raises(RuntimeError, match="no batch runner"):
        _engine(runner=False)._check_request_supported([], [], {})
    with pytest.raises(ValueError, match="no vision encoder"):
        _engine(vision=False)._check_request_supported(["a.png"], [], {})
    with pytest.raises(RuntimeError, match="no mlx_vlm processor"):
        _engine(processor=False)._check_request_supported([], ["a.wav"], {})
    # Audio-only models (no vision tower) still take audio.
    _engine(vision=False)._check_request_supported([], ["a.wav"], {})


def test_qwen_capacity_gate_avoids_uncacheable_32k_prefill():
    engine = object.__new__(VLMEngine)
    engine._config = {
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_size": 5120,
        }
    }
    engine._apc_semantic_hash = 123
    engine._apc_backend = SimpleNamespace(
        memory_max_bytes=int(1.5 * 2**30),
        lock=contextlib.nullcontext(),
        _exact_cache={},
    )
    assert engine._apc_capacity_allows(mx.arange(8426))
    assert not engine._apc_capacity_allows(mx.arange(32226))
    engine._apc_backend.memory_max_bytes = 6 * 2**30
    assert engine._apc_capacity_allows(mx.arange(32226))


def test_qwen_capacity_gate_preserves_existing_partial_prefix():
    engine = object.__new__(VLMEngine)
    engine._config = {
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_size": 5120,
        }
    }
    engine._apc_semantic_hash = 123
    entry = SimpleNamespace(extra_hash=123, token_ids=tuple(range(1024)))
    engine._apc_backend = SimpleNamespace(
        memory_max_bytes=int(1.5 * 2**30),
        lock=contextlib.nullcontext(),
        _exact_cache={1: entry},
    )
    assert engine._apc_capacity_allows(mx.arange(32226))
