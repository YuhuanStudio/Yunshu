"""Runner routing and the Qwen APC capacity gate."""

import contextlib
from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine


def _runner_eligible(**kwargs):
    engine = object.__new__(VLMEngine)
    engine._batch_runner = object()
    return engine._runner_text_eligible(
        logprobs=False, top_logprobs=None, kwargs=kwargs
    )


def test_runner_serves_ordinary_requests():
    # Sampling, thinking, stop, schema, tools and logprobs all stay on the runner.
    assert _runner_eligible()
    assert _runner_eligible(json_schema={"type": "object"})
    assert _runner_eligible(reasoning_effort="low")


def test_runner_leaves_unimplemented_knobs_to_legacy_loop():
    for key in ("lora_adapter", "logits_processors"):
        assert not _runner_eligible(**{key: 1})
    # Implemented by the runner (TokenMaskProcessor / RowSampler) or accepted.
    for key in (
        "xtc_probability",
        "min_tokens",
        "ignore_eos",
        "suppress_tokens",
        "top_n_sigma",
        "spec_decode",
    ):
        assert _runner_eligible(**{key: 1})
    engine = object.__new__(VLMEngine)
    engine._batch_runner = None
    assert not engine._runner_text_eligible(
        logprobs=False, top_logprobs=None, kwargs={}
    )


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
