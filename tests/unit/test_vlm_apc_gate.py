"""APC must leave unsupported VLM request semantics on the existing path."""

import contextlib
from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine


def _eligible(**changes):
    engine = object.__new__(VLMEngine)
    engine._apc_backend = object()
    params = dict(
        temperature=0,
        top_p=1,
        top_k=0,
        min_p=0,
        repetition_penalty=1,
        stop=None,
        stop_token_ids=None,
        enable_thinking=False,
        logprobs=False,
        top_logprobs=None,
        kwargs={"apc_allowed": True},
    )
    params.update(changes)
    return engine._apc_text_eligible(**params)


def test_plain_greedy_text_can_use_apc():
    assert _eligible()


def test_apc_does_not_steal_unsupported_request_modes():
    assert not _eligible(kwargs={"apc_allowed": False})  # tools/media gate
    assert not _eligible(
        kwargs={"apc_allowed": True, "json_schema": {"type": "object"}}
    )
    assert not _eligible(kwargs={"apc_allowed": True, "lora_adapter": object()})
    assert not _eligible(kwargs={"apc_allowed": True, "spec_decode": True})
    assert not _eligible(temperature=0.5)
    assert not _eligible(enable_thinking=True)
    assert not _eligible(stop=["END"])
    assert not _eligible(logprobs=True)


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
