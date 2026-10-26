"""Behaviour of engine_diagnostics: activity flag, model-id matching and KV sizing from a model config."""

from types import SimpleNamespace

import pytest

from yunshu_engine.engine_diagnostics import EngineDiagnosticsMixin as D


def test_has_active_requests_counts_fast_path_and_core():
    core = SimpleNamespace(has_active_requests=False)
    eng = SimpleNamespace(_active_fast_path_count=0, _engine_core=core)
    assert D.has_active_requests(eng) is False
    core.has_active_requests = True
    assert D.has_active_requests(eng) is True
    eng = SimpleNamespace(_active_fast_path_count=2, _engine_core=None)
    assert D.has_active_requests(eng) is True
    assert D.has_active_requests(SimpleNamespace(_engine_core=None)) is False


@pytest.mark.parametrize(
    "name,model_id,expected",
    [
        ("org/Qwen3-8B", "Qwen3-8B", True),
        ("org/Qwen3-8B", "org/Qwen3-8B", True),
        ("org/Qwen3-8B", "qwen3-8b", True),  # case-insensitive
        ("org/Qwen3-8B", "other/Qwen3-8B", True),  # repo prefix is ignored
        ("org/Qwen3-8B", "Qwen3-4B", False),
        ("/models/local-dir", "local-dir", True),
        ("", "anything", False),
        (None, "anything", False),
    ],
)
def test_resolve_model_id(name, model_id, expected):
    assert D.resolve_model_id(SimpleNamespace(model_name=name), model_id) is expected


def _cfg(**kw):
    return SimpleNamespace(**kw)


def test_extract_model_arch_reads_standard_hf_config():
    model = SimpleNamespace(
        config=_cfg(
            num_hidden_layers=32,
            num_key_value_heads=8,
            num_attention_heads=32,
            hidden_size=4096,
        )
    )
    assert D._extract_model_arch(model) == {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,  # hidden_size // attention heads
    }


def test_extract_model_arch_explicit_head_dim_and_mha_fallback():
    model = SimpleNamespace(
        args=_cfg(n_layers=4, n_heads=16, head_dim=64),
    )
    assert D._extract_model_arch(model) == {
        "num_layers": 4,
        "num_kv_heads": 16,  # no kv heads -> MHA, equal to attention heads
        "head_dim": 64,
    }


def test_extract_model_arch_counts_layers_on_the_model_when_config_lacks_them():
    model = SimpleNamespace(
        config=_cfg(num_key_value_heads=2, head_dim=32),
        layers=[object()] * 6,
    )
    assert D._extract_model_arch(model)["num_layers"] == 6


def test_extract_model_arch_hybrid_counts_only_full_attention_layers():
    types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ] * 4
    by_types = SimpleNamespace(
        config=_cfg(
            num_hidden_layers=16,
            layer_types=types,
            num_key_value_heads=4,
            head_dim=128,
        )
    )
    assert D._extract_model_arch(by_types)["num_layers"] == 4
    by_interval = SimpleNamespace(
        config=_cfg(
            num_hidden_layers=16,
            full_attention_interval=4,
            num_key_value_heads=4,
            head_dim=128,
        )
    )
    assert D._extract_model_arch(by_interval)["num_layers"] == 4


@pytest.mark.parametrize(
    "model",
    [None, SimpleNamespace(), SimpleNamespace(config=_cfg(num_hidden_layers=2))],
)
def test_extract_model_arch_returns_empty_when_incomplete(model):
    assert D._extract_model_arch(model) == {}
