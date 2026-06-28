"""setting YUNSHU_QUANT_CONFIG made model load fail entirely. The loader did
kwargs["quantization"] = <str> and passed it to mlx_lm.utils.load, which has NO
`quantization` parameter (path/tokenizer_config/model_config/adapter_path/lazy/
return_config/revision) → TypeError → the engine's start() crashed (the except only caught
ValueError). It was even marked ✅-fixed in an audit doc. Now the env is parsed into a
quantization dict and passed via the valid `model_config` mechanism, with a defensive
TypeError guard that retries a clean load."""
from __future__ import annotations

import inspect

from yunshu_engine.batched_engine import _parse_quant_config_env


def test_parse_json_dict():
    assert _parse_quant_config_env('{"bits":4,"group_size":32}') == {"bits": 4, "group_size": 32}


def test_parse_bare_int():
    assert _parse_quant_config_env("8") == {"bits": 8, "group_size": 64}


def test_parse_compact_forms():
    assert _parse_quant_config_env("4,64") == {"bits": 4, "group_size": 64}
    assert _parse_quant_config_env("4:128") == {"bits": 4, "group_size": 128}


def test_parse_unparseable_returns_none():
    for bad in ("", "garbage", "true", "false", "[]", "{not json"):
        assert _parse_quant_config_env(bad) is None


def test_load_passes_model_config_not_quantization_kwarg():
    # the loader must use the valid mlx-lm `model_config` mechanism, never the bogus
    # `quantization=` kwarg that caused the TypeError, and must guard TypeError.
    import yunshu_engine.batched_engine as _m
    src = inspect.getsource(_m)
    assert 'kwargs["quantization"] = qconfig' not in src
    assert '"quantization": _qd' in src
    assert "except TypeError as e:" in src


def test_mlx_lm_load_has_no_quantization_param():
    # pin the contract this fix depends on: load() accepts model_config, not quantization
    from mlx_lm.utils import load
    params = set(inspect.signature(load).parameters)
    assert "model_config" in params
    assert "quantization" not in params
