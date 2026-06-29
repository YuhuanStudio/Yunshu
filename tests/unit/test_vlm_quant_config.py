"""(MED): the VLM loader read only config["quantization"], so a checkpoint that
ships an HF-style `quantization_config` block (mxfp4 / compressed-tensors VLMs, possibly
nested in text_config) loaded UN-quantized → load_weights shape-mismatch → silent text-only
fallback (vision lost) or a Metal fault. _derive_vlm_quantization now mirrors mlx_vlm's
derivation. (The text engine path was fine — it delegates to mlx-lm, which derives upstream;
this was a VLM-specific gap, the "claimed-mirrors-mlx_vlm but incomplete on a sibling" class.)
"""

from __future__ import annotations

import inspect

from yunshu_engine import vlm_engine
from yunshu_engine.vlm_engine import _derive_vlm_quantization


def test_top_level_quantization_passthrough():
    q = {"group_size": 64, "bits": 4, "mode": "affine"}
    assert _derive_vlm_quantization({"quantization": q}) is q


def test_mxfp4_quantization_config_derived():
    cfg = {"quantization_config": {"quant_method": "mxfp4"}}
    assert _derive_vlm_quantization(cfg) == {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }


def test_compressed_tensors_derived():
    cfg = {"quantization_config": {"quant_method": "compressed-tensors"}}
    assert _derive_vlm_quantization(cfg) == {
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
    }


def test_nested_text_config_quantization_config():
    cfg = {"text_config": {"quantization_config": {"quant_method": "mxfp4"}}}
    assert _derive_vlm_quantization(cfg) == {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }


def test_unsupported_method_returns_none():
    for m in ("awq", "gptq", "bitnet"):
        assert (
            _derive_vlm_quantization({"quantization_config": {"quant_method": m}})
            is None
        )


def test_unquantized_returns_none():
    assert _derive_vlm_quantization({}) is None
    assert _derive_vlm_quantization({"text_config": {}}) is None


def test_loader_uses_derived_quantization():
    src = inspect.getsource(vlm_engine.VLMEngine._load_vision_model)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_derive_vlm_quantization(config)" in code
    assert 'config.get("quantization")' not in code  # the old narrow read is gone
    assert 'quantization.get("bits", 4)' in code  # no hard KeyError subscript
