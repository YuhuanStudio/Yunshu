"""Qwen4 family detection and patch targets follow QWEN4_MODEL_TYPES (a renamed release needs one line)."""

import re
import sys
import types
from pathlib import Path

import pytest

from yunshu_engine import qwen4_family as fam


@pytest.fixture
def renamed(monkeypatch):
    monkeypatch.setattr(fam, "QWEN4_MODEL_TYPES", ("qwen4_exp", "qwen4_moe"))


def test_preview_and_unknown():
    assert fam.is_qwen4_family({"model_type": "qwen4_exp"})
    assert not fam.is_qwen4_family({"model_type": "qwen4_moe"})
    assert not fam.is_qwen4_family({"model_type": "qwen3_5"})
    assert not fam.is_qwen4_family({})
    assert not fam.is_qwen4_family(None)


def test_text_config_type_accepted(renamed):
    assert fam.is_qwen4_family(
        {"model_type": "x", "text_config": {"model_type": "qwen4_moe"}}
    )
    assert not fam.is_qwen4_family(
        {"model_type": "x", "text_config": {"model_type": "y"}}
    )


def _pack(tmp_path, model_type):
    import json

    from yunshu_engine.ple_ssd import PLE_MARKER

    (tmp_path / "shard.safetensors").write_bytes(b"x")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.fc.weight": "shard.safetensors",
                    f"a{PLE_MARKER}b": "shard.safetensors",
                }
            }
        )
    )
    config = {
        "model_type": model_type,
        "text_config": {"mtp_num_hidden_layers": 1},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return config


def test_renamed_engages_mtp_and_ple_gates(renamed, tmp_path, monkeypatch):
    from yunshu_engine import mlxvlm_mtp, ple_ssd

    monkeypatch.setattr(ple_ssd, "manifest_for", lambda p: tmp_path / "m.json")
    config = _pack(tmp_path, "qwen4_moe")
    assert mlxvlm_mtp.is_mtp_capable(str(tmp_path))
    assert ple_ssd.external_ple(config, tmp_path) is True
    assert fam.is_qwen4_module_name("mlx_vlm.models.qwen4_moe.language")
    assert not fam.is_qwen4_module_name("mlx_vlm.models.qwen5.language")


def test_unknown_type_engages_no_gate(tmp_path, monkeypatch):
    from yunshu_engine import mlxvlm_mtp, ple_ssd

    monkeypatch.setattr(ple_ssd, "manifest_for", lambda p: tmp_path / "m.json")
    config = _pack(tmp_path, "qwen4_moe")
    assert not mlxvlm_mtp.is_mtp_capable(str(tmp_path))
    assert ple_ssd.external_ple(config, tmp_path) is False


def test_preview_type_still_engages(tmp_path, monkeypatch):
    from yunshu_engine import mlxvlm_mtp

    _pack(tmp_path, "qwen4_exp")
    assert mlxvlm_mtp.is_mtp_capable(str(tmp_path))


def test_spec_model_types_include_family():
    import yunshu_engine.vlm_engine as ve

    assert set(fam.QWEN4_MODEL_TYPES) <= set(ve.VLMEngine._SPEC_MODEL_TYPES)


def test_modules_apply_to_present_skip_absent(renamed, monkeypatch):
    present = types.ModuleType("mlx_vlm.models.qwen4_moe.language")
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen4_moe.language", present)
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.qwen4_exp.language", None
    )  # absent: ImportError
    assert fam.qwen4_model_modules("language") == [present]
    assert fam.qwen4_first_module("language") is present
    assert fam.loaded_qwen4_modules("language") == [present]
    other = types.ModuleType("mlx_vlm.models.qwen4_exp.language")
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen4_exp.language", other)
    assert fam.qwen4_model_modules("language") == [other, present]
    assert fam.qwen4_model_modules("nonexistent_sub_xyz") == []


def test_no_new_literal_qwen4_exp_in_code():
    root = Path(__file__).resolve().parents[2] / "python"
    pat = re.compile(r"""["']qwen4_exp["']|models\.qwen4_exp""")
    bad = []
    for path in root.rglob("*.py"):
        if path.name == "qwen4_family.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line):
                bad.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not bad, (
        "use yunshu_engine.qwen4_family instead of a literal:\n" + "\n".join(bad)
    )


def test_text_sub_config_type_is_qwen4():
    """2026-11-08 M5 yv: the real Flash-Next checkpoint's language model reports args.model_type
    'qwen4_exp_text' (config.json text_config.model_type); the driver routed it to the generic layer and
    decoded without speculation. The text sub-config type must count as Qwen4."""
    from yunshu_engine.qwen4_family import is_qwen4_type, is_qwen4_family

    assert is_qwen4_type("qwen4_exp_text")
    assert is_qwen4_type("qwen4_exp")
    assert not is_qwen4_type("qwen3_5_text")
    assert not is_qwen4_type("qwen4_exp_textual")
    assert is_qwen4_family({"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}})
    assert is_qwen4_family({"model_type": "other", "text_config": {"model_type": "qwen4_exp_text"}})
