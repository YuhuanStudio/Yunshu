"""MTP dispatch must be based on weights, not the model config alone."""

import json

import mlx.core as mx
import pytest

from yunshu_engine.mlxvlm_mtp import (
    _build_drafter,
    _load_drafter_in_memory,
    is_mtp_capable,
)


def test_mtp_detection_requires_indexed_existing_head(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    assert not is_mtp_capable(str(tmp_path))

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"weight_map": {"language_model.layers.0.weight": "model.safetensors"}}
        )
    )
    assert not is_mtp_capable(str(tmp_path))

    index.write_text(
        json.dumps(
            {"weight_map": {"language_model.mtp.fc.weight": "missing.safetensors"}}
        )
    )
    assert not is_mtp_capable(str(tmp_path))

    (tmp_path / "missing.safetensors").touch()
    assert is_mtp_capable(str(tmp_path))


def test_mtp_detection_rejects_unreadable_index(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text("{")
    assert not is_mtp_capable(str(tmp_path))


def test_drafter_split_reads_indexed_head_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    mx.save_safetensors(
        str(source / "shard.safetensors"),
        {
            "language_model.mtp.fc.weight": mx.ones((2, 2)),
            "language_model.layers.0.weight": mx.ones((2, 2)) * 9,
        },
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.mtp.fc.weight": "shard.safetensors",
                    "language_model.layers.0.weight": "shard.safetensors",
                }
            }
        )
    )
    output = tmp_path / "drafter"
    assert _build_drafter(str(source), str(output)) == str(output)
    tensors = mx.load(str(output / "model.safetensors"))
    assert set(tensors) == {"fc.weight"}
    assert tensors["fc.weight"].shape == (2, 2)


def test_unindexed_head_warns_in_doctor_and_startup(caplog, monkeypatch, tmp_path):
    from yunshu_cli.doctor import check_speculative
    from yunshu_engine.mlxvlm_mtp import unindexed_mtp_warning

    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"base.weight": "model.safetensors"}})
    )
    (tmp_path / "mtp-weights.safetensors").touch()
    warning = unindexed_mtp_warning(str(tmp_path))
    assert "draft=off" in warning
    checks = check_speculative(str(tmp_path))
    assert checks[0].status == "warn"
    assert "mtp-weights.safetensors" in checks[0].detail
    # Stop at drafter selection: verify the real startup log without model/GPU work.
    from types import SimpleNamespace

    import pytest

    from yunshu_engine import spec_select
    from yunshu_engine.vlm_engine import VLMEngine

    class SelectionReachedError(Exception):
        pass

    def stop_at_selection(*_a, **_kw):
        raise SelectionReachedError

    engine = object.__new__(VLMEngine)
    engine._config = json.loads((tmp_path / "config.json").read_text())
    engine._model = SimpleNamespace(language_model=object())
    # This test exercises speculative startup selection, not VM admission.
    monkeypatch.setattr(
        "yunshu_engine.utils.hardware.is_paravirtual_metal", lambda: False
    )
    monkeypatch.setattr(VLMEngine, "_round_driver_wanted", lambda *_: False)
    monkeypatch.setattr(spec_select, "choose", stop_at_selection)
    with pytest.raises(SelectionReachedError):
        engine._build_batch_runner(str(tmp_path))
    assert warning in caplog.text


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen4_exp"])
def test_doctor_reports_speculation_for_every_engine_spec_family(
    monkeypatch, tmp_path, model_type
):
    # doctor must use the engine's own family list: Flash-Next (qwen4_exp) was reported as "none"
    from yunshu_cli.doctor import check_speculative
    from yunshu_engine import mlxvlm_mtp, spec_select

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "text_config": {"hidden_size": 2560, "num_hidden_layers": 48},
            }
        )
    )
    monkeypatch.setattr(spec_select, "find_dflash_drafter", lambda *a, **k: None)
    monkeypatch.setattr(mlxvlm_mtp, "is_mtp_capable", lambda _p: True)
    monkeypatch.setattr(mlxvlm_mtp, "unindexed_mtp_warning", lambda _p: None)
    checks = check_speculative(str(tmp_path))
    assert checks and checks[0].detail.startswith("MTP head"), checks


def test_moe_checkpoint_with_indexed_head_is_mtp_capable(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5_moe", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    (tmp_path / "shard.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": "shard.safetensors"}})
    )
    assert is_mtp_capable(str(tmp_path))


# Same suffixes upstream Qwen3_5MTPDraftModel.sanitize shifts by +1 (HF zero-centred norms).
NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def test_moe_head_loads_hf_per_expert_layout(tmp_path):
    # HF stores one tensor per expert and zero-centred norms; the in-memory loader must
    # stack the experts and shift the norms back so every drafter parameter round-trips.
    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )

    text = dict(
        model_type="qwen3_5_moe_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=4096,
        mtp_num_hidden_layers=1,
        tie_word_embeddings=False,
    )
    ref = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig.from_dict(
            {"model_type": "qwen3_5_mtp", "text_config": text, "block_size": 3}
        )
    )
    mx.eval(ref.parameters())
    hf = {}
    for name, value in tree_flatten(ref.parameters()):
        if ".switch_mlp." in name:
            base, rest = name.split(".switch_mlp.")
            for e in range(text["num_experts"]):
                hf[f"mtp.{base}.experts.{e}.{rest}"] = value[e]
        elif name.endswith(NORM_SUFFIXES):
            hf[f"mtp.{name}"] = value - 1.0
        else:
            hf[f"mtp.{name}"] = value
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), hf)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "mtp.safetensors" for k in hf}})
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "text_config": text})
    )

    loaded = _load_drafter_in_memory(str(tmp_path))
    got = dict(tree_flatten(loaded.parameters()))
    want = dict(tree_flatten(ref.parameters()))
    assert set(got) == set(want)
    for name in want:
        assert mx.allclose(got[name], want[name]).item(), name
