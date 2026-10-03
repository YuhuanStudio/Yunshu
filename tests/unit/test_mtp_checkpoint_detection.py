"""MTP dispatch must be based on weights, not the model config alone."""

import json

import mlx.core as mx

from yunshu_engine.mlxvlm_mtp import _build_drafter, is_mtp_capable


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
    monkeypatch.setattr(VLMEngine, "_round_driver_wanted", lambda *_: False)
    monkeypatch.setattr(spec_select, "choose", stop_at_selection)
    with pytest.raises(SelectionReachedError):
        engine._build_batch_runner(str(tmp_path))
    assert warning in caplog.text
