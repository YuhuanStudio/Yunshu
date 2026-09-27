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
