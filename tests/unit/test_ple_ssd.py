"""Flash-Next PLE table served from the checkpoint files (ple_ssd): config rewrite, skipped tensors and
the load-time memory plan, on a fake pack (headers only; no MLX)."""

import json
import struct

import pytest

from yunshu_engine import memory_admission, ple_ssd

PLE = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"
DENSE = "language_model.model.layers.1.mlp.gate_proj.weight"


def _shard(path, tensors):
    header, offset = {}, 0
    for name, nbytes in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [nbytes],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


@pytest.fixture
def pack(tmp_path):
    _shard(tmp_path / "model-00001-of-00002.safetensors", {DENSE: 1000})
    _shard(tmp_path / "model-00002-of-00002.safetensors", {PLE: 5000})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    DENSE: "model-00001-of-00002.safetensors",
                    PLE: "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"hidden_size": 8},
        "quantization": {
            "bits": 2,
            "group_size": 64,
            PLE.rsplit(".", 1)[0]: {"bits": 4},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, config


@pytest.fixture
def no_build(monkeypatch, tmp_path):
    built = []

    def fake(model_path):
        built.append(model_path)
        return tmp_path / "ple-store.json"

    monkeypatch.setattr(ple_ssd, "manifest_for", fake)
    return built


def test_default_reads_the_table_from_the_files(pack, no_build, monkeypatch):
    monkeypatch.delenv("YUNSHU_PLE_STORAGE", raising=False)
    path, config = pack
    assert ple_ssd.external_ple(config, path) is True
    assert config["text_config"]["ple_storage"]["manifest"].endswith("ple-store.json")
    assert not any(ple_ssd.PLE_MARKER in k for k in config["quantization"])
    assert config["quantization"]["bits"] == 2
    assert ple_ssd.is_ple_tensor(PLE) and not ple_ssd.is_ple_tensor(DENSE)
    assert no_build == [path]


def test_memory_mode_and_prepared_views_are_left_alone(pack, no_build, monkeypatch):
    path, config = pack
    monkeypatch.setenv("YUNSHU_PLE_STORAGE", "memory")
    assert ple_ssd.external_ple(config, path) is False
    monkeypatch.delenv("YUNSHU_PLE_STORAGE")
    view = dict(config, text_config={"ple_storage": {"manifest": "ple-store.json"}})
    assert ple_ssd.external_ple(view, path) is False
    other = dict(config, model_type="qwen3_5")
    assert ple_ssd.external_ple(other, path) is False
    assert no_build == []


def test_memory_plan_does_not_count_the_table(pack, monkeypatch):
    path, _ = pack
    monkeypatch.delenv("YUNSHU_PLE_STORAGE", raising=False)
    on_disk = sum(p.stat().st_size for p in path.glob("*.safetensors"))
    assert ple_ssd.bytes_read_from_files(path) == 5000
    assert memory_admission.weights_bytes_on_disk(path) == on_disk - 5000
    monkeypatch.setenv("YUNSHU_PLE_STORAGE", "memory")
    assert memory_admission.weights_bytes_on_disk(path) == on_disk


def test_scale_tensors_are_recognised():
    scale = (
        "language_model.model.layers.3.ple.ple_embedding.ngram_embedding.weight_scale"
    )
    assert ple_ssd.is_ple_scale(scale) and not ple_ssd.is_ple_tensor(scale)
    assert not ple_ssd.is_ple_scale(PLE)
