"""CPU fixtures reject incomplete checkpoints before a queued GPU load."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "bigmoe_census", Path(__file__).parents[2] / "scripts/research/bigmoe_census.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def checkpoint(tmp_path):
    header = json.dumps(
        {
            "model.weight": {"data_offsets": [0, 4]},
            "mtp.weight": {"data_offsets": [4, 8]},
        }
    ).encode()
    (tmp_path / "a.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"12345678"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.weight": "a.safetensors",
                    "mtp.weight": "a.safetensors",
                }
            }
        )
    )
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}')
    return tmp_path


def test_accounting(tmp_path):
    result = module.census(checkpoint(tmp_path))
    assert result["base_bytes"] == result["mtp_bytes"] == 4
    assert result["complete"]
    assert result["budget"]["cache_headroom_gib"] == 80 - 4 / 2**30
    assert module.census(tmp_path, 100)["admission"] == "reject"


def test_missing_shard(tmp_path):
    root = checkpoint(tmp_path)
    (root / "a.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        module.census(root)


def test_index_mismatch(tmp_path):
    root = checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"wrong":"a.safetensors"}}'
    )
    with pytest.raises(ValueError, match="mismatch"):
        module.census(root)
