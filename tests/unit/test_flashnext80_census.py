import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_census as fc  # noqa: E402


def _write(path, tensors):
    header, off = {}, 0
    for name, n in tensors.items():
        header[name] = {"dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def test_census_buckets(tmp_path):
    names = {
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight": 100,
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": 50,
        "mtp.layers.0.self_attn.q_proj.weight": 7,
        "vision_tower.blocks.0.weight": 3,
        "language_model.model.layers.0.self_attn.q_proj.weight": 5,
    }
    _write(tmp_path / "model-1.safetensors", names)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj": {
                        "bits": 4,
                        "group_size": 64,
                    }
                }
            }
        )
    )
    r = fc.census(tmp_path)
    c = r["components_gb"]
    assert c["ple"] == 100 / 1e9 and c["routed_experts"] == 50 / 1e9
    assert c["mtp"] == 7 / 1e9 and c["vision"] == 3 / 1e9 and c["self_attn"] == 5 / 1e9
    assert r["resident_without_ple_gb"] == 65 / 1e9
    assert r["expert_projection_quant"] == {"4b/g64": 1}
