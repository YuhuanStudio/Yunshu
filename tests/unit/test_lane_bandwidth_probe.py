"""CPU checks for checkpoint inventory used before GPU admission."""

import importlib.util
import json
import struct
from pathlib import Path

PROBE = Path(__file__).parents[2] / "scripts/research/probe_lane_bandwidth.py"
spec = importlib.util.spec_from_file_location("bandwidth_probe", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_inventory_reads_mixed_bits_and_excludes_mtp(tmp_path):
    meta = {}
    for layer, bits in [(0, 4), (1, 5)]:
        prefix = f"language_model.model.layers.{layer}.mlp.down_proj"
        meta[prefix + ".weight"] = {"shape": [5120, 17408 * bits // 32]}
        meta[prefix + ".scales"] = {"shape": [5120, 272]}
    meta["language_model.mtp.layers.0.mlp.up_proj.weight"] = {"shape": [17408, 640]}
    raw = json.dumps(meta).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    got = probe.inventory(tmp_path)
    assert set(got) == {
        ("mlp.down_proj", 5120, 17408, 4),
        ("mlp.down_proj", 5120, 17408, 5),
    }
    assert all(len(members) == 1 for members in got.values())


def test_draft_fc_and_projection_names(tmp_path):
    meta = {
        k: {"shape": v}
        for k, v in [
            ("fc.weight", [5120, 25600]),
            ("layers.0.self_attn.q_proj.weight", [4096, 5120]),
            ("hidden_norm.weight", [5120]),
        ]
    }
    raw = json.dumps(meta).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert set(probe.inventory(tmp_path, True)) == {
        ("fc", 5120, 25600, 8),
        ("self_attn.q_proj", 4096, 5120, 8),
    }


def test_direct8_transform_preserves_arithmetic_and_eliminates_stage():
    import ast

    tree = ast.parse(
        (
            PROBE.parent.parent.parent
            / "python/yunshu_engine/kernels/tensorfold/lane_widen_m5.py"
        ).read_text()
    )
    # Loading the source module executes only string transforms, no MLX import.
    namespace = {}
    exec(compile(tree, "lane_widen_source", "exec"), namespace)
    module_spec = importlib.util.spec_from_file_location(
        "direct8", PROBE.parent / "lane_direct8.py"
    )
    direct = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(direct)
    before = namespace["BYTES"]
    after = direct.transform(before)
    assert "stage[" not in after
    assert "simdgroup_barrier" not in after
    assert "if (TILED) op.run(a, bt, P); else op.run(a, b, P);" in after
    assert (
        after[after.index("    for (int t = 0; t < TMR; t++) {") :]
        == before[before.index("    for (int t = 0; t < TMR; t++) {") :]
    )


def test_draft_inventory_includes_serving_fusions(tmp_path):
    meta = {}
    for name in (
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
    ):
        n = 1024 if name.startswith("self_attn") else 17408
        meta[f"layers.0.{name}.weight"] = {"shape": [n, 5120]}
    raw = json.dumps(meta).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    got = probe.inventory(tmp_path, True)
    assert len(got[("self_attn.kv", 2048, 5120, 8)]) == 1
    assert (
        got[("mlp.gate_up", 34816, 5120, 8)][0][1]
        == "layers.0.mlp.gate_proj|layers.0.mlp.up_proj"
    )
