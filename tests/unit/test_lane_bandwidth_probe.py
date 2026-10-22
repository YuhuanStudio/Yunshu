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
            ("candidate_selector.hidden_projection.weight", [256, 5120]),
            ("layers.0.self_attn.q_proj.weight", [4096, 5120]),
            ("hidden_norm.weight", [5120]),
        ]
    }
    raw = json.dumps(meta).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert set(probe.inventory(tmp_path, True)) == {
        ("fc", 5120, 25600, 8),
        ("candidate_selector.hidden_projection", 256, 5120, 8),
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


def test_half_tile_only_changes_weight_address():
    module_spec = importlib.util.spec_from_file_location(
        "half_tile", PROBE.parent / "lane_half_tile.py"
    )
    half = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(half)
    needle = "(int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2)"
    source = "before\n" + needle + "\nafter"
    new = half.transform(source)
    assert new.startswith("before\n") and new.endswith("\nafter")
    for group in range(80):
        for tile in range(10):
            offset = ((tile // 2) * 80 + group) * (32 * 64 // 2) + (tile % 2) * (
                16 * 64 // 2
            )
            for column in range(16):
                n = tile * 16 + column
                assert (
                    offset + column * 32
                    == ((n // 32) * 80 + group) * 1024 + (n % 32) * 32
                )


def test_half_tile_keeps_split_k_and_can_pair_rows(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    module_spec = importlib.util.spec_from_file_location(
        "half_route", PROBE.parent / "lane_half_tile.py"
    )
    half = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(half)

    def base(*args, **kwargs):
        return kwargs

    kernel = object()
    q = SimpleNamespace(
        _resolve_variant=lambda: "m5",
        _kernel=lambda name: kernel,
        _MAIN_TILED="(int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2)",
        _Baked=lambda *args: object(),
        _kernels={},
        lane_matmul=base,
        weight_bits=lambda *args: 4,
    )
    fake = ModuleType("yunshu_engine.kernels.tensorfold")
    fake.lane_qmm = q
    monkeypatch.setitem(sys.modules, "yunshu_engine.kernels.tensorfold", fake)
    undo = half.install(row_block=32)
    for rows in (1, 16, 17, 32):
        x = SimpleNamespace(size=rows * 5120, shape=(rows, 5120))
        result = q.lane_matmul(x, None, None, tiled=True, sk=4, row_block=16)
        assert result["nt"] == 16
        assert result["sk"] == 4  # changing the column tile must never retune K slices
        assert result["row_block"] == (32 if rows > 16 else 16)
    undo()
    assert q.lane_matmul is base
    assert q._kernels["main_tiled"] is kernel


def test_projection_dependency_chain_does_not_modify_inputs():
    from types import SimpleNamespace

    inputs = [object(), object(), object()]
    dependencies = []
    seen = []

    def depends(x, previous):
        dependencies.append((x, previous))
        return x

    def layer(x):
        seen.append(x)
        return len(seen)

    fake = SimpleNamespace(depends=depends)
    assert probe.projection_graph(fake, [layer] * 3, inputs, serial=True) == [1, 2, 3]
    assert seen == inputs
    assert dependencies == [(inputs[1], 1), (inputs[2], 2)]


def test_direct8_selective_route_avoids_measured_wide_regression():
    spec = importlib.util.spec_from_file_location(
        "direct_route", PROBE.parent / "lane_direct8.py"
    )
    direct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct)
    for rows in range(1, 33):
        assert direct.direct_for(34816, 5120, rows) is (rows <= 16)
        assert direct.direct_for(5120, 25600, rows)


def test_pair16_keeps_group_fma_and_split_reduction_source():
    spec = importlib.util.spec_from_file_location(
        "pair16_source", PROBE.parent / "lane_pair16.py"
    )
    pair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    # Includes strict source guards for descriptor, group-fma expression,
    # stage barrier count and byte-identical split-K reduction suffix.
    result = pair.preflight(PROBE.parents[2])
    assert set(result) == {"main", "bytes"}
    assert all(len(value) == 64 for value in result.values())


def test_reference_mode_restores_native_without_installing_a_variant():
    from types import SimpleNamespace

    native, old = object(), object()
    q = SimpleNamespace(lane_matmul=native)
    reference = SimpleNamespace(lane_matmul=old)

    def fail():
        raise AssertionError("research installer must not run in native reference mode")

    undo = probe.activate_mode(q, reference, native, fail, "base-r1")
    assert q.lane_matmul is old
    undo()
    assert q.lane_matmul is native
    undo = probe.activate_mode(q, reference, native, fail, "cand")
    assert q.lane_matmul is native
    undo()
