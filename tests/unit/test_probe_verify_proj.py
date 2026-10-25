"""CPU checks for the projection bandwidth probe's geometry and bookkeeping."""

import importlib.util
from pathlib import Path

PROBE = Path(__file__).parents[2] / "scripts/research/probe_verify_proj.py"
spec = importlib.util.spec_from_file_location("probe_verify_proj", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_geometry_down_proj_16_rows():
    g = probe.geometry(5120, 17408, 4, 16, 64)
    assert g["kernel"] == "main_tiled"
    assert g["sk"] == 8 and g["rows_per_tg"] == 16
    assert g["threadgroups"] == 160 and g["simdgroups"] == 1280


def test_geometry_blocks_and_sk_override():
    assert probe.geometry(17408, 5120, 4, 32, 64)["rows_per_tg"] == 16
    assert probe.geometry(17408, 5120, 4, 8, 64)["rows_per_tg"] == 16
    assert probe.geometry(17408, 5120, 4, 16, 64, sk=4)["threadgroup"] == [128, 1, 1]
    assert probe.geometry(5120, 6144, 5, 16, 64)["kernel"] == "bytes"


def test_copies_and_gbps():
    assert probe.copies_for(44 * 2**20, 768 * 2**20) == 18
    assert probe.copies_for(2**30, 2**20) == 2
    assert abs(probe.gbps(1e9, 1, 0.002) - 500) < 1e-6


def test_presets_cover_shapes():
    names = {r[0] for r in probe.PRESETS["27b"]}
    assert {"mlp.down4", "mlp.gate", "gdn.b4", "attn.k4"} <= names
    for rows in probe.PRESETS.values():
        for _, n, k, _bits, _group, _c in rows:
            assert k % 64 == 0 and (n % 4 == 0)


def test_moe_probe_bytes():
    spec2 = importlib.util.spec_from_file_location(
        "probe_moe_experts", PROBE.with_name("probe_moe_experts.py")
    )
    moe = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(moe)
    assert moe.picked_bytes(10, 100, 7) == 1007


def test_boundary_probe_params():
    spec3 = importlib.util.spec_from_file_location(
        "probe_boundary", PROBE.with_name("probe_boundary.py")
    )
    pb = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(pb)
    n16 = 50 * 2**20 // 16
    first = pb.params_list(0, 48, n16, 8, 0, 1 << 18, 1 << 14)
    assert first[:4] == [0, n16, n16, 8 * 2**20 // 16] and first[5] == 1 << 14
    last = pb.params_list(47, 48, n16, 8, 0, 1 << 18, 1 << 14)
    assert last[3] == 0 and last[5] == 0  # no next slice: no prefetch threads
    assert pb.params_list(3, 48, n16, 0, 0, 1 << 18, 1 << 14)[3] == 0
    assert pb.params_list(0, 4, n16, 0, 0, 8, 8, big=True)[1] == 4 * n16
