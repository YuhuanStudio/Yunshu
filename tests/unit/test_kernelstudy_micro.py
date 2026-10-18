"""CPU-only fail-closed contract for external-kernel screening."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "kernelstudy_micro",
    Path(__file__).parents[2] / "scripts/research/kernelstudy_micro.py",
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def row():
    return dict(
        candidate="attention",
        context=32768,
        rep=0,
        ours_ms=1.0,
        external_ms=0.8,
        max_abs=0.0,
        bit_equal=True,
    )


def test_dry_run_does_not_import_mlx(capsys):
    assert (
        probe.main(["--candidate", "attention", "--out", "/unused", "--dry-run"]) == 0
    )
    assert '"gpu_executed": false' in capsys.readouterr().out


def test_complete_is_not_serving_qualification():
    result = probe.validate([row()], "attention", [32768], 1)
    assert result["complete"] and not result["serving_qualified"]


@pytest.mark.parametrize("rows", [[], [row(), row()]])
def test_short_duplicate_fails(rows):
    with pytest.raises(ValueError):
        probe.validate(rows, "attention", [32768], 1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ours_ms", float("nan")),
        ("external_ms", -1),
        ("bit_equal", None),
        ("candidate", "moe"),
    ],
)
def test_malformed_fails(key, value):
    entry = row()
    entry[key] = value
    with pytest.raises(ValueError):
        probe.validate([entry], "attention", [32768], 1)


def test_numerical_mismatch_is_rejected_not_hidden():
    entry = row()
    entry.update(bit_equal=False, max_abs=0.01)
    result = probe.validate([entry], "attention", [32768], 1)
    assert result["complete"] and not result["component_parity"]


def test_missing_long_context_fails():
    with pytest.raises(ValueError):
        probe.validate([row()], "attention", [32768, 65536, 131072], 1)


def test_steel_harvest_contract():
    root = Path("/Volumes/Micron/yunshu-refsync/mlx")
    if not root.exists():
        pytest.skip("external source clone absent")
    header = probe.steel_header(root)
    assert "METAL_FUNC void kernelstudy_steel(" in header
    assert "threadgroup T Q_smem[" not in header
    assert "function_constant(" not in header
    assert "constant bool save_lse = false;" in header


def test_metal_body_extracts_nested_scopes_only():
    source = "void f(int a) { if(a) { a++; } } void next(){}"
    assert probe.metal_body(source, "f") == " if(a) { a++; } "


def test_release_vector_source_has_prefetch():
    path = Path(
        "/Volumes/Micron/yunshu-refsync/mlx/mlx/backend/metal/kernels/sdpa_vector.h"
    )
    if not path.exists():
        pytest.skip("external clone absent")
    body = probe.metal_body(path.read_text(), "sdpa_vector_2pass_1")
    assert "T k_reg[TK][qk_per_thread]" in body
    assert "for (; i + (TK - 1) * blocks < N;" in body
