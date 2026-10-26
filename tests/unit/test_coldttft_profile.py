"""CPU-only attribution contracts (no MLX import)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[2] / "scripts/research/coldttft_profile.py"
spec = importlib.util.spec_from_file_location("coldttft_profile", PATH)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_nested_exclusive_does_not_double_count():
    assert profile.exclusive(10, 4) == 6
    assert profile.exclusive(1, 1.00001) == 0


def test_projection_shape_uses_all_leading_dimensions():
    module = SimpleNamespace(input_dims=5120, output_dims=17408, bits=4)
    assert profile.projection_key(module, SimpleNamespace(shape=(1, 2048, 5120))) == (
        "gemm:M2048:K5120:N17408:b4"
    )


def test_pointer_induction_fails_closed_on_unexpected_source():
    spec = importlib.util.spec_from_file_location(
        "coldttft_loader", PATH.with_name("coldttft_loader.py")
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    import pytest

    with pytest.raises(ValueError, match="source changed"):
        loader.induction_header("unknown header")


def test_induced_addresses_equal_original_for_all_packs():
    # Integer-only proof of each byte address, across both packed N tiles and K groups.
    for bits in (4, 5, 8):
        for n in (0, 1, 31, 32, 63, 5120, 17407):
            for kg in (80, 96, 272):
                ptr = (((n // 32) * kg) * 32 + n % 32) * (64 * bits // 8)
                for g in range(kg):
                    assert ptr + g * 32 * (64 * bits // 8) == (
                        ((n // 32 * kg + g) * 32 + n % 32) * (64 * bits // 8)
                    )


def test_induction_transforms_the_current_loader_without_arithmetic_changes():
    import ast

    spec = importlib.util.spec_from_file_location(
        "coldttft_loader", PATH.with_name("coldttft_loader.py")
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    source = (
        PATH.parents[2] / "python/yunshu_engine/kernels/nax_prefill.py"
    ).read_text()
    tree = ast.parse(source)
    body = next(
        n.value.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "loader" for t in n.targets)
    )
    transformed = loader.induction_header(body)
    arithmetic = (
        "for(int i=0;i<READS;i++) dequantize<T,PF,bits>(src+i*BP,scale,bias,dst+i*PF);"
    )
    assert arithmetic in body and arithmetic in transformed
    assert "W += 32*(64*bits/8); SB += 2*N;" in transformed
    assert "SB[2*(g*N+n)]" not in transformed


def test_cold_validation_requires_complete_and_zero_cache():
    import pytest

    spec = importlib.util.spec_from_file_location(
        "coldttft_analyze", PATH.with_name("coldttft_analyze.py")
    )
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    cell = dict(
        part="decode",
        phase="cold",
        cached=0,
        ct=1,
        finish="length",
        device="m5",
        ttft_s=33.0,
    )
    assert analyze.validate_cold([cell, dict(complete=True)]) == [cell]
    for rows in (
        [cell],
        [dict(complete=True)],
        [{**cell, "cached": 32767}, dict(complete=True)],
    ):
        with pytest.raises(ValueError):
            analyze.validate_cold(rows)
