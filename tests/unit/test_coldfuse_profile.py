"""CPU-only attribution contracts (no MLX import)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[2] / "scripts/research/coldfuse_profile.py"
spec = importlib.util.spec_from_file_location("coldfuse_profile", PATH)
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


def test_cold_validation_requires_complete_and_zero_cache():
    import pytest

    spec = importlib.util.spec_from_file_location(
        "coldfuse_analyze", PATH.with_name("coldfuse_analyze.py")
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
    long_cell = {**cell, "ct": 64}
    assert analyze.validate_cold(
        [long_cell, dict(complete=True)], expected_tokens=64
    ) == [long_cell]
    with pytest.raises(ValueError):
        analyze.validate_cold([long_cell, dict(complete=True)])
    for rows in (
        [cell],
        [dict(complete=True)],
        [{**cell, "cached": 32767}, dict(complete=True)],
    ):
        with pytest.raises(ValueError):
            analyze.validate_cold(rows)


def test_copy_key_records_callsite():
    import inspect

    frame = inspect.currentframe()
    assert profile.copy_key("concatenate", frame).startswith(
        f"copy.concatenate:{__file__}:"
    )


def test_norm_pilot_preserves_bfloat_rounding_boundaries():
    spec = importlib.util.spec_from_file_location(
        "coldfuse_norm_probe", PATH.with_name("coldfuse_norm_probe.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert "bfloat16_t normed = bfloat16_t(vals[i] * inv)" in probe.SOURCE
    assert "float(normed) * float(bfloat16_t(INVERSE ?" in probe.SOURCE


def test_paired_analysis_requires_exact_logprobs_and_same_workload():
    import pytest

    spec = importlib.util.spec_from_file_location(
        "coldfuse_analyze", PATH.with_name("coldfuse_analyze.py")
    )
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    cell = dict(
        ctx=32768,
        kind="code",
        device="m5",
        cached=0,
        ct=2,
        sha="same",
        lp_tokens=["a", "b"],
        lps=[-0.2, -0.3],
        ttft_s=33.0,
    )
    got = analyze.compare_arms([cell], [{**cell, "ttft_s": 32.5}])
    assert got[0]["saved_s"] == 0.5
    timing = {**cell, "lp_tokens": None, "lps": None}
    assert analyze.compare_arms([timing], [timing], require_logprobs=False)
    for changed in (
        {"cached": 1},
        {"device": "m3"},
        {"lps": [-0.2, -0.31]},
        {"lp_tokens": []},
        {"sha": None},
        {"ct": 0},
        {"ctx": 8192},
    ):
        with pytest.raises(ValueError):
            analyze.compare_arms([cell], [{**cell, **changed}])


def test_rehearsal_logprob_namespace_expects_cold_not_invariant_hit():
    import pytest

    spec = importlib.util.spec_from_file_location(
        "splitinv_rehearsal", PATH.with_name("splitinv_rehearsal.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.check_rehearsal_cache({"cached": 32768})
    probe.check_rehearsal_cache({"cached": 0}, logprobs=True)
    for cached, logprobs in ((0, False), (32767, False), (32768, True)):
        with pytest.raises(RuntimeError):
            probe.check_rehearsal_cache({"cached": cached}, logprobs=logprobs)
