"""Timing cells require an overlapping window and real warm-prefix accounting."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "bqsa_cost", Path(__file__).resolve().parents[2] / "scripts/research/bqsa_cost.py"
)
assert spec and spec.loader
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)


def test_window_rates_use_only_simultaneous_decode():
    result = cost.window_rates(
        [{"times": [0, 1, 2, 3, 4]}, {"times": [1, 2, 3, 4, 5]}], 3
    )
    assert result["row_tps"] == [1.0, 1.0]
    assert result["step_ms"] == 1000
    with pytest.raises(RuntimeError, match="too short"):
        cost.window_rates([{"times": [0, 1, 2]}, {"times": [1, 2, 3]}], 3)


def test_every_row_requires_a_full_warm_prefix():
    cost.require_warm_cache({"cached": [8192, 8128], "prompt_tokens": [8193, 8193]})
    for cached in ([8192, 0], [8192, None], [8192, 8000], [8192, True]):
        with pytest.raises(RuntimeError):
            cost.require_warm_cache({"cached": cached, "prompt_tokens": [8193, 8193]})


def test_calibration_and_actual_prompt_guard():
    assert cost.calibrated_chars(lambda n: n // 2, 8192) == 16384
    cost.require_context({"prompt_tokens": [8191, 8200]}, 8192)
    with pytest.raises(RuntimeError, match="unexpected prompt"):
        cost.require_context({"prompt_tokens": [8192, 10000]}, 8192)
