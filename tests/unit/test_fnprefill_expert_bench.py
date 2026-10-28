"""CPU checks of the expert bench's row generator."""

import importlib.util
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parents[2] / "scripts/research"
spec = importlib.util.spec_from_file_location(
    "fnprefill_expert_bench", base / "fnprefill_expert_bench.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_rows_are_sorted_and_cover_experts():
    rng = np.random.default_rng(0)
    ids = bench.rows_per_expert(20480, 0.0, rng)
    assert len(ids) == 20480 and (np.diff(ids.astype(np.int64)) >= 0).all()
    assert len(set(ids.tolist())) > 500  # uniform: nearly every expert gets rows
    skewed = bench.rows_per_expert(20480, 0.5, rng)
    assert np.bincount(skewed.astype(np.int64)).max() > 2 * np.bincount(ids.astype(np.int64)).max()
