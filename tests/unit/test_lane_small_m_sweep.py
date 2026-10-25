import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lane_small_m_sweep",
    Path(__file__).parents[2] / "scripts/research/lane_small_m_sweep.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_helpers():
    assert mod.gbps(1e9, 0.002) == 500.0
    assert mod.summarize([0.0, 3.0, 1.0, 2.0]) == 2.0
    assert all(n % 4 == 0 and k % 64 == 0 for _, n, k, _ in mod.SHAPES)
