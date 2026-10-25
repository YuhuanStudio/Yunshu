import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "decode_step_ablate",
    Path(__file__).parents[2] / "scripts/research/decode_step_ablate.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_summarize_ms():
    got = mod.summarize([0.030, 0.040, 0.050])
    assert got["median_ms"] == 40.0 and got["min_ms"] == 30.0 and got["max_ms"] == 50.0


def test_arms_cover_the_ablations():
    assert set(mod.ARMS) == {"lane", "no_xsum", "sk4", "sk8"}
