"""ple_read_bench: imports without MLX and times a call (the real table is only opened in main)."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/research/ple_read_bench.py"
_SPEC = importlib.util.spec_from_file_location("ple_read_bench", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_timed_returns_seconds_and_the_result():
    seconds, out = bench.timed(lambda a, b: a + b, 2, 3)
    assert out == 5 and seconds >= 0
