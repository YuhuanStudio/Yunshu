import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "slice_stats", Path(__file__).parents[2] / "scripts/research/slice_stats.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parse_groups_rows():
    lines = [
        "x slice t=1.0 dt=0.040 spec=False decode_only=True rows=4 emitted=4 uncached_waiting=0",
        "x slice t=2.0 dt=0.060 spec=False decode_only=True rows=4 emitted=4 uncached_waiting=0",
        "noise",
    ]
    got = mod.parse(lines)
    assert got[(4, True, False)] == [(0.04, 4), (0.06, 4)]
