import importlib.util
from pathlib import Path


def test_gbs():
    p = Path(__file__).parents[2] / "scripts/research/rowcost_lane_rows.py"
    spec = importlib.util.spec_from_file_location("rc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.gbs(1_000_000, 1.0) == 1.0
    assert m.ROWS[0] == 1
