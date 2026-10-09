import importlib.util
from pathlib import Path

s = importlib.util.spec_from_file_location(
    "c", Path(__file__).parents[2] / "scripts/research/spec_identity_compare.py"
)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def _cell(text, drafted):
    return {"text": text, "drafted": drafted, "rounds": 1}


def test_equal_text_and_engaged_passes():
    k = (1024, "prose", "cold")
    assert m.compare({k: _cell("abc", 5)}, {k: _cell("abc", 0)}) == []


def test_text_mismatch_and_unengaged_lane_both_fail():
    k = (1024, "prose", "cold")
    assert (
        "differ at char 1" in m.compare({k: _cell("abc", 5)}, {k: _cell("axc", 0)})[0]
    )
    assert "never engaged" in m.compare({k: _cell("abc", 0)}, {k: _cell("abc", 0)})[0]
    assert m.compare({}, {}) != []
