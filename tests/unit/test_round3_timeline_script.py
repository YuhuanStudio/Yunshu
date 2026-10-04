import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tl",
    Path(__file__).resolve().parents[2] / "scripts/research/round3_prefill_timeline.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_summarize():
    s = m.summarize([{"kind": "decode", "dur": 0.5}, {"kind": "decode", "dur": 0.25}])
    assert s == {"decode": {"n": 2, "s": 0.75, "tokens": 0}}
