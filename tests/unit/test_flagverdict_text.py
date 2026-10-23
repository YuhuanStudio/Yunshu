import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fv", Path(__file__).parents[2] / "scripts/research/flagverdict_text.py"
)
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)


def test_identical():
    assert fv.identical(["a", "b"], ["a", "b"])
    assert not fv.identical(["a"], ["a", "b"])
    assert not fv.identical(["a"], ["b"])


def test_summarize():
    assert fv.summarize([]) == {"n": 0}
    assert fv.summarize([1.0, None, 3.0])["mean"] == 2.0
