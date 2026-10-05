import importlib.util
import pathlib
import sys

p = pathlib.Path(__file__).parents[2] / "scripts/research"
sys.path.insert(0, str(p))
spec = importlib.util.spec_from_file_location("hol_longreply", p / "hol_longreply.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def test_summarize_and_judge():
    s = h.summarize([1.0, 1.5, 2.0], "x", True, 3.0)
    assert s["tok_s"] == 2.0 and s["ttft_s"] == 1.0
    assert h.summarize([], "", False, 1.0)["tok_s"] == 0.0
    a = dict(s, chunks=1500)
    b = dict(s, text=h.needle(5), chunks=3)
    ok = {"m": [{"scenarios": {"s1": {"doc": 5, "a": a, "b": b}}}]}
    assert h.judge(ok) == []
    short = {"m": [{"scenarios": {"s1": {"doc": 5, "a": s, "b": b}}}]}
    assert h.judge(short)
    miss = {"m": [{"scenarios": {"s1": {"doc": 6, "a": a, "b": b}}}]}
    assert h.judge(miss)
    assert h.judge({"m": []}) == ["m: no reps"]
