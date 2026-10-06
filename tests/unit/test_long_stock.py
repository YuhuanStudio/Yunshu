"""long_stock.py helpers (CPU only): the request wrapper and needle scoring."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/research"))
spec = importlib.util.spec_from_file_location(
    "long_stock", REPO / "scripts/research/long_stock.py"
)
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)


def fake_gen(model, proc, prompt, max_tokens, **kw):
    for i in range(3):
        yield SimpleNamespace(
            text=f"w{i} ",
            generation_tokens=i + 1,
            prompt_tokens=7,
            finish_reason="length" if i == 2 else None,
            generation_tps=12.34,
        )


def test_run_one_collects_text_and_counts():
    r = ls.run_one(fake_gen, None, None, "p", 3)
    assert r["text"] == "w0 w1 w2 " and r["ct"] == 3 and r["finish"] == "length"
    assert r["pt"] == 7 and r["dec_tps"] == 12.34


def test_score_needle():
    items = [("A", "111111"), ("B", "222222")]
    s = ls.score_needle(items, ["it is 111111", "unknown"])
    assert [x["correct"] for x in s] == [True, False]


def test_empty_stream_is_an_error():
    import pytest

    with pytest.raises(RuntimeError):
        ls.run_one(lambda *a, **k: iter(()), None, None, "p", 3)
