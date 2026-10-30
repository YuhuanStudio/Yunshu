import importlib.util
from pathlib import Path


def test_summarize():
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.summarize({"gdn@1024": [2, 1.23456]}) == {
        "gdn@1024": {"calls": 2, "seconds": 1.2346}
    }
