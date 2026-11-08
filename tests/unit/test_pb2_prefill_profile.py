import importlib.util
from pathlib import Path


def test_summarize():
    p = Path(__file__).resolve().parents[2] / "scripts/research/pb2_prefill_profile.py"
    spec = importlib.util.spec_from_file_location("pp", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    s = m.summarize([(0, "gdn", 0.1), (1, "gdn", 0.3), (3, "attn", 0.2)])
    assert s["gdn"] == {"total": 0.4, "calls": 2, "mean_ms": 200.0}
    assert s["attn"]["calls"] == 1
