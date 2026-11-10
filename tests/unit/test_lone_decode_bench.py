import importlib.util
from pathlib import Path


def test_summarize():
    p = Path(__file__).resolve().parents[2] / "scripts/research/lone_decode_bench.py"
    spec = importlib.util.spec_from_file_location("lone_decode_bench", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.summarize(
        [{"text": "a", "ct": 100, "s": 1.0}, {"text": "a", "ct": 100, "s": 2.0}]
    )
    assert r["texts_equal"] and r["tok_s_med"] == 75.0


def test_split():
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/stock_vlm_decode_probe.py"
    )
    spec = importlib.util.spec_from_file_location("p", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert abs(m.split(0.2, 100, 1.2)["decode_ms_per_token"] - 10.0) < 1e-9
