import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "round3_warm",
    Path(__file__).resolve().parents[2] / "scripts/research/round3_warm.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_every_request_gets_its_own_slice():
    s = m.spans(32768, 2048, 3)
    assert s[0] == (0, 32768, 34816) and s[1][0] == 34816
    ends = [x[2] for x in s]
    assert all(s[i + 1][0] == ends[i] for i in range(2))


def test_summary_and_serve_wrapper():
    rows = [
        {
            "ttft_s": 2.0,
            "decode_tps": 30.0,
            "completion_tokens": 1500,
            "cached_tokens": 32768,
        },
        {
            "ttft_s": 4.0,
            "decode_tps": 20.0,
            "completion_tokens": 1400,
            "cached_tokens": 32768,
        },
    ]
    s = m.summarize(rows, 90.0)
    assert s["mean_ttft_s"] == 3.0 and s["max_ttft_s"] == 4.0
    assert s["mean_decode_tps"] == 25.0 and s["min_decode_tps"] == 20.0
    compile(m.SERVE, "serve", "exec")
