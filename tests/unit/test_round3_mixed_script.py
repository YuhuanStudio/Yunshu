import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "round3_mixed",
    Path(__file__).resolve().parents[2] / "scripts/research/round3_mixed.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_workload_is_the_same_for_both_arms_and_mixed():
    assert m.workload(8, 1) == m.workload(8, 1)
    assert m.workload(8, 1) != m.workload(8, 2)
    w = m.workload(8, 0)
    assert {r["kind"] for r in w} == {"code_python", "novel_en"}
    assert all(r["pp"] in m.LENGTHS and 0 <= r["at"] <= m.STAGGER_S for r in w)
    assert [r["at"] for r in w] == sorted(r["at"] for r in w)


def test_summary_and_p90():
    rows = [
        {"ttft_s": float(i), "decode_tps": 10.0, "completion_tokens": 100}
        for i in range(1, 11)
    ]
    s = m.summarize(rows, 20.0)
    assert s["mean_ttft_s"] == 5.5 and s["p90_ttft_s"] == 10.0
    assert s["aggregate_tps"] == 50.0
    compile(m.SERVE, "serve", "exec")
