import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "agentload_repro",
    Path(__file__).resolve().parents[2] / "scripts/research/agentload_repro.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_rates_and_summary():
    chunks = [(10.0 + i * 0.1, i + 1) for i in range(100)]  # 10 tok/s
    w = mod.rates(chunks, 10.0, 20.0, 5.0)
    assert [x["tps"] for x in w] == [10.0, 9.8]
    s = mod.summarize(chunks, 9.0)
    assert s["tokens"] == 100 and s["ttft_s"] == 1.0 and round(s["mean_tps"]) == 10


def test_gpu_busy_reads_slices():
    rows = [
        {"gpu": {"m": {"slices": {"busy_seconds": 1.0}}}},
        {"gpu": {"m": {"slices": {"busy_seconds": 9.5}}}},
    ]
    assert mod.gpu_busy(rows) == (1.0, 9.5)
