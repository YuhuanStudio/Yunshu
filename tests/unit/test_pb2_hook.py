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


def test_runner_trace_logs_stats_timestamps(tmp_path, monkeypatch):
    import types
    from dataclasses import dataclass

    out = tmp_path / "trace.txt"
    monkeypatch.setenv("PB2_TRACE_OUT", str(out))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_trace", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    @dataclass
    class RunStats:
        t_submit: float = 0.0
        t_first: float = 0.0
        prefill_total: int = 0
        cached_tokens: int = 0

    mod = types.SimpleNamespace(RunStats=RunStats, other=3)
    m.patch_runner(mod)
    s = RunStats()
    s.prefill_total = 7
    s.t_submit = 1.5
    s.t_first = 2.5
    lines = out.read_text().splitlines()
    assert len(lines) == 2 and "t_submit" in lines[0] and "t_first" in lines[1]
    assert "pt=7" in lines[0]
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]
