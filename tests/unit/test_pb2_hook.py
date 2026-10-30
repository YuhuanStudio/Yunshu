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


def test_apc_trace_times_calls_and_profiles_slow_ones(tmp_path, monkeypatch):
    import time
    import types

    out = tmp_path / "apc.txt"
    monkeypatch.setenv("PB2_APC_OUT", str(out))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_apc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class _Coordinator:
        def checkpoint_lengths(self):
            return [1]

        def lookup(self):
            time.sleep(0.03)
            return self.checkpoint_lengths()

        def store_checkpoint(self):
            return None

        def prepare_prefill(self):
            return None

    m.patch_apc(types.SimpleNamespace(_Coordinator=_Coordinator))
    assert _Coordinator().lookup() == [1]
    text = out.read_text()
    assert "apc.lookup" in text and "apc.checkpoint_lengths" in text
    assert "|" in text  # the slow outer call carries a profile
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]


def test_timeline_wrapper_logs_label_duration_and_start(tmp_path, monkeypatch):
    import types

    out = tmp_path / "tl.txt"
    monkeypatch.setenv("PB2_TL_OUT", str(out))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_tl", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    holder = types.SimpleNamespace(f=lambda: 5)
    m.wrap_tl(holder, "f", "x.f")
    m.wrap_tl(holder, "missing", "x.missing")  # absent attributes are skipped
    assert holder.f() == 5
    line = out.read_text().strip()
    assert line.startswith("x.f ") and " at " in line
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]
