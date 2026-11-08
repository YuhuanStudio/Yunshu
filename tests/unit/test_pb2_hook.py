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


def test_timeline_wrapper_keeps_static_and_class_methods(tmp_path, monkeypatch):
    out = tmp_path / "tl2.txt"
    monkeypatch.setenv("PB2_TL_OUT", str(out))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_tl2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class Runner:
        @staticmethod
        def static(group):
            return ("static", group)

        @classmethod
        def klass(cls, x):
            return (cls.__name__, x)

        def method(self, x):
            return ("method", x)

    for name in ("static", "klass", "method"):
        m.wrap_tl(Runner, name, f"Runner.{name}")
    r = Runner()
    assert r.static(1) == ("static", 1) and Runner.static(2) == ("static", 2)
    assert r.klass(3) == ("Runner", 3)
    assert r.method(4) == ("method", 4)
    labels = [ln.split()[0] for ln in out.read_text().splitlines()]
    assert labels == ["Runner.static", "Runner.static", "Runner.klass", "Runner.method"]
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]


def test_pool_trace_logs_takes_and_offers(tmp_path, monkeypatch):
    import types

    out = tmp_path / "pool.txt"
    monkeypatch.setenv("PB2_TL_OUT", str(out))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_pool", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class A:
        shape = (1, 2, 8, 4)
        nbytes = 100

    mod = types.SimpleNamespace(
        take_restore_buffer=lambda a, capacity: None if capacity > 8 else a,
        offer_arrays=lambda arrays, aged=False: True,
        offer_restore_buffers=lambda caches: None,
        restore_buffer_bytes=lambda: 0,
        _STATE={"limit": 123},
    )
    m.wrap_pool(mod)
    assert mod.take_restore_buffer(A(), 16) is None
    assert mod.take_restore_buffer(A(), 4) is not None
    assert mod.offer_arrays([A(), A()]) is True
    text = out.read_text()
    assert "pool.take miss" in text and "pool.take hit" in text
    assert "pool.offer taken bytes=200 limit=123" in text
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]


def test_layer_checksum_probe_logs_input_and_output_sums(tmp_path, monkeypatch):
    import types

    import pytest

    mx = pytest.importorskip("mlx.core")
    out = tmp_path / "lsum.txt"
    monkeypatch.setenv("PB2_LAYERSUM_OUT", str(out))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_lsum", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class Layer:
        def __call__(self, x, *args, **kw):
            return x * 2

    m.patch_layersum(types.SimpleNamespace(Qwen3_5DecoderLayer=Layer))
    Layer()(mx.ones((1, 4, 3)))
    Layer()(mx.ones((1, 1, 3)))  # a single row is not logged
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert "rows=4" in lines[0] and "in=12" in lines[0] and "out=24" in lines[0]
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]


def test_memory_holders_inventory_reads_known_retainers(tmp_path, monkeypatch):
    import types

    import pytest

    pytest.importorskip("mlx.core")
    monkeypatch.setenv("PB2_MEM_OUT", str(tmp_path / "mem.txt"))
    p = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/pb2_hook/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("pb2hook_mem", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    gib = 1 << 30
    runner = types.SimpleNamespace(
        apc_manager=types.SimpleNamespace(resident_bytes=lambda: 3 * gib),
        driver=None,
        _aux_driver=None,
    )
    got = m.holders(runner)
    assert got["apc_resident"] == 3.0
    assert {"restore_pool", "dflash_windows"} <= set(got)
    # a broken retainer is skipped, not fatal
    runner.apc_manager = types.SimpleNamespace()
    assert "apc_resident" not in m.holders(runner)
    import sys

    sys.meta_path[:] = [f for f in sys.meta_path if "Finder" not in type(f).__name__]
