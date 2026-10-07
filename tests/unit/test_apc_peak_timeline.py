"""CPU checks of the footprint timeline harness (no server, no GPU)."""

import importlib.util
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
GIB = 1 << 30


def _load():
    path = ROOT / "scripts" / "research" / "apc_peak_timeline.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("apc_peak_timeline_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_names_the_peak_and_the_gauges_around_it():
    m = _load()
    fp = [(0.0, 40 * GIB), (1.0, 48 * GIB), (2.0, 41 * GIB), (3.0, 40 * GIB)]
    gauges = [
        (0.9, {"active": 36.0, "cache": 9.0, "apc": 12.0}),
        (1.1, {"active": 37.0, "cache": 8.5, "apc": 12.0}),
        (3.0, {"active": 30.0, "cache": 0.0, "apc": 12.0}),
    ]
    s = m.summarize(fp, gauges, t_end=3.0)
    assert s["peak_gib"] == 48.0
    assert s["t_peak_before_end"] == 2.0
    assert s["active_max_near"] == 37.0 and s["cache_max_near"] == 9.0
    assert s["end_gib"] == 40.0
    assert m.summarize([], gauges, 3.0) is None


def test_gauge_names():
    m = _load()
    raw = {
        'yunshu_gpu_memory_bytes{type="active"}': 1.5,
        'yunshu_gpu_memory_bytes{type="cache"}': 2.5,
        'yunshu_gpu_memory_bytes{type="peak"}': 3.5,
        'yunshu_apc_resident_bytes{model_id="default"}': 4.5,
    }
    assert m.gauge_row(raw) == {"active": 1.5, "cache": 2.5, "peak": 3.5, "apc": 4.5}


def test_sidecar_samples_fast_and_windows_by_time():
    m = _load()
    side = m.Sidecar(0, lambda: 7 * GIB, lambda: {}, fp_period=0.005, gauge_period=0.02)
    side.start()
    t0 = time.time()
    time.sleep(0.2)
    t1 = time.time()
    side.close()
    fp, _ = side.take(t0, t1)
    assert len(fp) >= 10
    assert all(b == 7 * GIB for _, b in fp)


def test_session_mode_is_wired_to_the_branch_scenarios():
    m = _load()
    assert "session" in m.run_arm.__code__.co_varnames
    import apc_branch_ab

    assert "long_session" in apc_branch_ab.scenarios.__code__.co_varnames
