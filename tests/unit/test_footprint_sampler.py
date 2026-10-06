import importlib.util
import pathlib
import sys
import time

from yunshu_engine import footprint_sampler as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_peak_catches_short_spike():
    vals = iter([100, 100, 900, 100, 100] + [100] * 10000)
    s = fs.FootprintSampler(0.001, read=lambda: next(vals)).start()
    time.sleep(0.15)
    s.stop()
    assert s.peak == 900 and s.samples > 5


def test_self_footprint_positive_on_macos():
    if sys.platform == "darwin":
        assert fs.self_footprint_bytes() > 1_000_000


def test_metric_lines_and_setting(monkeypatch):
    monkeypatch.setattr(fs, "_SAMPLER", None)
    assert fs.metric_lines() == []
    s = fs.FootprintSampler(read=lambda: 5)
    s.sample_once()
    monkeypatch.setattr(fs, "_SAMPLER", s)
    assert 'yunshu_process_footprint_bytes{type="peak"} 5' in fs.metric_lines()
    from yunshu_engine import settings

    assert settings.get("YUNSHU_FOOTPRINT_SAMPLE_MS") == 0


def test_memory_ab_reads_server_peak():
    p = ROOT / "scripts/research/memory_ab.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("mab", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (
        m.server_peak_gib({'yunshu_process_footprint_bytes{type="peak"}': 12.5}) == 12.5
    )
    assert m.server_peak_gib({}) is None
