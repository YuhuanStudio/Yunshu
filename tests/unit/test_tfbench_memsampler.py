import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import tfbench  # noqa: E402

GIB = 2**30


def _patch(monkeypatch, footprint, used):
    import process_memory

    monkeypatch.setattr(
        process_memory,
        "process_tree_memory",
        lambda pid: {
            "physical_footprint_sum_bytes": footprint,
            "rss_sum_bytes": footprint,
        },
    )
    seq = iter(used)
    return lambda: next(seq)


def test_system_delta_counts_mmap_weights_missing_from_footprint(monkeypatch):
    # footprint 3 GiB (mmap'd weights absent) but host used grew 20 GiB
    fn = _patch(monkeypatch, 3 * GIB, [30 * GIB, 50 * GIB, 41 * GIB])
    s = tfbench.MemSampler(None, baseline_bytes=10 * GIB, used_fn=fn)
    assert s.method == "system-delta"
    assert s.sample() == 20.0
    s.sample()
    assert s.sample() == 31.0
    assert s.peak_gib == 40.0 and s.last_gib == 31.0
    assert s.peak_footprint_gib == 3.0


def test_without_baseline_falls_back_and_is_labelled(monkeypatch):
    fn = _patch(monkeypatch, 3 * GIB, [])
    s = tfbench.MemSampler(None, baseline_bytes=None, used_fn=fn)
    assert s.method == "process-footprint"
    assert s.sample() == 3.0
