"""CPU-only: swap_probe parsers, summary rule and one real sample of this process."""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import swap_probe as sp

FOOT = """\
      Dirty     (Swapped)         Clean   Reclaimable    Regions    Category
        ---           ---           ---           ---        ---    ---
 100000 B   900000 B           0 B           0 B          9    IOAccelerator
   5000 B           0 B     7000 B           0 B          5    Malloc Small
 105000 B   900000 B        7000 B           0 B         14    TOTAL
"""
VM = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   10.
Pages inactive:                               20.
Pages wired down:                             30.
File-backed pages:                            40.
Anonymous pages:                              50.
Pages stored in compressor:                   60.
Pages occupied by compressor:                 70.
Pageouts:                                     80.
Swapins:                                      90.
Swapouts:                                     100.
"""


def test_parse_footprint_categories_and_total():
    f = sp.parse_footprint(FOOT)
    assert f["total"]["swapped"] == 900000
    assert f["categories"]["IOAccelerator"]["swapped"] == 900000
    assert sp.top_swapped(f) == [("IOAccelerator", 900000)]
    with pytest.raises(ValueError):
        sp.parse_footprint(FOOT.replace("TOTAL", "OTHER"))


def test_parse_vm_stat_and_swapusage():
    vm = sp.parse_vm_stat(VM)
    assert (vm["page"], vm["compressor_stored"], vm["swapouts"]) == (16384, 60, 100)
    with pytest.raises(ValueError):
        sp.parse_vm_stat(VM.replace("Swapouts", "X"))
    u = sp.parse_swapusage(
        "vm.swapusage: total = 4096.00M  used = 2953.69M  free = 1142.31M  (encrypted)"
    )
    assert u == dict(total_mb=4096.0, used_mb=2953.69, free_mb=1142.31)


def _s(t, swapped, swapouts, used):
    vm = sp.parse_vm_stat(VM)
    vm["swapouts"] = swapouts
    return dict(
        t=t,
        foot_swapped_bytes=swapped,
        top_swapped=[],
        vm=vm,
        swapusage=dict(total_mb=4096.0, used_mb=used, free_mb=1.0),
    )


def test_summary_separates_footprint_swapped_from_real_swapouts():
    samples = [_s(0.0, 0, 100, 10), _s(1.0, 5e9, 100, 10), _s(2.0, 5e9, 100, 10)]
    out = sp.summarize(samples, [1.1, 1.2, 2.1])
    assert out["peak_foot_swapped_mb"] == 5000
    assert out["swapouts_delta_mb"] == 0 and out["swapusage_used_max_delta_mb"] == 0
    assert out["decode_tok_s_when_swapped_growing"] == 2
    assert out["decode_tok_s_when_swapped_flat"] == 1
    with pytest.raises(ValueError):
        sp.summarize([], [])


def test_sample_once_reads_this_process():
    s = sp.sample_once(os.getpid(), time.time())
    assert s["foot_swapped_bytes"] >= 0 and s["vm"]["page"] > 0
    assert s["swapusage"]["total_mb"] >= 0


def test_request_rates_split_by_request_marks():
    marks = {
        "request0_start": 0.0,
        "request0_end": 2.0,
        "request0_hog": False,
        "request1_start": 3.0,
        "request1_end": 6.0,
        "request1_hog": True,
    }
    tokens = [0.5, 1.5, 2.0, 3.5, 4.5, 5.5, 6.0]
    rows = sp.request_rates(marks, tokens, 3)
    assert [r["request"] for r in rows] == [0, 1]
    assert rows[0]["decode_tok_s"] == pytest.approx(2 / 1.5)
    assert rows[1]["decode_tok_s"] == pytest.approx(3 / 2.5)
    assert rows[1]["under_hog"] is True


def test_hog_stops_when_target_already_reached():
    assert sp.hog(1e6, 0.2, chunk_mb=1) == 0
