"""CPU-only checks of the push/poll bench helpers (the bench itself is a gpuq job)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import front_push_bench as fpb  # noqa: E402


def test_window_cpu_slices_by_time():
    s = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 100.0)]
    assert fpb.window_cpu(s, 2.0, 3.0) == {"n": 2, "mean_pct": 25.0, "max_pct": 30.0}
    assert fpb.window_cpu(s, 9.0, 10.0) == {"n": 0}
