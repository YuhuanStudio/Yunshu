"""The swap-stability preflight job must outlive its own window (lead 2026-11-03, budget check).

gpuq sends SIGINT to a job when its --timeout (or the --short cap) expires or when its log stops
growing for STALL_S. The swap gate prints nothing until its window ends, so it must have: a timeout
that covers the window plus startup, a short-lane cap that does not cut it below the window, and a
stall threshold above the window. CPU only: reads constants, starts nothing.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_run as fr  # noqa: E402

GPUQ_SRC = Path(fr.GPUQ + ".py").read_text()  # the literal gpuq path the runner uses
STARTUP_MARGIN_S = 60


def _gpuq_const(name: str) -> float:
    m = re.search(rf"^{name} = (?:float\(os\.environ\.get\([^,]+,\s*)?([0-9.]+)", GPUQ_SRC, re.M)
    assert m, f"gpuq.py constant {name} not found"
    return float(m.group(1))


def test_swap_gate_timeout_covers_window_and_startup():
    default_window = 300.0  # fullcmp_preflight.py --window-s default (argparse)
    assert float(fr.PREFLIGHT_TIMEOUT_MIN) * 60 >= default_window + STARTUP_MARGIN_S


def test_swap_gate_timeout_is_not_cut_by_the_short_lane_cap():
    # the job is submitted --short: gpuq caps it at SHORT_MIN, so the requested timeout must fit
    assert float(fr.PREFLIGHT_TIMEOUT_MIN) * 60 <= _gpuq_const("SHORT_MIN") * 60


def test_stall_threshold_is_above_the_silent_window():
    # the preflight prints only at the end: a shorter stall threshold would SIGINT it mid-window
    assert _gpuq_const("STALL_S") > 300.0 + 10.0


def test_swap_gate_argv_requests_the_short_lane_with_that_timeout():
    src = inspect.getsource(fr.swap_gate)
    assert '"--short"' in src
    assert "PREFLIGHT_TIMEOUT_MIN" in src
