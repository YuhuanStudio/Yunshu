"""A --smoke-conc same-prompt row (sweep smoke-conc-same) gets --conc-same, like the real conc-same sweep."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402


def _row(sweep):
    row = fm._cell_row("yunshu", fm.SMOKE_BATCH, fm.spec(sweep=sweep, ctx="1K", kind="code", out=64,
                                                         conc=2, phases=("conc",)))
    row["id"] += ".v0"
    row["reps"] = 1
    row["prompt_variant"] = 0
    return row


def test_smoke_conc_same_row_passes_conc_same():
    part, extra, reason = fr.harness_part(_row("smoke-conc-same"))
    assert reason is None and part == "conc"
    assert "--conc-same" in extra


def test_smoke_conc_distinct_row_does_not():
    part, extra, reason = fr.harness_part(_row("smoke-conc-distinct"))
    assert reason is None and part == "conc"
    assert "--conc-same" not in extra


def test_real_conc_same_sweep_still_passes_conc_same():
    part, extra, _ = fr.harness_part(_row("conc-same"))
    assert "--conc-same" in extra
