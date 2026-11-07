"""The decode phases a cell runs are the phases its manifest row plans (lead 2026-11-03).

Before the fix, tfbench ran cold, warm and turn2 for every decode cell, so the smoke-profile cell
(planned ["cold"]) wrote three request rows and failed record_count. CPU only: no gpuq, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import tfbench  # noqa: E402


def _phases_arg(args):
    return args[args.index("--phases") + 1].split(",") if "--phases" in args else None


def test_smoke_profile_cell_runs_only_its_planned_cold_phase():
    row = next(r for r in fm.build_smoke_rows() if r["sweep"] == "smoke-profile")
    part, args, _ = fr.harness_part(row)
    assert part == "decode"
    assert _phases_arg(args) == ["cold"]


def test_smoke_core_cell_runs_its_three_planned_phases_in_order():
    row = next(r for r in fm.build_smoke_rows() if r["sweep"] == "smoke-core")
    part, args, _ = fr.harness_part(row)
    assert _phases_arg(args) == ["cold", "warm", "turn2"]


def test_every_decode_harness_phase_list_is_accepted_by_tfbench():
    for row in fm.build_smoke_rows():
        part, args, _ = fr.harness_part(row)
        if part != "decode":
            continue
        phases = _phases_arg(args)
        assert phases is not None, row["id"]
        assert set(phases) == set(row["phases"]) & {"cold", "warm", "turn2"}, row["id"]
        tfbench.parse_phases(",".join(phases))  # raises SystemExit on a bad list


def test_conc_cell_is_not_given_a_phase_list():
    row = next(r for r in fm.build_smoke_rows() if r["sweep"] == "smoke-conc")
    part, args, _ = fr.harness_part(row)
    assert part == "conc" and "--phases" not in args
