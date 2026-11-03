"""Engaged-mode parsers in tfbench: each engine's log line is recognised, and a log without the line
reports no mode (so the expected-mode check fails closed). CPU only, log text only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import tfbench  # noqa: E402


def test_omlx_dflash_drafter_line_engages_dflash():
    log = "boot ok\nDFlash enabled for qwen3.8 as a batched drafter, draft=/m/dflash\nready\n"
    mode, line = tfbench.engaged_spec_mode_line("omlx-0.7.3", log)
    assert mode == "dflash"
    assert "batched drafter" in line


def test_omlx_refusal_line_does_not_engage():
    log = "DFlash enabled for qwen3.8 but no draft model is set\n"
    assert tfbench.engaged_spec_mode_line("omlx-0.7.3", log) == (None, None)


def test_omlx_rc_uses_the_same_line():
    log = "DFlash enabled for m as a batched drafter, draft=/d\n"
    assert tfbench.engaged_spec_mode_line("omlx-0.7.3rc1", log)[0] == "dflash"


def test_missing_line_reports_no_mode():
    assert tfbench.engaged_spec_mode_line("omlx-0.7.3", "nothing here\n") == (
        None,
        None,
    )
