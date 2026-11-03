"""Engaged-mode evidence lines from the 2026-11-03 wiring smoke (TensorFold 1.1.8, Splash 1.5.0):
the tfbench parser returns the real line, and the audit probe regex matches that same line.
CPU only: log text only."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit_manifest as fam  # noqa: E402
import tfbench  # noqa: E402

TF118_LOG = (
    "info: Qwen3.8-27B: 1 streams of 140000 tokens, DFlash2 drafts for one of them\n"
    "[tensorfold] serving Qwen3.8-27B-MLX-4bit at http://127.0.0.1:18990/v1 "
    "(sampling: temperature 1.0, top_k 20, top_p 0.95; drafts: on; context: 140000; loaded in 14.5s)\n"
)
SPLASH_LOG = (
    "00:04:46 Decode cycles verify every row of the draft's block (--verify-rows auto, as "
    "/Volumes/P5Plus/models/TensorFold/Qwen3.8-27B-MLX-4bit is dense).\n"
    "00:06:09 Ready · TensorFold/Qwen3.8-27B-MLX-4bit · context 140,000 · http://127.0.0.1:18990\n"
)


def test_tf_1_1_8_engaged_line_is_the_drafts_for_line():
    mode, line = tfbench.engaged_spec_mode_line("tf-1.1.8", TF118_LOG)
    assert mode == "dflash"
    assert line.startswith("info: Qwen3.8-27B: 1 streams") and "DFlash2 drafts for" in line


def test_tf_1_1_8_without_drafts_on_does_not_engage():
    log = TF118_LOG.replace("drafts: on", "drafts: off")
    assert tfbench.engaged_spec_mode_line("tf-1.1.8", log) == (None, None)


def test_tf_1_1_8_probe_regex_matches_the_real_line():
    pat = fam.EXPECTED_MODE_PROBE["tf-1.1.8"]["log_patterns"][0]["regex"]
    assert re.search(pat, TF118_LOG)


def test_splash_engaged_line_is_the_draft_block_line():
    mode, line = tfbench.engaged_spec_mode_line("splash-1.5.0", SPLASH_LOG)
    assert mode == "dflash" and "verify every row of the draft's block" in line
    assert tfbench.engaged_spec_mode_line("splash-1.5.0", "00:06:09 Ready\n") == (None, None)


def test_splash_probe_regex_matches_the_real_line():
    pat = fam.EXPECTED_MODE_PROBE["splash-1.5.0"]["log_patterns"][0]["regex"]
    assert re.search(pat, SPLASH_LOG)


def test_splash_pin_is_the_interpreter_the_harness_launches():
    assert fam.INSTALL["splash-1.5.0"].endswith("python/bin/python3")
