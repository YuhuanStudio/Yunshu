"""The probe's dry run must not import MLX or start a server."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_dry_run():
    script = (
        Path(__file__).resolve().parents[2] / "scripts/research/restore_touch_probe.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"complete": True, "dry_run": True}


def test_rehearsal_identity_rejects_logprob_drift_or_missing_logprobs():
    import runpy

    script = (
        Path(__file__).resolve().parents[2] / "scripts/research/splitinv_rehearsal.py"
    )
    same = runpy.run_path(str(script))["same_output"]
    reference = {"toks": [1, 2], "lps": [-0.1, -0.2]}
    assert same(reference, reference)
    assert not same({"toks": [1, 2], "lps": [-0.1, -0.200001]}, reference)
    assert not same({"toks": [1, 2], "lps": None}, reference)
    assert not same({"toks": [1, 2], "lps": [-0.1]}, reference)
    assert not same({"toks": [1, 3], "lps": [-0.1, -0.2]}, reference)


def test_rehearsal_can_accept_a_complete_short_eos_without_weakening_default():
    import runpy

    script = (
        Path(__file__).resolve().parents[2] / "scripts/research/splitinv_rehearsal.py"
    )
    check = runpy.run_path(str(script))["check_reply_len"]
    row = {"finish": "stop", "ct": 68}
    with pytest.raises(RuntimeError):
        check(row, 256, "reply")
    check(row, 256, "reply", min_reply_tokens=64)
    for invalid in (
        {"finish": "stop", "ct": 63},
        {"finish": "length", "ct": 0},
        {"finish": None, "ct": 68},
        {"finish": "stop", "ct": 257},
    ):
        with pytest.raises(RuntimeError):
            check(invalid, 256, "reply", min_reply_tokens=64)
