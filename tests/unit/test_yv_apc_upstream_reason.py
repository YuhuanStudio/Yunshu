"""apc reports the identity failure that stopped it, not "run the identity stage" (2026-11-02:
identity ran and failed, so its evidence was incomplete and apc said it never ran)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify import stages  # noqa: E402
from verify.core import RunDir  # noqa: E402


def _ctx(tmp_path, identity_rows):
    run = RunDir(tmp_path / "run")
    with (run.path / "identity.jsonl").open("w") as f:
        for row in identity_rows:
            f.write(json.dumps(row) + "\n")
    return SimpleNamespace(
        run=run,
        exe=SimpleNamespace(jobs=[]),
        suite={"ctx": [8192], "kinds": ["prose"], "spec_modes": ["default"]},
    )


def test_apc_reports_the_failed_identity_stage(tmp_path):
    failed = {
        "ev": "stage_complete",
        "name": "identity",
        "status": "FAIL",
        "passed": False,
        "reasons": [
            "base != cand: [{'cell': [8192, 'prose', 'turn2'], 'why': 'digest differs'}]"
        ],
    }
    res = stages.stage_apc(_ctx(tmp_path, [failed]))
    assert not res.passed
    text = " ".join(res.reasons)
    assert "identity stage failed" in text
    assert "digest differs" in text
    assert "run the identity stage" not in text


def test_apc_says_identity_was_not_run_when_it_never_ran(tmp_path):
    res = stages.stage_apc(_ctx(tmp_path, []))
    assert not res.passed
    assert "identity stage not run" in " ".join(res.reasons)
