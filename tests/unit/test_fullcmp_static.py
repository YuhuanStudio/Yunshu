"""Static check: no undefined names (ruff F821) in the full-comparison harness modules.

A missing import in a rarely-run path (smoke5: tfbench conc aggregate, NameError on `statistics`)
must fail CI, not a GPU job. CPU only.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "scripts/research/tfbench.py",
    "scripts/research/fullcmp_run.py",
    "scripts/research/fullcmp_audit.py",
    "scripts/research/fullcmp_telemetry.py",
]


def _ruff():
    for cand in (ROOT / ".venv" / "bin" / "ruff", shutil.which("ruff")):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def test_no_undefined_names_in_harness_modules():
    ruff = _ruff()
    if ruff is None:
        pytest.fail("ruff not found in the worktree venv: the static check cannot run")
    res = subprocess.run(
        [
            ruff,
            "check",
            "--no-cache",
            "--select",
            "F821",
            "--output-format",
            "concise",
            *FILES,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stdout + res.stderr
