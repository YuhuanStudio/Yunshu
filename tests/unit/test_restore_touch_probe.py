"""The probe's dry run must not import MLX or start a server."""

import json
import subprocess
import sys
from pathlib import Path


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
