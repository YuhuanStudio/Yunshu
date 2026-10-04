import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _dry(script, *args):
    out = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts/research" / script),
            "--output",
            "x.jsonl",
            "--dry-run",
            *args,
        ],
        text=True,
    )
    return json.loads(out.strip().splitlines()[-1])


def test_quality_dry_run_schema():
    assert _dry("wide_quality200.py")["paired_items"] == 200


def test_api_probe_dry_run_counts_cells():
    r = _dry(
        "wide_api_probe.py",
        "--contexts",
        "256",
        "1024",
        "--reps",
        "2",
        "--rep-offset",
        "1",
    )
    assert r["complete"] and r["cells"] == 2 * 2 * 2 * 2
