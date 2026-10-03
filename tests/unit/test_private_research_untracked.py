"""docs/research/ holds private notes, runs and captures; it is gitignored and must never be committed."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_tracked_files_under_docs_research():
    tracked = subprocess.run(
        ["git", "ls-files", "docs/research"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == [], (
        f"private files tracked by git: {tracked} (git rm --cached them)"
    )
