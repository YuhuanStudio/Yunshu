"""Release-note extraction must not leak adjacent sections or stale compare links."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "release_notes", ROOT / "scripts/release/notes.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_notes_are_versioned_and_bounded():
    changelog = """## [Unreleased]
Future changes
## [0.1.3] - 2026-10-03
Current changes [bench](docs/BENCHMARKS.md#results)
[Full changelog: draft](https://example.com/main)
## [0.1.2] - 2026-10-02
Older changes
"""
    notes = MODULE.render_notes(changelog, "0.1.3")
    assert "Current changes" in notes
    assert "Future changes" not in notes and "Older changes" not in notes
    assert "example.com" not in notes
    assert "/blob/v0.1.3/docs/BENCHMARKS.md#results" in notes
    assert 'python -m pip install --upgrade "yunshu[vision]==0.1.3"' in notes
    assert 'uv tool install --upgrade "yunshu[vision]==0.1.3"' in notes
    assert "brew upgrade yuhuanstudio/tap/yunshu" in notes
    assert notes.count("/compare/v0.1.2...v0.1.3") == 1


@pytest.mark.parametrize("version", ["0.1.4", "0.1.3; echo bad", "v0.1.3"])
def test_missing_or_invalid_version_fails(version):
    with pytest.raises(ValueError):
        MODULE.render_notes("## [0.1.3]\nChanges\n", version)


def test_empty_release_fails():
    with pytest.raises(ValueError):
        MODULE.render_notes("## [0.1.3]\n## [0.1.2]\nOlder\n", "0.1.3")


def test_first_release_links_to_tag_history():
    notes = MODULE.render_notes("## [0.1.0]\nFirst\n", "0.1.0")
    assert "/commits/v0.1.0" in notes
