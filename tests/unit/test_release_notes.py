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
- Current changes [bench](docs/BENCHMARKS.md#results)
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
    assert 'uv tool install --upgrade --python 3.13 "yunshu[vision]==0.1.3"' in notes
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
    notes = MODULE.render_notes("## [0.1.0]\n- First\n", "0.1.0")
    assert "/commits/v0.1.0" in notes


def test_dev_release_takes_the_unreleased_section():
    """User 2026-11-10: every push is a dev release (0.1.8.devN) whose notes are [Unreleased]."""
    changelog = """## [Unreleased]
- Future changes
## [0.1.7] - 2026-11-10
- Current changes
"""
    notes = MODULE.render_notes(changelog, "0.1.8.dev1")
    assert "Future changes" in notes and "Current changes" not in notes
    assert '"yunshu[vision]==0.1.8.dev1"' in notes
    assert "/compare/v0.1.7...v0.1.8.dev1" in notes
    assert "brew upgrade" not in notes  # the tap tracks stable releases


def test_empty_headings_are_dropped_and_a_section_without_entries_fails():
    """2026-11-11: v0.1.8.dev1 shipped empty Highlights/Performance/Fixed headings."""
    changelog = """## [Unreleased]
Intro.
### Highlights

### Performance

| a | b |
|---|---|

### Added

- Real entry
### Fixed

## [0.1.7]
- old
"""
    notes = MODULE.render_notes(changelog, "0.1.8.dev2")
    assert "### Added" in notes and "- Real entry" in notes
    for empty in ("### Highlights", "### Performance", "### Fixed", "| a | b |"):
        assert empty not in notes
    with pytest.raises(ValueError, match="no entries"):
        MODULE.render_notes(
            "## [Unreleased]\nIntro.\n### Fixed\n## [0.1.7]\n- x\n", "0.1.8.dev2"
        )


def test_a_later_dev_release_lists_only_what_is_new_since_the_previous_one():
    """User 2026-11-12: dev2's notes repeated everything dev1 had already shipped."""
    from scripts.release.notes import previous_dev, render_notes

    dev1 = """# Changelog

## [Unreleased]

### Performance

| Machine | Metric | Before -> after |
|---|---|---|
| M5 | old row | 1 -> 2 |

### Fixed

- Old fix shipped in dev1.

## [0.1.7] - 2026-11-10

### Fixed

- Older.
"""
    dev2 = dev1.replace(
        "- Old fix shipped in dev1.\n",
        "- Old fix shipped in dev1.\n- New fix in dev2.\n",
    ).replace(
        "| M5 | old row | 1 -> 2 |\n",
        "| M5 | old row | 1 -> 2 |\n| M5 | new row | 3 -> 4 |\n",
    )
    notes = render_notes(dev2, "0.1.8.dev2", dev1)
    assert (
        previous_dev("0.1.8.dev2") == "0.1.8.dev1"
        and previous_dev("0.1.8.dev1") is None
    )
    assert "Changes since 0.1.8.dev1" in notes
    assert "- New fix in dev2." in notes and "Old fix shipped in dev1" not in notes
    assert "| M5 | new row | 3 -> 4 |" in notes and "old row" not in notes
    assert "| Machine | Metric | Before -> after |" in notes
    assert "compare/v0.1.8.dev1...v0.1.8.dev2" in notes
