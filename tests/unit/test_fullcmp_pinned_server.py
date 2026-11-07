"""Pinned-server verification: the version check and the live command-line check are separate, so the
dry run can skip only the command line. CPU only, fake process command lines."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_provenance as fp  # noqa: E402

PIN = {"version": "0.1.7.dev0", "python": "/venv/bin/python"}
GOOD_STATUS = {"version": "0.1.7.dev0"}


def test_cmdline_matches_the_pinned_module_form():
    fp.verify_pinned_cmdline("/venv/bin/python -m yunshu_cli serve -m m --port 1", PIN)


def test_cmdline_from_another_python_is_refused():
    with pytest.raises(fp.ProvenanceError, match="not the pinned python"):
        fp.verify_pinned_cmdline("/other/python -m yunshu_cli serve", PIN)


def test_cmdline_without_the_module_is_refused():
    with pytest.raises(fp.ProvenanceError, match="-m yunshu_cli"):
        fp.verify_pinned_cmdline("/venv/bin/python serve.py", PIN)


def test_missing_cmdline_is_refused():
    with pytest.raises(fp.ProvenanceError, match="not the pinned python"):
        fp.verify_pinned_cmdline("", PIN)
    with pytest.raises(fp.ProvenanceError, match="not the pinned python"):
        fp.verify_pinned_cmdline(None, PIN)


def test_version_match_and_mismatch():
    fp.verify_pinned_status(GOOD_STATUS, PIN)
    with pytest.raises(fp.ProvenanceError, match="reports version"):
        fp.verify_pinned_status({"version": "0.1.6"}, PIN)
    with pytest.raises(fp.ProvenanceError, match="reports version"):
        fp.verify_pinned_status(None, PIN)


def test_full_check_still_requires_both_halves():
    fp.verify_pinned_server(GOOD_STATUS, "/venv/bin/python -m yunshu_cli serve", PIN)
    with pytest.raises(fp.ProvenanceError):
        fp.verify_pinned_server(
            {"version": "0.1.6"}, "/venv/bin/python -m yunshu_cli", PIN
        )
    with pytest.raises(fp.ProvenanceError):
        fp.verify_pinned_server(GOOD_STATUS, "/venv/bin/python serve.py", PIN)
