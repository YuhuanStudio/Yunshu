"""Test-only stub for the pin builder's disk and git reads (fullcmp_audit_manifest.build_pins).

Production builds the yunshu pin from the pinned worktree (git rev-parse) and the launched python's
digest, and fails closed when either is wrong or missing. Unit tests use fakes: the tree sha is the
pin's own sha and every digest is a fixed value. The fail-closed behaviour itself is tested in
test_fullcmp_pins.py.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "research"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fullcmp_audit_manifest as fam  # noqa: E402

FAKE_DIGEST = "a" * 64


def stub_pin_io(monkeypatch):
    orig = fam.build_pins

    def build(census, pin, sha_of=None, git_head=None):
        pin = dict(pin)
        pin.setdefault("worktree", "/fake/tree")
        pin.setdefault("venv", "/fake/venv")
        return orig(
            census,
            pin,
            sha_of=lambda p: FAKE_DIGEST,
            git_head=lambda w: pin["sha"],
        )

    monkeypatch.setattr(fam, "build_pins", build)
    return functools.partial(build)
