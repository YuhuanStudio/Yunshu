"""harness_identity guard (fullcmp_audit.check_harness_identity): a real header without a harness commit
fails; the e2e stand-in (conftest) is the only thing that supplies one. Synthetic headers, no run."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit as fa  # noqa: E402


def _run(header):
    cell = {"cell": "SMOKE.smoke-core.yunshu.1K.c1.prose.o64.single.th-off.none.greedy.v0", "engine": "yunshu"}
    cf = SimpleNamespace(cell=cell, rep=0, path=Path("/x/c-r0.jsonl"), header=header, rows=[])
    return SimpleNamespace(files=[cf])


def test_real_header_without_harness_commit_fails():
    res = fa.check_harness_identity(_run({"harness_dirty": False}))
    assert any("no harness_commit" in v for v in res.violations)


def test_header_with_commit_and_clean_tree_passes():
    res = fa.check_harness_identity(_run({"harness_dirty": False, "harness_commit": "a" * 40}))
    assert res.violations == []
