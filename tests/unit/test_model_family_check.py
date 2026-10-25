"""scripts/research/model_family_check.py: the per-model judge fails closed (CPU only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import model_family_check as mf  # noqa: E402


def _ok():
    return {n: {"status": "pass"} for n in mf.CELLS}


def test_all_pass():
    assert mf.judge(_ok()) == []


def test_missing_and_failed_cells_are_problems():
    cells = _ok()
    del cells["tools"]
    cells["chat"] = {"status": "fail", "detail": "empty"}
    assert mf.judge(cells) == ["chat: empty", "tools: missing"]


def test_untested_needs_a_reason_and_load_must_pass():
    cells = _ok()
    cells["vision"] = {"status": "untested"}
    assert mf.judge(cells) == ["vision: untested without a reason"]
    cells["vision"] = {"status": "untested", "detail": "text only"}
    assert mf.judge(cells) == []
    cells["load"] = {"status": "untested", "detail": "x"}
    assert mf.judge(cells) == ["load: did not pass"]
