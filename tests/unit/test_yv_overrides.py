"""yv ab --kinds overrides the suite's prompt kinds (HARNESS row 437)."""

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_kinds_flag_overrides_suite_kinds():
    from verify.runner import apply_overrides

    cfg = {"kinds": ["prose", "code"], "ctx": [1024]}
    out = apply_overrides(cfg, Namespace(kinds="code,zhcn"))
    assert out["kinds"] == ["code", "zhcn"]
    assert cfg["kinds"] == ["prose", "code"]  # the suite table is not mutated


def test_kinds_flag_absent_keeps_suite_kinds():
    from verify.runner import apply_overrides

    out = apply_overrides({"kinds": ["prose"]}, Namespace(kinds=None))
    assert out["kinds"] == ["prose"]
