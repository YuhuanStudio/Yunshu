import importlib.util
from pathlib import Path

import pytest


def _mod():
    p = Path(__file__).resolve().parents[2] / "scripts/research/turn2gap.py"
    s = importlib.util.spec_from_file_location("turn2gap", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_parse_cases():
    m = _mod()
    assert m.parse_cases("imm,gap") == ["imm", "gap"]
    with pytest.raises(ValueError):
        m.parse_cases("imm,nope")


def test_profile_summary():
    import json

    p = Path(__file__).resolve().parents[2] / "scripts/research/turn2gap_prof.py"
    s = importlib.util.spec_from_file_location("t2p", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    d = dict(
        start=0,
        rows=5,
        cached=[0],
        wall_s=1.0,
        attributed_s=0.5,
        method="x",
        timeline=[],
        spans={
            "gemm:M5:K1:N1:b4": dict(exclusive_s=0.3, count=2),
            "attention:M5": dict(exclusive_s=0.2, count=1),
        },
    )
    r = m.summarise("x COLDTTFT_PROFILE " + json.dumps(d))
    assert r[0]["groups"] == {"gemm": 0.3, "attention": 0.2}
