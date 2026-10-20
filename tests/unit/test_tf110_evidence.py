"""Fail closed for missing/short TensorFold round evidence, CPU only."""

import pytest
from scripts.research.tf110_evidence import parse_rounds, round_metrics


def test_cold_and_warm_rounds():
    line = "done prompt=32768 cached=0 thinking=False tokens=2048 sha=abc finish=length rounds=163 accepted=1885/3821"
    rows = parse_rounds(line + "\n" + line.replace("cached=0", "cached=32761"), 32768)
    assert [r["rounds"] for r in rows] == [163, 163]
    assert rows[1]["cached"] == 32761


@pytest.mark.parametrize(
    "log",
    ["", "done prompt=32768 cached=0 tokens=12 finish=stop rounds=1 accepted=11/15"],
)
def test_missing_or_short_rejected(log):
    with pytest.raises(ValueError):
        parse_rounds(log, 32768)


def test_round_wall_not_forward():
    r = round_metrics(2048, 163, 20, 10)
    assert r["outputs_per_round"] == pytest.approx(2047 / 163)
    assert r["wall_ms_per_round"] == pytest.approx(10000 / 163)
    assert "NOT isolated" in r["scope"]


@pytest.mark.parametrize("args", [(1, 1, 2, 1), (2048, 0, 2, 1), (2048, 1, 1, 2)])
def test_invalid_timing_rejected(args):
    with pytest.raises(ValueError):
        round_metrics(*args)


def test_original_scheduler_receipt_overrides_stale_output_flags():
    from scripts.research.tf110_evidence import accepted_receipt

    clean = {"state": "done", "rc": 0, "contended": False, "swap_mb": 0}
    assert accepted_receipt(clean)
    for bad in [
        {"contended": True},
        {"swap_mb": 64.1},
        {"state": "pending"},
        {"rc": 1},
    ]:
        assert not accepted_receipt(dict(clean, **bad))
    assert not accepted_receipt({"state": "done", "rc": 0, "contended": False})
