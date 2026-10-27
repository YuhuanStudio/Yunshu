"""CPU checks for the served-path round probe's parsers (fail closed on missing evidence)."""

import pytest
from scripts.research.roundgap_run import parse_tf_done, round_summary

LINE = (
    "[tensorfold] done req-a prompt=947 cached=0 thinking=False effort=none tokens=256 sha=0ec96a4ee0b0 "
    "finish=length tok/s=145.9 ttft=1.06s prefill=1.05s rounds=35 accepted=220/557 ms/round=49.4 "
    "forward=49.4 draft=0.0 post=0.0 rows=16.9 checkpoints=3"
)


def test_parse_tf_done():
    d = parse_tf_done(LINE)
    assert d and d["rounds"] == 35 and d["accepted"] == 220 and d["rows"] == 16.9
    assert d["ms_per_round"] == 49.4


def test_parse_ignores_other_lines():
    assert parse_tf_done("INFO something") is None


def test_round_summary():
    s = round_summary(256, 35, 1.75)
    assert s["tokens_per_round"] == pytest.approx(255 / 35)
    assert s["wall_ms_per_round"] == pytest.approx(50.0)


@pytest.mark.parametrize("args", [(256, 0, 1.0), (1, 5, 1.0), (256, 5, 0.0)])
def test_round_summary_fails_closed(args):
    with pytest.raises(ValueError):
        round_summary(*args)


def test_parse_tf_done_without_timing_fields():
    line = "[tensorfold] done chatcmpl-x prompt=1034 cached=0 tokens=512 sha=4551c4eae614 finish=length rounds=97 accepted=415/1455"
    d = parse_tf_done(line)
    assert d and d["rounds"] == 97 and d["ms_per_round"] is None
