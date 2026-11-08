"""Stage timers of a cold Flash-Next request: line format, accumulation, one line per request."""

import logging
import re

from yunshu_engine import fn_ttft

LINE = re.compile(
    r"fn_ttft path=(driver|lane) ids=\d+ hit=\d+ segs=\d+ \| (\w+=[\d.]+ )+other=-?[\d.]+ total=[\d.]+ ms$"
)


def test_timer_charges_stage_and_formats_one_line():
    t = fn_ttft.Timer()
    t.segs = 2
    t.mark("submit")
    t.mark("wait")
    line = t.line("driver", 100, 0)
    assert LINE.match(line), line
    for k in fn_ttft.STAGES:
        assert f"{k}=" in line


def test_merge_sums_stages_and_spans_steps():
    a, b = fn_ttft.Timer(), fn_ttft.Timer()
    a.stage["wait"], b.stage["wait"] = 0.5, 0.25
    a.segs = b.segs = 1
    a.merge(b)
    assert a.stage["wait"] == 0.75 and a.segs == 2
    assert LINE.match(a.line("lane", 1, 0, total_s=1.0))


def test_driver_logs_one_line_per_cold_request(caplog):
    from tests.unit.test_round_driver_flashnext import PROMPTS, make_tiny, run

    lm, head = make_tiny()
    with caplog.at_level(logging.INFO, logger="yunshu_engine.fn_ttft"):
        run(lm, [PROMPTS[0]], head=head)
    lines = [r.message for r in caplog.records if r.message.startswith("fn_ttft")]
    assert len(lines) == 1 and LINE.match(lines[0]), lines
