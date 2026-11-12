"""ple_wait_report parses fn_ple lines into per-request spans and per-bucket wait summaries."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/research/ple_wait_report.py"
_SPEC = importlib.util.spec_from_file_location("ple_wait_report", _PATH)
rep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rep)

LOG = """noise
INFO fn_ple span=0:2048 ple_wait=900.0 ms rows=100 prefetched=0
INFO fn_ple span=2048:4096 ple_wait=40.0 ms rows=100 prefetched=100
INFO fn_ple span=4096:6144 ple_wait=60.0 ms rows=100 prefetched=100
INFO fn_ple span=6144:8192 ple_wait=20.0 ms rows=100 prefetched=50
INFO fn_ple span=0:512 ple_wait=5.0 ms rows=10 prefetched=0
"""


def test_parse_splits_requests_at_span_zero():
    reqs = rep.parse(LOG)
    assert [len(r) for r in reqs] == [4, 1]
    assert reqs[0][1] == dict(
        start=2048, end=4096, wait_ms=40.0, rows=100, prefetched=100
    )


def test_summary_separates_the_first_span_and_skips_unbucketed_prompts():
    table = rep.summarize(rep.parse(LOG))
    assert list(table) == [8192]  # the 512-token request is in no bucket
    row = table[8192]
    assert row["first_span_ms"] == 900.0
    assert row["later_median_ms"] == 40.0 and row["later_max_ms"] == 60.0
    assert abs(row["prefetched_share"] - 250 / 400) < 1e-9
