"""CPU checks for the Phase B HTTP measurement and its identity verdict."""

import pytest
from scripts.research import phaseb_conc as p


def test_plan_and_cli_are_cpu_only():
    a = p.parser().parse_args(
        ["--base-src", "/a", "--cand-src", "/b", "--out", "/unused"]
    )
    assert list(map(len, p.validate(a))) == [2, 4, 2]
    a.tokens = 1
    with pytest.raises(ValueError):
        p.validate(a)


def test_identity_fails_closed():
    with pytest.raises(RuntimeError):
        p.check_identity({"A": {"digest": "same"}}, {"A": {"digest": "changed"}})
    with pytest.raises(RuntimeError):
        p.check_identity({"A": {"digest": "same"}}, {})
    p.check_identity({"A": {"digest": "same"}}, {"A": {"digest": "same"}})


def test_metrics_include_running_row_during_prefill():
    a = dict(times=list(range(21)), ttft=0, usage={"completion_tokens": 21}, digest="a")
    b = dict(
        t0=5, times=[9, 10, 11], ttft=4, usage={"completion_tokens": 3}, digest="b"
    )
    assert p.interval_rates(a, b) == dict(
        prefill_tps=1.0, overlap_tps=1.0, after_tps=1.0
    )
    summary = p.summarize({"A": a, "B": b})
    assert summary["per_row"]["A"]["tps"] == 1.0
    assert summary["aggregate_tps"] == 1.2
