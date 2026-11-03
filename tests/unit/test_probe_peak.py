"""CPU-only tests for scripts/research/probe_peak.py: pure helpers only (no MLX, no GPU).

The timing loop is driven by a fake op and a fake clock; the schema is checked on synthetic reports.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import probe_peak as pp  # noqa: E402


def test_module_does_not_import_mlx():
    # the helpers and their tests must run on a machine with no GPU stack
    import inspect

    src = inspect.getsource(pp)
    top = src.split("def measure(")[0]
    assert "import mlx" not in top


def test_stats_odd_and_even_median():
    s = pp.stats([3.0, 1.0, 2.0])
    assert s["median"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0 and s["n"] == 3
    assert s["spread_pct"] == pytest.approx((3 - 1) / 2 * 100)
    e = pp.stats([1.0, 2.0, 3.0, 4.0])
    assert e["median"] == 2.5


@pytest.mark.parametrize(
    "bad", [[], [1.0, 0.0], [1.0, -2.0], [1.0, float("nan")], [float("inf")]]
)
def test_stats_rejects_bad_samples(bad):
    with pytest.raises(ValueError):
        pp.stats(bad)


def test_time_loop_warmup_untimed_and_iters_timed():
    calls = {"op": 0, "eval": 0}
    ticks = iter(range(0, 1000))
    clock_vals = []

    def clock():
        v = next(ticks) * 0.5  # each clock read advances 0.5 s
        clock_vals.append(v)
        return v

    def op():
        calls["op"] += 1
        return calls["op"]

    def ev(_):
        calls["eval"] += 1

    durs = pp.time_loop(op, ev, warmup=2, iters=3, clock=clock)
    assert calls == {"op": 5, "eval": 5}  # 2 warm-up + 3 timed, eval every iteration
    assert len(durs) == 3
    assert all(
        d == pytest.approx(0.5) for d in durs
    )  # only timed iterations read the clock


def test_time_loop_rejects_zero_iters():
    with pytest.raises(ValueError):
        pp.time_loop(lambda: 1, lambda _: None, warmup=0, iters=0, clock=lambda: 0.0)


def test_rate_and_non_positive_duration():
    assert pp.rate(1e9, [0.5, 0.25]) == [2e9, 4e9]
    with pytest.raises(ValueError):
        pp.rate(1.0, [0.0])


def _result(name="bw", median=10.0, values=(9.0, 10.0, 11.0)):
    return pp.make_result(name, "GB/s", list(values))


def test_make_report_schema_roundtrip(tmp_path):
    rep = pp.make_report(
        [_result("a"), _result("b", values=(1.0, 2.0, 3.0))],
        mlx_version="0.0.0",
        chip="Apple M5 Max",
        date="2026-11-03",
    )
    assert rep["label"] == "measured peak on this machine (M5 Max)"
    assert rep["results"][0]["spread_pct"] == pytest.approx((11 - 9) / 10 * 100)
    p = tmp_path / "peak.json"
    p.write_text(json.dumps(rep))
    back = json.loads(p.read_text())
    pp.validate_report(back)


def _good():
    return pp.make_report(
        [_result("a")], mlx_version="0.0.0", chip="c", date="2026-11-03"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("mlx_version"),
        lambda r: r.pop("chip"),
        lambda r: r.__setitem__("label", "spec sheet"),
        lambda r: r.__setitem__("results", []),
        lambda r: r["results"][0].pop("spread_pct"),
        lambda r: r["results"][0].__setitem__("n", 5),
        lambda r: r["results"][0].__setitem__("median", 99.0),  # outside [min, max]
        lambda r: r["results"].append(copy.deepcopy(r["results"][0])),  # duplicate name
    ],
)
def test_validate_rejects_broken_reports(mutate):
    rep = _good()
    mutate(rep)
    with pytest.raises(ValueError):
        pp.validate_report(rep)


def test_chip_falls_back_to_unknown(monkeypatch):
    def boom(*a, **k):
        raise OSError("no sysctl")

    monkeypatch.setattr(pp.subprocess, "run", boom)
    assert pp._chip() == "unknown"
