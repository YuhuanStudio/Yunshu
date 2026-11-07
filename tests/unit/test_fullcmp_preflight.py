"""CPU-only tests for the swap-stability preflight (scripts/research/fullcmp_preflight.py).

All series are synthetic: the sampler is a list iterator, sleep and clock are fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_preflight as pf  # noqa: E402
import fullcmp_telemetry as tm  # noqa: E402

MB = 1024 * 1024


def _run(values, **kw):
    it = iter(values)
    clock_t = [0.0]

    def sleep(s):
        clock_t[0] += s

    return pf.swap_stable(
        sample=lambda: next(it),
        sleep=sleep,
        clock=lambda: clock_t[0],
        **kw,
    )


def test_stable_series_passes():
    base = 3607 * MB
    series = [base + (i % 2) * MB for i in range(31)]  # wobbles by 1 MB, no growth
    res = _run(series, window_s=300, interval_s=10)
    assert res["ok"] is True
    assert len(res["series"]) == 31
    assert res["growth_bytes"] == series[-1] - series[0]


def test_growing_series_fails():
    series = [1000 * MB + i * 5 * MB for i in range(31)]  # +150 MB over the window
    res = _run(series, window_s=300, interval_s=10)
    assert res["ok"] is False
    assert res["growth_bytes"] == 150 * MB
    assert res["max_used_bytes"] == series[-1]


def test_growth_exactly_at_limit_passes_and_one_byte_over_fails():
    base = 500 * MB
    at = [base, base + 64 * MB]
    over = [base, base + 64 * MB + 1]
    assert _run(at, window_s=10, interval_s=10)["ok"] is True
    assert _run(over, window_s=10, interval_s=10)["ok"] is False


def test_shrinking_swap_passes():
    series = [900 * MB, 800 * MB, 700 * MB]
    res = _run(series, window_s=20, interval_s=10)
    assert res["ok"] is True
    assert res["growth_bytes"] < 0


def test_sample_error_propagates_fail_closed():
    def boom():
        raise tm.ProbeError("swapusage: unparseable")

    with pytest.raises(tm.ProbeError):
        pf.swap_stable(window_s=10, interval_s=10, sample=boom, sleep=lambda s: None)


def test_bad_window_rejected():
    with pytest.raises(ValueError):
        pf.swap_stable(
            window_s=0, interval_s=10, sample=lambda: 0, sleep=lambda s: None
        )


def test_parse_swapusage_real_format():
    text = "vm.swapusage: total = 5120.00M  used = 3607.06M  free = 1512.94M  (encrypted)\n"
    got = tm.parse_swapusage(text)
    assert got["used_bytes"] == int(3607.06 * MB)
    assert got["total_bytes"] == 5120 * MB


def test_cli_exit_codes(monkeypatch, capsys):
    seq = iter([100 * MB, 100 * MB, 100 * MB])
    monkeypatch.setattr(pf, "read_swap_used_bytes", lambda: next(seq))
    monkeypatch.setattr(pf.time, "sleep", lambda s: None)
    assert pf._cli(["--window-s", "20", "--interval-s", "10"]) == 0
    seq2 = iter([100 * MB, 300 * MB, 500 * MB])
    monkeypatch.setattr(pf, "read_swap_used_bytes", lambda: next(seq2))
    assert pf._cli(["--window-s", "20", "--interval-s", "10"]) == 1
    out = capsys.readouterr().out
    assert '"growth_bytes"' in out and '"series"' in out
