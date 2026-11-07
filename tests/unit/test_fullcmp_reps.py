"""Unit tests for scripts/research/fullcmp_reps.py (rep policy: 2 reps, automatic rep 3)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_reps", ROOT / "scripts" / "research" / "fullcmp_reps.py"
)
reps = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_reps"] = reps
_SPEC.loader.exec_module(reps)

TH = {"decode_tps": 3.0, "cold_ttft": 5.0, "peak_gib": 5.0}


def rep(decode=100.0, ttft=1.0, mem=None):
    row = {"decode_tps": decode, "cold_ttft": ttft}
    if mem is not None:
        row["peak_gib"] = mem
    return row


# --- within threshold ------------------------------------------------------
def test_two_reps_within_threshold_is_ok_with_median_value():
    out = reps.decide([rep(100.0), rep(101.0)], TH)  # 1% gap on tok/s, under 3%
    assert out["status"] == "ok"
    assert out["flags"] == []
    assert out["exceeding"] == []
    assert out["value"]["decode_tps"] == pytest.approx(100.5)
    assert out["per_metric"]["decode_tps"]["values"] == [100.0, 101.0]


# --- gap > threshold -> need_rep3 -----------------------------------------
def test_two_reps_gap_over_threshold_needs_rep3():
    out = reps.decide([rep(100.0), rep(104.0)], TH)  # 4% > 3%
    assert out["status"] == "need_rep3"
    assert out["exceeding"] == ["decode_tps"]
    assert out["value"] is None


def test_gap_is_relative_to_the_smaller_value():
    # |a-b| / min(a,b): 103 vs 100 is 3.0 %, exactly at the threshold, so not over it.
    out = reps.decide([rep(100.0), rep(103.0)], TH)
    assert out["per_metric"]["decode_tps"]["gap_pct"] == pytest.approx(3.0)
    assert out["status"] == "ok"


# --- after rep 3: median + flag -------------------------------------------
def test_after_rep3_within_2x_threshold_gives_median_and_spread_flag():
    # tok/s gap 4 % after two reps; rep 3 settles at 101, spread 4 % (between 1x and 2x of 3 %)
    out = reps.decide([rep(100.0), rep(104.0), rep(101.0)], TH)
    assert out["status"] == "spread_flag"
    assert out["flags"] == ["3 reps (spread)"]
    assert out["value"]["decode_tps"] == pytest.approx(101.0)  # median of 100, 104, 101
    assert out["exceeding"] == ["decode_tps"]


def test_after_rep3_within_threshold_still_flagged_as_three_reps():
    out = reps.decide([rep(100.0), rep(104.0), rep(102.0)], TH)
    # 100/104/102: gap 4 % > 3 %, so this is spread_flag, not ok
    assert out["status"] == "spread_flag"
    assert out["value"]["decode_tps"] == pytest.approx(102.0)


# --- still > 2x threshold -> unstable keeps all values --------------------
def test_after_rep3_still_over_2x_threshold_is_unstable_with_all_values():
    # tok/s gap 10 % > 2 x 3 % = 6 %
    out = reps.decide([rep(100.0), rep(110.0), rep(105.0)], TH)
    assert out["status"] == "unstable"
    assert out["flags"] == ["3 reps (spread)", "unstable"]
    assert out["value"]["decode_tps"] == [100.0, 110.0, 105.0]
    assert out["value"]["cold_ttft"] == [1.0, 1.0, 1.0]


# --- memory fixed 5 % ------------------------------------------------------
def test_memory_uses_fixed_five_percent():
    ok = reps.decide([rep(mem=10.0), rep(mem=10.4)], TH)  # 4 % < 5 %
    assert ok["status"] == "ok"
    need = reps.decide([rep(mem=10.0), rep(mem=10.6)], TH)  # 6 % > 5 %
    assert need["status"] == "need_rep3"
    assert need["exceeding"] == ["peak_gib"]
    # 11 % after rep 3 is over 2 x 5 %: unstable
    bad = reps.decide([rep(mem=10.0), rep(mem=10.6), rep(mem=11.5)], TH)
    assert bad["status"] == "unstable"


# --- metrics with different thresholds ------------------------------------
def test_each_metric_is_judged_against_its_own_threshold():
    # tok/s gap 4 % (threshold 3 %) exceeds; TTFT gap 4 % (threshold 5 %) does not
    out = reps.decide([rep(100.0, 1.0), rep(104.0, 1.04)], TH)
    assert out["status"] == "need_rep3"
    assert out["exceeding"] == ["decode_tps"]
    assert out["per_metric"]["cold_ttft"]["exceeds"] is False
    assert out["per_metric"]["cold_ttft"]["threshold_pct"] == 5.0


def test_metric_missing_in_some_reps_raises():
    with pytest.raises(ValueError):
        reps.decide([rep(100.0), {"decode_tps": 100.0}], TH)


def test_bad_rep_count_raises():
    with pytest.raises(ValueError):
        reps.decide([rep(100.0)], TH)


# --- rep-3 order slot rotation --------------------------------------------
def test_rep3_order_rotates_by_two():
    order = ["a", "b", "c", "d"]
    assert reps.rep3_order(order) == ["c", "d", "a", "b"]
    assert reps.rep3_order(order, rep=0) == order


def test_rep3_slot_differs_from_rep0_and_rep1_for_groups_of_three_or_more():
    order = ["a", "b", "c", "d", "e"]
    for eng in order:
        slots = [reps.rotate(order, r).index(eng) for r in (0, 1, 2)]
        assert len(set(slots)) == 3, (eng, slots)


# --- thresholds file and derivation ---------------------------------------
def test_derive_thresholds_floor_and_three_times_max():
    samples = {
        "decode_tps": [0.2, 0.3],
        "cold_ttft": [0.1, 0.2, 0.4],
        "warm_ttft": [12.0, 4.0, 1.0],
    }
    body = reps.derive_thresholds(samples, sources=["x"])
    th = body["thresholds_pct"]
    assert th["decode_tps"] == 3.0  # max(3 x 0.3, 3) = floor
    assert th["cold_ttft"] == pytest.approx(max(3 * 0.4, 5.0))  # floor 5 wins
    assert th["warm_ttft"] == pytest.approx(36.0)  # 3 x 12 %, three samples
    assert body["metrics"]["warm_ttft"]["low_confidence"] is False
    assert body["metrics"]["cold_ttft"]["low_confidence"] is False
    assert th["peak_gib"] == 5.0


def test_low_confidence_uses_floor_only():
    body = reps.derive_thresholds({"decode_tps": [9.0]}, sources=[])
    assert body["thresholds_pct"]["decode_tps"] == 3.0
    assert body["metrics"]["decode_tps"]["low_confidence"] is True


def test_thresholds_file_round_trip(tmp_path):
    body = reps.derive_thresholds({"decode_tps": [0.5, 0.6, 0.7]}, sources=[])
    path = tmp_path / "rep_thresholds.json"
    path.write_text(json.dumps(body))
    loaded = reps.thresholds_from_file(path)
    assert loaded["decode_tps"] == body["thresholds_pct"]["decode_tps"]
    assert loaded["peak_gib"] == 5.0


# --- CLI over an audit-layout results dir ---------------------------------
def _write_cell(
    root: Path, engine: str, cell: str, rep_no: int, decode: float, group="g1"
):
    d = root / engine
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "part": "session",
            "group": group,
            "engine": engine,
            "cell": cell,
            "rep": rep_no,
        },
        {
            "part": "decode",
            "phase": "cold",
            "dec_tps": decode,
            "ttft_s": 1.0,
            "finish": "length",
        },
    ]
    (d / f"{cell}-r{rep_no}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )


def test_cli_prints_need_rep3_and_exits_zero(tmp_path, capsys):
    _write_cell(tmp_path, "yunshu", "c1", 0, 100.0)
    _write_cell(tmp_path, "yunshu", "c1", 1, 104.0)
    _write_cell(tmp_path, "yunshu", "c2", 0, 50.0)
    _write_cell(tmp_path, "yunshu", "c2", 1, 50.5)
    th = tmp_path / "th.json"
    th.write_text(
        json.dumps(reps.derive_thresholds({"decode_tps": [0.1, 0.2, 0.3]}, sources=[]))
    )
    rc = reps.main(["--results-dir", str(tmp_path), "--thresholds", str(th)])
    assert rc == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert [(x["engine"], x["cell"], x["status"]) for x in lines] == [
        ("yunshu", "c1", "need_rep3")
    ]
    assert lines[0]["exceeding"] == ["decode_tps"]
    assert lines[0]["rep"] == 2


def test_cli_unstable_with_all_values_exits_zero(tmp_path, capsys):
    _write_cell(tmp_path, "yunshu", "c1", 0, 100.0)
    _write_cell(tmp_path, "yunshu", "c1", 1, 110.0)
    _write_cell(tmp_path, "yunshu", "c1", 2, 105.0)
    th = tmp_path / "th.json"
    th.write_text(
        json.dumps(reps.derive_thresholds({"decode_tps": [0.1, 0.2, 0.3]}, sources=[]))
    )
    rc = reps.main(["--results-dir", str(tmp_path), "--thresholds", str(th)])
    assert rc == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert lines[0]["status"] == "unstable"
    assert lines[0]["value"]["decode_tps"] == [100.0, 110.0, 105.0]


def test_cli_unreadable_thresholds_exit_two(tmp_path):
    rc = reps.main(
        ["--results-dir", str(tmp_path), "--thresholds", str(tmp_path / "missing.json")]
    )
    assert rc == 2
