"""yv probe: log parsers on real excerpts, verdict logic, base cache reuse."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from verify import probe  # noqa: E402
from verify.suites import SUITES, parse_suite  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "yv_probe"


def _rows(cts, tps=50.0):
    return [
        {
            "part": "decode",
            "phase": "cold",
            "ctx": 1024,
            "kind": f"k{i}",
            "ct": ct,
            "dec_tps": tps,
            "ttft_s": 1.0,
        }
        for i, ct in enumerate(cts)
    ]


def test_parse_driver_log():
    p = probe.parse_log((FX / "driver_rows.log").read_text())
    assert p["fmt"] == "driver"
    assert len(p["requests"]) == 4 and p["requests"][2]["tpr"] == 3.40
    assert p["requests"][2]["hist"][:3] == [6, 14, 25]
    assert len(p["windows"]) == 3
    assert probe.arm_ms(p) == pytest.approx((59.7 + 60.7 + 63.4) / 3)


def test_parse_lane_log_derives_tokens_per_round():
    text = (FX / "lane_profile.log").read_text()
    p = probe.parse_log(text)
    assert p["fmt"] == "lane" and len(p["requests"]) == 4
    assert probe.arm_ms(p) == pytest.approx(
        (66 * 49.89 + 66 * 49.79 + 71 * 49.22 + 35 * 52.0) / 238, rel=0.02
    )
    s = probe.summarize(text, _rows([264, 264, 284, 140]))
    assert s["cells"][0]["tpr"] == pytest.approx(4.0)


def test_no_counter_line_fails_closed():
    with pytest.raises(ValueError):
        probe.parse_log("INFO engine started\nnothing here\n")
    with pytest.raises(ValueError):  # fewer counter records than decode rows
        probe.join_rows(
            probe.parse_log((FX / "lane_profile.log").read_text()), _rows([1] * 9)
        )


def _summ():
    return probe.summarize((FX / "driver_rows.log").read_text(), _rows([255, 255]))


def test_identical_arms_pass_and_moves_are_flagged():
    b = _summ()
    assert probe.compare(b, copy.deepcopy(b))["ok"]
    c = copy.deepcopy(b)
    c["cells"][0]["nodes"] += 5.0
    r = probe.compare(b, c)
    assert not r["ok"] and "nodes/round" in r["moved"][0]
    assert "run the speed stage" in probe.verdict_text(r)
    c = copy.deepcopy(b)
    c["cells"][1]["hist"][0] += 1
    assert "histogram" in probe.compare(b, c)["moved"][0]
    c = copy.deepcopy(b)
    c["ms_per_round"] *= 1.05
    assert "ms/round" in probe.compare(b, c)["moved"][0]
    c["ms_per_round"] = b["ms_per_round"] * 1.02  # inside 3 %
    assert probe.compare(b, c)["ok"]
    assert not probe.compare(b, c, {"ms_pct": 1.0})["ok"]


def test_cross_format_regression_visible():
    lane = probe.summarize(
        (FX / "lane_profile.log").read_text(), _rows([264, 264, 284, 140])
    )
    drv = _summ()
    drv["cells"] = drv["cells"][:2] + [
        dict(c, cell=f"k{i}@1024:cold") for i, c in enumerate(drv["cells"][:2])
    ]
    for i, c in enumerate(drv["cells"]):
        c["cell"] = lane["cells"][i]["cell"]
    r = probe.compare(lane, drv)
    assert not r["ok"] and any("nodes/round" in m for m in r["moved"])


def test_cache_roundtrip(tmp_path):
    assert probe.cache_load(tmp_path, "k") is None
    s = _summ()
    probe.cache_store(tmp_path, "k", s)
    assert probe.cache_load(tmp_path, "k") == s
    (probe.cache_path(tmp_path, "k")).write_text("{bad")
    assert probe.cache_load(tmp_path, "k") is None


def test_probe_is_first_stage_after_preflight():
    for name in ("quick", "decode"):
        assert parse_suite(name)["stages"][:2] == ["preflight", "probe"]
    assert "probe" in SUITES and parse_suite("probe")["stages"] == [
        "preflight",
        "probe",
    ]


# ── TTFT stages (cold / warm / turn-2 client TTFT + the engine's per-request stage lines) ──
def _ttft_arm(log, cold_s, warm_s, ids=1034):
    rows = [
        {"part": "decode", "phase": ph, "kind": "code", "ctx": 1024, "prompt_tokens": ids,
         "ttft_s": t, "ct": 256}
        for ph, t in (("cold", cold_s), ("warm", warm_s))
    ]  # fmt: skip
    st = probe.join_stages(probe.parse_stage_log((FX / log).read_text()), rows)
    cells = [
        {"cell": f"code@1024:{r['phase']}", "tpr": 3.0, "nodes": 10.0, "rounds": 50,
         "hist": None, "dec_tps": None, "ttft_s": r["ttft_s"], "stages": s}
        for r, s in zip(rows, st)  # noqa: B905
    ]  # fmt: skip
    return {"fmt": "driver", "ms_per_round": 50.0, "cells": cells, "stage_lines": True}


def test_parse_real_stage_lines_and_join_skips_warmup_requests():
    d = probe.parse_stage_log((FX / "stages_driver.log").read_text())
    assert len(d["turn"]) == 6 and len(d["fn"]) == 6
    assert d["turn"][3]["stages"]["add"] == 71.3 and d["turn"][3]["hit"] == 1033
    assert d["fn"][2]["path"] == "driver" and d["fn"][2]["stages"]["submit"] == 728.4
    lane = probe.parse_stage_log((FX / "stages_lane.log").read_text())
    assert lane["turn"] == [] and lane["fn"][2]["path"] == "lane"
    cand = _ttft_arm("stages_driver.log", 1.43, 0.068)
    # the two ids=15 warmup requests are skipped: cold = the first 1034 record, warm = the next
    assert cand["cells"][0]["stages"]["fn"]["total"] == 1424.3
    assert cand["cells"][1]["stages"]["turn"]["stages"]["add"] == 71.3
    assert cand["cells"][1]["stages"]["fn"]["total"] == 3.6


def test_todays_warm_regression_is_caught_and_attributed():
    base = _ttft_arm("stages_lane.log", 1.437, 0.026)  # main's lane
    cand = _ttft_arm("stages_driver.log", 1.430, 0.068)  # the driver build: warm +42 ms
    r = probe.compare(base, cand)
    assert not r["ok"]
    warm = [m for m in r["moved"] if "warm" in m]
    assert any("client TTFT 26 -> 68 ms" in m for m in warm), r["moved"]
    # the lane prints no 'turn stages', so the culprit stage comes from the candidate's own line
    assert any("'add': 71.3" in m for m in warm), r["moved"]
    assert not any("cold" in m and "client TTFT" in m for m in r["moved"])


def test_stage_move_is_named_when_both_arms_print_it():
    base = _ttft_arm("stages_driver.log", 1.43, 0.026)
    cand = copy.deepcopy(base)
    cand["cells"][1]["stages"]["turn"] = {
        **cand["cells"][1]["stages"]["turn"],
        "stages": {**cand["cells"][1]["stages"]["turn"]["stages"], "add": 3.6},
    }
    base["cells"][1]["stages"]["turn"]["stages"]["add"] = (
        71.3  # base slow, cand fast: not a regression
    )
    assert probe.compare(base, cand)["ok"]
    cand["cells"][1]["stages"]["turn"]["stages"]["add"] = (
        120.0  # cand slower than base's 71.3
    )
    r = probe.compare(base, cand)
    assert any("turn stage add 71.3 -> 120.0" in m for m in r["moved"])


def test_ttft_inside_the_floors_passes_and_tolerances_are_overridable():
    base = _ttft_arm("stages_driver.log", 1.43, 0.026)
    cand = copy.deepcopy(base)
    cand["cells"][0]["ttft_s"] = 1.43 + 0.015  # +15 ms, +1 %: inside both floors
    assert probe.compare(base, cand)["ok"]
    cand["cells"][0]["ttft_s"] = 1.43 + 0.060  # +60 ms, +4.2 %: beyond both
    assert not probe.compare(base, cand)["ok"]
    assert probe.compare(base, cand, {"ttft_pct": 10.0})["ok"]


def test_summary_without_stage_lines_is_marked():
    s = probe.summarize((FX / "driver_rows.log").read_text(), _rows([255, 255]))
    assert s["stage_lines"] is False and s["stage_schema"] == 1
    assert probe.parse_stage_log("INFO nothing\n") == {"turn": [], "fn": []}


def test_arm_without_round_counters_still_compares_ttft_stages():
    """2026-11-08 proof run (Flash-Next, main's MTP lane prints no round counters): the probe stopped at
    'no counter evidence' and never compared the fn_ttft stages both arms print."""
    lane = (FX / "stages_lane.log").read_text()
    s = probe.summarize(lane, _rows([255, 255]))
    assert s["fmt"] == "none" and s["stage_lines"]
    r = probe.compare(s, copy.deepcopy(s))
    assert r["ok"], r["moved"]
    with pytest.raises(ValueError):
        probe.summarize("INFO nothing\n", _rows([1]))


def test_older_lane_log_with_unpaired_profiles_is_nocount():
    """2026-11-08 push-gate: the base build printed profile lines for only some requests (plus 1-round
    warmups), so the last-N join paired a 256-token reply with a 1-round record ('tokens/round 256.00')."""
    text = (FX / "lane_profile_partial.log").read_text()
    rows = _rows([256] * 6)
    with pytest.raises(ValueError):
        probe.join_rows(probe.parse_log(text), rows)
    s = probe.summarize(text + "\n" + (FX / "stages_lane.log").read_text(), rows)
    assert s["fmt"] == "none"
    assert probe.compare(s, copy.deepcopy(s))["ok"]
