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
