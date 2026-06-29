"""Report thermal-suspect demotion (scripts/report.py _analyze).

A real code regression hits SPECIFIC KPIs; a hot/memory-pressured machine depresses
EVERY model's cache numbers at once. The end-of-run snapshot tag can read cool even when
the cache section ran hot mid-run, so the per-KPI cool-gate alone let those whole-snapshot
thermal depressions masquerade as 🔴 regressions. _analyze must demote a uniform
cross-model (≥3 models) cache depression in one snapshot into `thermal_suspect`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


def _snap(ts, tflops, cache_vals):
    kpis = {"gpu_tflops": tflops}
    for model, dec in cache_vals.items():
        kpis[f"cache/{model}/decode_tps"] = dec
    return {"timestamp": ts, "kpis": kpis}


def test_uniform_cross_model_cache_depression_is_demoted():
    import report

    # Two cool baselines (healthy), then one snapshot that READS cool (8.7 ≥ 8.5 floor)
    # but depresses ALL three models' decode together — the thermal-ghost signature.
    snaps = [
        _snap("20260101T000000Z", 9.0, {"m-a": 100.0, "m-b": 50.0, "m-c": 26.0}),
        _snap("20260102T000000Z", 9.0, {"m-a": 100.0, "m-b": 50.0, "m-c": 26.0}),
        _snap("20260103T000000Z", 8.7, {"m-a": 66.0, "m-b": 33.0, "m-c": 16.0}),
    ]
    a = report._analyze(snaps)
    regr_kpis = {f["kpi"] for f in a["flags"] if not f["improved"]}
    suspect_kpis = {f["kpi"] for f in a["thermal_suspect"]}
    # None of the uniformly-depressed cache decode KPIs should be a hard regression…
    assert not any(
        k.startswith("cache/") and k.endswith("decode_tps") for k in regr_kpis
    )
    # …they're all in the thermal-suspect bucket instead.
    assert suspect_kpis == {
        "cache/m-a/decode_tps",
        "cache/m-b/decode_tps",
        "cache/m-c/decode_tps",
    }
    assert "20260103T000000Z" in a["suspect_ts"]


def test_single_model_cache_regression_stays_a_real_flag():
    import report

    # Only ONE model regresses (the others hold) → NOT the uniform thermal signature,
    # so it must remain a genuine 🔴 regression, not be demoted.
    snaps = [
        _snap("20260101T000000Z", 9.0, {"m-a": 100.0, "m-b": 50.0, "m-c": 26.0}),
        _snap("20260102T000000Z", 9.0, {"m-a": 100.0, "m-b": 50.0, "m-c": 26.0}),
        _snap("20260103T000000Z", 9.0, {"m-a": 60.0, "m-b": 50.0, "m-c": 26.0}),
    ]
    a = report._analyze(snaps)
    regr_kpis = {f["kpi"] for f in a["flags"] if not f["improved"]}
    assert "cache/m-a/decode_tps" in regr_kpis
    assert not a["thermal_suspect"]
