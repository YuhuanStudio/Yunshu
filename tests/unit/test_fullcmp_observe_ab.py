"""CPU-only tests for the observe on/off A/B driver (scripts/research/fullcmp_observe_ab.py).

No server, no GPU: the arm environment, the engagement expectation, the profile parser, the argument
checks and the summary arithmetic (outlier exclusion, overhead sign, thresholds) are exercised on
synthetic records.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_observe_ab as obs  # noqa: E402

THR = {"decode_tps": 3.03, "turn2_decode_tps": 3.0, "cold_ttft": 5.0, "turn2_ttft": 32.93}


def test_arm_on_sets_observe_and_arm_off_removes_it():
    base = {"HOME": "/x", "YUNSHU_OBSERVE": "0", "PATH": "/bin"}
    on = obs.arm_env(base, "B")
    off = obs.arm_env(base, "A")
    assert on["YUNSHU_OBSERVE"] == "1"
    assert "YUNSHU_OBSERVE" not in off
    assert on["HOME"] == off["HOME"] == "/x"
    assert base["YUNSHU_OBSERVE"] == "0"  # input not mutated


def test_expected_debug_status_follows_the_arm():
    assert obs.expected_debug_status("B") == 200  # observe on mounts /debug
    assert obs.expected_debug_status("A") == 404  # observe off: route absent


def test_profile_lines_parse_per_round_ms():
    text = (
        "INFO x: DFlash round profile: 37 rounds, 15.2 nodes/round, draft 6.10 ms, verify 44.50 ms, "
        "commit+host 1.20 ms\n"
        "INFO x: DFlash round profile: 12 rounds, 14.0 nodes/round, draft 6.00 ms, verify 46.50 ms\n"
    )
    got = obs.parse_profiles(text)
    assert [p["verify_ms"] for p in got] == [44.5, 46.5]
    assert got[0]["rounds"] == 37 and got[0]["draft_ms"] == 6.1


def test_no_profile_lines_means_no_profiles():
    assert obs.parse_profiles("plain server log\n") == []


def test_profile_regex_is_shared_with_the_sampler_driver():
    import fullcmp_sampler_ab as ab

    assert obs.ab.PROFILE_RE is ab.PROFILE_RE


def test_bad_sequences_are_rejected_before_any_launch():
    assert obs.run_cmd("code-1024", "BA", 0, "t") == 2  # must start with A (global ABAB order)
    assert obs.run_cmd("code-1024", "ABA", 0, "t") == 2  # must be a pair
    assert obs.run_cmd("code-1024", "AC", 0, "t") == 2  # unknown arm
    assert obs.run_cmd("code-4096", "AB", 0, "t") == 2  # unknown prompt


def _rec(prompt, arm, idx, cold, turn2, ttft_c, ttft_t, verify, quiet=False):
    return {
        "prompt": prompt, "arm": arm, "idx": idx, "tag": "t", "quiet_flag": quiet,
        "cold_dec_tps": cold, "turn2_dec_tps": turn2, "cold_ttft_s": ttft_c,
        "turn2_ttft_s": ttft_t, "verify_ms_round": verify,
    }


def test_summary_overhead_sign_and_within_noise():
    recs = [
        _rec("code-1024", "off", 0, 80.0, 80.0, 1.0, 2.0, 40.0),
        _rec("code-1024", "off", 1, 80.0, 80.0, 1.0, 2.0, 40.0),
        _rec("code-1024", "on", 2, 79.0, 79.0, 1.01, 2.02, 40.5),
        _rec("code-1024", "on", 3, 79.0, 79.0, 1.01, 2.02, 40.5),
    ]
    out = obs.summarize_prompt("code-1024", recs, THR)
    assert out["overhead"]["decode_tps"]["pct"] == 1.25  # observe slower -> positive
    assert out["overhead"]["cold_ttft"]["pct"] == 1.0
    assert out["within_noise"] is True
    assert out["outlier_servers"] == []


def test_slow_outlier_is_counted_and_excluded_from_means():
    recs = [
        _rec("code-1024", "off", 0, 80.0, 80.0, 1.0, 2.0, 40.0),
        _rec("code-1024", "off", 1, 80.0, 80.0, 1.0, 2.0, 41.0),
        _rec("code-1024", "on", 2, 80.0, 80.0, 1.0, 2.0, 40.0),
        _rec("code-1024", "on", 3, 20.0, 20.0, 9.0, 9.0, 139.0),  # verify > 1.5 x median
    ]
    out = obs.summarize_prompt("code-1024", recs, THR)
    assert out["per_arm"]["on"]["outliers"] == 1
    assert out["per_arm"]["on"]["included"] == 1
    assert out["per_arm"]["on"]["decode_cold_tps"]["mean"] == 80.0  # outlier not in the mean
    assert [o["idx"] for o in out["outlier_servers"]] == [3]


def test_real_regression_outside_threshold_is_flagged():
    recs = [
        _rec("code-8192", "off", 0, 80.0, 80.0, 1.0, 2.0, 40.0),
        _rec("code-8192", "off", 1, 80.0, 80.0, 1.0, 2.0, 40.0),
        _rec("code-8192", "on", 2, 70.0, 70.0, 1.0, 2.0, 40.0),
        _rec("code-8192", "on", 3, 70.0, 70.0, 1.0, 2.0, 40.0),
    ]
    out = obs.summarize_prompt("code-8192", recs, THR)
    assert out["overhead"]["decode_tps"]["within"] is False
    assert out["within_noise"] is False


def test_import_does_not_mutate_the_process_environment():
    before = dict(os.environ)
    import importlib

    importlib.reload(obs)
    assert {k: v for k, v in os.environ.items() if k.startswith("YUNSHU_")} == {
        k: v for k, v in before.items() if k.startswith("YUNSHU_")
    }
