"""The round driver's synced stage timer and its YUNSHU_LAYER2 line (2026-11-09).

Before: the Phase B round driver printed no stage split at all, so every fullcmp profile rep of a
Yunshu engine reported stages.decode_round.drafter / commit as MISSING.
"""

import importlib.util
import logging
import sys
from pathlib import Path

from yunshu_engine.round_driver.stage_profile import RoundProfile

ROOT = Path(__file__).resolve().parents[2]


def _profile_module():
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    spec = importlib.util.spec_from_file_location(
        "fullcmp_profile_t", ROOT / "scripts/research/fullcmp_profile.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _round(p, rid, attention):
    p.begin()
    for stage in ("copy_lookup", "drafter", "plan", "verify", "commit"):
        p.mark(stage)
    p.end([rid], attention)


def test_off_is_a_no_op():
    p = RoundProfile(on=False)
    _round(p, 1, 0.5)
    assert p.finish(1) is None


def test_a_request_logs_every_decode_stage_once(caplog):
    p = RoundProfile(on=True)
    for _ in range(3):
        _round(p, 7, 0.0)
    with caplog.at_level(logging.INFO):
        payload = p.finish(7)
    stages = payload["stages"]["decode_round"]
    assert payload["rounds"] == 3
    assert set(stages) == {
        "drafter",
        "plan",
        "copy_lookup",
        "verify_weights",
        "verify_attention",
        "commit",
    }
    assert all(v >= 0 for v in stages.values())
    assert p.finish(7) is None  # logged once
    lines = [
        r.getMessage() for r in caplog.records if "YUNSHU_LAYER2" in r.getMessage()
    ]
    assert len(lines) == 1


def test_attention_never_exceeds_the_verify_time():
    p = RoundProfile(on=True)
    _round(p, 1, 1e9)
    stages = p.finish(1)["stages"]["decode_round"]
    assert stages["verify_weights"] == 0.0


def test_a_packed_round_reports_the_verify_total_only():
    p = RoundProfile(on=True)
    _round(p, 1, None)
    stages = p.finish(1)["stages"]["decode_round"]
    assert "verify_total" in stages and "verify_attention" not in stages


def test_the_harness_reads_the_engine_line_through_the_logger_prefix():
    fp = _profile_module()
    line = (
        "2026-11-09 10:00:00,000 INFO yunshu_engine.round_driver.stage_profile: YUNSHU_LAYER2 "
        '{"rounds": 2, "stages": {"decode_round": {"drafter": 1.0, "commit": 2.0}}}'
    )
    got = fp.parse_yunshu_layer2(line)
    assert got["stages"]["decode_round"]["drafter"] == 1.0
