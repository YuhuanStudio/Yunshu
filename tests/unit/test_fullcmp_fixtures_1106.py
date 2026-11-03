"""Real-log fixtures and negatives for the concurrent verdicts (wire-1106c/d). The oMLX rc1 mixed log is copied
into tests/unit/fixtures/ (read only by these tests); the synthetic negatives pin the fail-closed rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_engaged as fe  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "omlx_rc1_mixed_1106d.server.txt"
# the 8 mixed requests in the order their chat lines were printed (prompt lengths from the log)
MIXED_PTS = [1024, 1024, 1024, 8204, 8204, 1024, 1024, 1024]


def _item(engine="omlx-0.7.3rc1"):
    return next(e for e in fe.load_table()["engines"][engine]["modes"]["dflash"]["evidence"]
                if e["kind"] == "acceptance_regex" and e.get("group") == "accepted")


def _reqs(pts=MIXED_PTS, ct=64):
    return [{"rid": i, "pt": p, "ct": ct} for i, p in enumerate(pts)]


def test_real_omlx_rc1_mixed_matches_eight_requests_with_record_pt_ct():
    """Real log: 3 warm-up/ping chat lines then 8 mixed chat lines; each request matches its own chat line
    (prompt length and completion count exact), accepted from its merged MTP group."""
    log = FIX.read_text(errors="replace")
    matches, why = fe.correlate_multiset(_item(), log, _reqs(), 0, skip=3)
    assert why is None, why
    assert len(matches) == 8
    assert [m["rid"] for m in matches] == list(range(8))
    assert [m["ct_merged"] for m in matches] == [64, 58, 59, 64, 32, 29, 49, 47]
    assert [m["partial_coverage"] for m in matches] == [False, False, False, False, True, True, True, True]


def test_real_omlx_rc1_mixed_partial_coverage_label_explains_the_gap():
    log = FIX.read_text(errors="replace")
    conc = [{"part": "mixed", "n": 8, "requests": _reqs()}]
    ev, problems = fe.concurrent_check(fe.load_table(), "omlx-0.7.3rc1", "dflash", log, conc, 0)
    assert problems == [], problems
    assert ev["engaged_mode"] == "dflash"
    assert "oMLX drafts at <=2 active rows" in ev["label"]
    assert "% of tokens decoded without drafting" in ev["label"]


def test_real_omlx_rc1_mixed_wrong_record_pt_fails_closed():
    """The record's pt must equal the chat line's prompt length: a wrong pt for one request fails the cell."""
    log = FIX.read_text(errors="replace")
    bad = _reqs(pts=[1024, 1024, 1024, 8192, 8204, 1024, 1024, 1024])  # 8192 is not a logged prompt length
    ev, problems = fe.concurrent_check(fe.load_table(), "omlx-0.7.3rc1", "dflash", log,
                                       [{"part": "mixed", "n": 8, "requests": bad}], 0)
    assert problems and ev["engaged_mode"] == "unverified"


def test_omlx_unknown_finish_kind_fails_closed():
    """An unparsed finish kind is not silently dropped: the cell fails and the line is quoted."""
    log = ("2026-11-04 - omlx.patches.mlx_lm_mtp.batch_generator - INFO - MTP[1] finish=weird-new-kind tokens=10 "
           "cycles=2 tok/cycle=5.00 accept=6/7 (85.7%)\n")
    raw = [{"uid": "1", "finish": "weird-new-kind", "ct": 10, "value": 6.0}]
    merged = fe._merge_segments(raw)
    assert merged[0].get("bad") and "weird-new-kind" in merged[0]["bad"]


def test_omlx_uid_without_terminal_segment_is_a_partial_group_not_a_failure():
    """A uid may end without a terminal MTP line (its tokens were decoded while MTP was inactive): its
    segments are a partial group; the Chat completion line is the completion record."""
    raw = [{"uid": "4", "finish": "late-join-handoff", "ct": 18, "value": 13.0},
           {"uid": "4", "finish": "batch-standard-calibration", "ct": 9, "value": 2.0}]
    merged = fe._merge_segments(raw)
    assert len(merged) == 1 and merged[0].get("partial") and not merged[0].get("bad")
    assert merged[0]["ct"] == 27 and merged[0]["calibration_tokens"] == 9


def test_omlx_all_non_terminal_kinds_merge_into_the_terminal_request_regardless_of_interleaving():
    raw = [{"uid": "8", "finish": "batch-standard-calibration", "ct": 13, "value": 6.0},
           {"uid": "9", "finish": "late-join-handoff", "ct": 15, "value": 8.0},
           {"uid": "8", "finish": "length", "ct": 16, "value": 6.0},
           {"uid": "9", "finish": "batch-standard-calibration", "ct": 11, "value": 4.0},
           {"uid": "9", "finish": "length", "ct": 18, "value": 5.0}]
    merged = {m["uid"]: m for m in fe._merge_segments(raw)}
    assert merged["8"]["ct"] == 29 and merged["8"]["calibration_tokens"] == 13
    assert merged["9"]["ct"] == 44 and merged["9"]["calibration_tokens"] == 11
    assert merged["9"]["segments"] == 3


def test_mlx_lm_ar_conc_passes_and_an_ar_engine_with_drafting_fails():
    conc = [{"part": "conc", "n": 2, "requests": [{"rid": 0, "pt": 1024, "ct": 64},
                                                  {"rid": 1, "pt": 1024, "ct": 64}]}]
    ev, problems = fe.concurrent_check(fe.load_table(), "mlx-lm-server", "ar", "", conc, 0)
    assert problems == [] and ev["engaged_mode"] == "ar"
    ev, problems = fe.concurrent_check(fe.load_table(), "mlx-lm-server", "ar",
                                       "[spec-stats] mode=mtp attempts=3 accepts=2\n", conc, 0)
    assert problems and ev["engaged_mode"] == "unverified"


def test_mlx_vlm_probe_batch_level_label_and_single_probe_not_used():
    conc = [{"part": "conc", "n": 2, "requests": [{"rid": 0, "pt": 1024, "ct": 64},
                                                  {"rid": 1, "pt": 1024, "ct": 64}]}]
    same = {"timings": {"draft_kind": "dflash", "draft_rounds": 152, "draft_n": 300, "draft_n_accepted": 100,
                        "predicted_per_second": 9.9}}
    ev, problems = fe.concurrent_check(None, "mlx-vlm-server", "dflash", "", conc, 0,
                                       probe_responses=[same, same])
    assert problems == [] and ev["engaged_mode"] == "dflash"
    assert "batch-level counters; acceptance 33%" in ev["label"]
    assert ev["stats"].get("batch_level") is True
    # no concurrent probe (only the sequential single probe exists): unverified, the single probe is not used
    ev, problems = fe.concurrent_check(None, "mlx-vlm-server", "dflash", "", conc, 0, probe_responses=None)
    assert problems == ["no concurrent probe with per-response stats"] and ev["engaged_mode"] == "unverified"
    zero = {"timings": {"draft_n": 0, "draft_n_accepted": 0}}
    ev, problems = fe.concurrent_check(None, "mlx-vlm-server", "dflash", "", conc, 0, probe_responses=[zero, zero])
    assert problems == [] and ev["engaged_mode"] == "ar" and ev["label"].startswith("ar (")


def test_yunshu_concurrent_without_counters_and_without_rounds_is_derived_ar():
    conc = [{"part": "conc", "n": 2, "requests": [{"rid": 0, "pt": 1024, "ct": 64, "spec": "unavailable (x)"},
                                                  {"rid": 1, "pt": 1024, "ct": 64, "spec": "unavailable (x)"}]}]
    log = "[yunshu] DFlash round profile: 1 rounds\n[yunshu] DFlash round profile: 1 rounds\n"
    ev, problems = fe.concurrent_check(None, "yunshu", "dflash", log, conc, 0)
    assert problems == [] and ev["engaged_mode"] == "ar"
    assert ev["label"].startswith("ar (dev0")
