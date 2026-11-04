"""Real-log fixtures and negatives for the concurrent verdicts (wire-1106c/d). The oMLX rc1 mixed log is copied
into tests/unit/fixtures/ (read only by these tests); the synthetic negatives pin the fail-closed rules."""

from __future__ import annotations

import json
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


def test_yunshu_conc_in_job_and_audit_paths_return_the_same_verdict_on_the_real_wire_1107_log():
    """Regression (wire-1107): the in-job check passed the bench id 'yunshu-main' and fell through to the
    acceptance-item path ('no per-request acceptance item'). Both callers (bench id and manifest id) must reach the
    one derivation and return the same verdict on the real log (2 DFlash round lines for 128 tokens -> ar)."""
    log = (ROOT / "tests" / "unit" / "fixtures" / "yunshu_conc_1107.server.txt").read_text(errors="replace")
    conc = [{"part": "conc", "n": 2, "requests": [{"rid": 0, "pt": 1024, "ct": 64, "spec": "unavailable (x)"},
                                                  {"rid": 1, "pt": 1024, "ct": 64, "spec": "unavailable (x)"}]}]
    in_job = fe.concurrent_check(None, "yunshu-main", "dflash", log, conc, 0)
    audit = fe.concurrent_check(None, "yunshu", "dflash", log, conc, 0)
    assert in_job == audit
    assert in_job[1] == [] and in_job[0]["engaged_mode"] == "ar"
    assert in_job[0]["label"].startswith("ar (dev0")


def test_every_fixture_cell_has_identical_in_job_and_audit_verdicts():
    """Generic: for each concurrent fixture cell, the in-job call and the audit call (same function, the bench
    id and the manifest id) return identical (mode, label, problems)."""
    cells = [
        ("yunshu", "yunshu_conc_1107.server.txt",
         [{"part": "conc", "n": 2, "requests": [{"rid": i, "pt": 1024, "ct": 64, "spec": "unavailable (x)"}
                                                for i in range(2)]}]),
    ]
    for eng, name, conc in cells:
        log = (ROOT / "tests" / "unit" / "fixtures" / name).read_text(errors="replace")
        a = fe.concurrent_check(None, eng + "-main" if eng == "yunshu" else eng, "dflash", log, conc, 0)
        b = fe.concurrent_check(None, eng, "dflash", log, conc, 0)
        assert (a[0]["engaged_mode"], a[0].get("label"), a[1]) == (b[0]["engaged_mode"], b[0].get("label"), b[1])


def _alias_pairs():
    import bench_engines as be

    seen = []
    for bench in sorted(be.ENGINES):
        manifest = be.manifest_id(bench)
        if manifest in ("ollama", "lmstudio"):
            continue
        seen.append((bench, manifest))
    return seen


def test_every_bench_and_manifest_id_gives_the_same_shared_verdict_on_the_fixture_cells():
    """The in-job path passes the bench id, the audit the manifest id: for every alias pair the shared verdict
    (mode, label, problems) is identical on each fixture cell (the wire-1107 Yunshu cell and the real oMLX rc1
    mixed cell here; other engines are covered by their own fixture tests)."""
    yun = (ROOT / "tests" / "unit" / "fixtures" / "yunshu_conc_1107.server.txt").read_text(errors="replace")
    omlx = (ROOT / "tests" / "unit" / "fixtures" / "omlx_rc1_mixed_1106d.server.txt").read_text(errors="replace")
    reqs2 = [{"rid": i, "pt": 1024, "ct": 64, "spec": "unavailable (x)"} for i in range(2)]
    cells = {
        "yunshu": (yun, [{"part": "conc", "n": 2, "requests": reqs2}], "dflash"),
        "omlx-0.7.3rc1": (omlx, [{"part": "mixed", "n": 8, "requests": [
            {"rid": i, "pt": p, "ct": 64} for i, p in enumerate([1024, 1024, 1024, 8204, 8204, 1024, 1024, 1024])]}],
            "dflash"),
    }
    for bench, manifest in _alias_pairs():
        for key, (log, conc, mode) in cells.items():
            if fe.manifest_key(bench) != key and manifest != key:
                continue  # only the engine's own id pairs with its fixture
            a = fe.concurrent_check(None, bench, mode, log, conc, 0)
            b = fe.concurrent_check(None, manifest, mode, log, conc, 0)
            assert (a[0]["engaged_mode"], a[0].get("label"), a[1]) == (b[0]["engaged_mode"], b[0].get("label"), b[1]), (
                bench, manifest)


def test_omlx_batch_without_any_mtp_line_is_dflash_zero_percent_drafted():
    """A concurrent batch that ran with no MTP line at all (batch inactive for >=3 rows): the Chat completion
    lines still match the record's pt/ct; the label is the evidence-derived '0% drafted'; not unverified."""
    log = "".join(
        f"2026-11-04 - omlx.server - INFO - Chat completion: model=Qwen3.8-27B-oQ4e-mtp, 64 tokens in 6.1s "
        f"(14.3 tok/s), prompt: {p}, finish_reason=length, max_tokens=64\n"
        for p in [20, 15, 15, 1024, 8192, 1024, 1024])  # 3 warm-up/pings + 4 measured
    conc = [{"part": "mixed", "n": 4, "requests": [{"rid": i, "pt": p, "ct": 64}
                                                   for i, p in enumerate([1024, 8192, 1024, 1024])]}]
    ev, problems = fe.concurrent_check(fe.load_table(), "omlx-0.7.3rc1", "dflash", log, conc, 0)
    assert problems == [], problems
    assert ev["engaged_mode"] == "dflash"
    assert ev["label"] == "dflash (0% drafted: >=3 active rows)"


def test_mlx_vlm_wire_1107_real_concurrent_probe_rows_are_read_and_drafting():
    """Failing-before (wire-1107): the mlx-vlm mixed cell has probe_conc rows whose timings carry no draft_* fields.
    They must be read (not 'no concurrent probe') and the cell is 'ar (concurrent path does not draft)'."""
    import json as _json

    jl = Path("/Volumes/Micron/yunshu-fullcmp/wire-1107/mlx-vlm-server/"
              "SMOKE.smoke-mixed.mlx-vlm-server.8K.c4.code.o64.mixed.th-off.none.greedy.v0-r0.jsonl")
    if not jl.exists():
        pytest.skip("no real wire-1107 mlx-vlm mixed raw on disk")
    rows = [_json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
    probe = [r for r in rows if r.get("part") == "probe_conc"]
    assert probe, "the real raw carries probe_conc rows"
    responses = [{"timings": r.get("timings")} for r in probe]
    recs = [r for r in rows if r.get("part") in ("conc", "mixed")]
    # the probe count is the sum of the records' requests (the mixed cell's two records)
    ev, problems = fe.concurrent_check(None, "mlx-vlm-server", "dflash", "", recs, 0, probe_responses=responses)
    assert "no concurrent probe with per-response stats" not in problems
    assert problems == [], problems
    # the real probes carry draft counters (7 of 8 identical, one differs): read per response -> dflash
    assert ev["engaged_mode"] == "dflash" and ev["stats"]["probe"][0] == (56, 36)
    assert ev["stats"].get("batch_level") is True  # 7 of 8 share the batch snapshot


def test_mlx_vlm_probe_counters_shared_by_all_but_one_response_are_batch_level():
    """The real wire-1107 mlx-vlm probes: 7 of 8 responses share one counter snapshot (batch level). The verdict is
    labelled batch-level and claims no per-request acceptance."""
    import json as _json

    jl = Path("/Volumes/Micron/yunshu-fullcmp/wire-1107/mlx-vlm-server/"
              "SMOKE.smoke-mixed.mlx-vlm-server.8K.c4.code.o64.mixed.th-off.none.greedy.v0-r0.jsonl")
    if not jl.exists():
        pytest.skip("no real wire-1107 mlx-vlm mixed raw on disk")
    rows = [_json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
    responses = [{"timings": r.get("timings")} for r in rows if r.get("part") == "probe_conc"]
    recs = [r for r in rows if r.get("part") in ("conc", "mixed")]
    ev, problems = fe.concurrent_check(None, "mlx-vlm-server", "dflash", "", recs, 0, probe_responses=responses)
    assert problems == []
    assert ev["stats"]["batch_level"] is True
    assert "batch-level counters" in ev["label"]
    assert "per_row" not in ev["stats"] or not ev["stats"]["per_row"]
    assert ev["stats"]["accepted"] == 176


def test_mlx_vlm_all_distinct_probe_counters_stay_per_request():
    responses = [{"timings": {"draft_n": 10 + i, "draft_n_accepted": 5 + i}} for i in range(4)]
    recs = [{"part": "conc", "n": 4, "requests": [{"rid": i, "pt": 1024, "ct": 64} for i in range(4)]}]
    ev, problems = fe.concurrent_check(None, "mlx-vlm-server", "dflash", "", recs, 0, probe_responses=responses)
    assert problems == [] and ev["engaged_mode"] == "dflash"
    assert ev["stats"].get("batch_level") is None


def test_concurrent_ar_verdict_passes_the_configured_dflash_mode_with_its_label_on_the_real_wire_1107y_cells():
    """wire-1107y Yunshu concurrent cells: the evidence says the concurrent path does not draft ('ar (dev0 ...)');
    under the configured speculative mode this is a labelled pass for concurrent cells (single-request cells must
    match the configured mode). Read in place from the wire-1107y raw."""
    import types

    import fullcmp_audit as fa

    names = ["SMOKE.smoke-conc-distinct.yunshu.1K.c2.code.o64.single.th-off.none.greedy.v0-r0.jsonl",
             "SMOKE.smoke-conc-same.yunshu.1K.c2.code.o64.single.th-off.none.greedy.v0-r0.jsonl",
             "SMOKE.smoke-mixed.yunshu.8K.c4.code.o64.mixed.th-off.none.greedy.v0-r0.jsonl"]
    base = Path("/Volumes/Micron/yunshu-fullcmp/wire-1107y/yunshu")
    paths = [base / n for n in names]
    if not all(p.exists() for p in paths):
        pytest.skip("wire-1107y yunshu conc raw not on disk")
    for p in paths:
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        h = next(r for r in rows if r.get("part") == "session")
        cell = {"cell": p.stem, "engine": "yunshu", "spec_mode": h.get("spec_mode_expected"), "group": "g"}
        cf = types.SimpleNamespace(header=h, cell=cell, rep=0, rows=[], path=p)
        res = fa.check_engaged_mode(types.SimpleNamespace(files=[cf]))
        assert not [v for v in res.violations if "engaged evidence says" in v], (p.name, res.violations)
