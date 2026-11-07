"""The concurrent verdict (fe.concurrent_check) on the REAL wire-1106c conc logs and records, per engine.
Skipped when the raw is not on disk. Parsers that failed on real logs before are pinned here."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_engaged as fe  # noqa: E402

LOGS = "/Volumes/Micron/yunshu-fullcmp/wire-1106c-work/all/tfout/out/server-{eng}-conc-0.log"
RECS = "/Volumes/Micron/yunshu-fullcmp/wire-1106c/{eng}/*.jsonl"


def _real(eng):
    logs = glob.glob(LOGS.format(eng=eng))
    recs = [r for r in sorted(glob.glob(RECS.format(eng=eng))) if "telemetry" not in r]
    if not logs or not recs:
        pytest.skip(f"no real wire-1106c conc raw for {eng}")
    return Path(logs[0]).read_text(errors="replace"), recs


def _conc_cells(recs):
    for jl in recs:
        rows = [json.loads(x) for x in Path(jl).read_text().splitlines() if x.strip()]
        header = next((r for r in rows if r.get("part") == "session"), None)
        conc = [r for r in rows if r.get("part") in ("conc", "mixed")]
        if header and conc:
            yield jl, header, conc


def test_omlx_real_conc_merges_the_handoff_segments_into_dflash():
    """oMLX splits a running request at each join (late-join-handoff + final segment): 7 real MTP lines
    after the warm-up and the two pings are 4 requests, each drafting (accept > 0), matched by order."""
    log, recs = _real("omlx-0.7.3")
    checked = list(_conc_cells(recs))
    assert checked
    for jl, header, conc in checked:
        ev, problems = fe.concurrent_check(
            fe.load_table(),
            "omlx-0.7.3",
            header["spec_mode_expected"],
            log,
            conc,
            header.get("server_log_ready_offset"),
        )
        assert problems == [], (jl, problems)
        assert ev["engaged_mode"] == "dflash"
        assert len(ev["stats"]["per_row"]) == sum(len(r["requests"]) for r in conc)


def test_omlx_rc1_real_conc_is_dflash():
    log, recs = _real("omlx-0.7.3rc1")
    for jl, header, conc in _conc_cells(recs):
        ev, problems = fe.concurrent_check(
            fe.load_table(),
            "omlx-0.7.3rc1",
            header["spec_mode_expected"],
            log,
            conc,
            header.get("server_log_ready_offset"),
        )
        assert problems == [] and ev["engaged_mode"] == "dflash", (jl, problems)


def test_mlx_lm_real_conc_is_ar_and_prints_no_drafting_marker():
    """AR engine: its real conc log holds no drafting line, so the cell is 'ar' (completion only)."""
    from fullcmp_engaged import AR_DRAFT_MARKERS, ar_drafting_line

    log, recs = _real("mlx-lm-server")
    assert AR_DRAFT_MARKERS.search(log) is None, (
        "mlx-lm prints a drafting marker: the AR verdict would fail"
    )
    assert ar_drafting_line(log) is None
    for jl, header, conc in _conc_cells(recs):
        ev, problems = fe.concurrent_check(
            fe.load_table(),
            "mlx-lm-server",
            header["spec_mode_expected"],
            log,
            conc,
            header.get("server_log_ready_offset"),
        )
        assert problems == [] and ev["engaged_mode"] == "ar", (jl, problems)


def test_ar_engine_whose_log_shows_drafting_fails():
    """Negative: an AR engine (mlx-lm) whose log shows a per-request acceptance line fails (mode mismatch)."""
    log = "[mlx-lm] request done\n[spec-stats] mode=mtp attempts=3 accepts=2\n"
    conc = [{"part": "conc", "n": 1, "requests": [{"rid": 0, "pt": 1024, "ct": 64}]}]
    ev, problems = fe.concurrent_check(
        fe.load_table(), "mlx-lm-server", "ar", log, conc, 0
    )
    assert problems and ev["engaged_mode"] == "unverified"
    assert "drafting" in problems[0]


def test_kapitsa_real_conc_four_measured_lines_after_the_pings():
    """kapitsa lines carry no timestamps: the two readiness pings are the first two chat lines (prompt 15),
    the four measured lines (prompt 1024) follow. Each request matches its own line."""
    log, recs = _real("kapitsa-0.2.26")
    for jl, header, conc in _conc_cells(recs):
        ev, problems = fe.concurrent_check(
            fe.load_table(),
            "kapitsa-0.2.26",
            header["spec_mode_expected"],
            log,
            conc,
            header.get("server_log_ready_offset"),
        )
        assert problems == [], (jl, problems)
        assert ev["engaged_mode"] == "dflash"
        assert len(ev["stats"]["per_row"]) == 4


def test_kapitsa_dropping_one_measured_line_fails_closed():
    log, recs = _real("kapitsa-0.2.26")
    lines = [ln for ln in log.splitlines(keepends=True) if "chat: prompt" in ln]
    assert len(lines) >= 6, "two pings and four measured lines"
    cut = log.replace(lines[-1], "", 1)
    for _jl, header, conc in _conc_cells(recs):
        ev, problems = fe.concurrent_check(
            fe.load_table(),
            "kapitsa-0.2.26",
            header["spec_mode_expected"],
            cut,
            conc,
            header.get("server_log_ready_offset"),
        )
        assert problems and ev["engaged_mode"] == "unverified"
        break


def test_mlx_vlm_concurrent_probe_both_outcomes_and_mixed_on_synthetic_probe_rows():
    """The wire-1106c mlx-vlm raw predates the concurrent probe (no probe_conc rows): the verdict is tested on
    synthetic probe responses. The streamed lines in its log do not break the probe path."""
    conc = [
        {
            "part": "conc",
            "n": 2,
            "requests": [
                {"rid": 0, "pt": 1024, "ct": 64},
                {"rid": 1, "pt": 1024, "ct": 64},
            ],
        }
    ]
    drafting = [
        {"timings": {"draft_n": 40, "draft_n_accepted": 30}},
        {"timings": {"draft_n": 38, "draft_n_accepted": 25}},
    ]
    zero = [
        {"timings": {"draft_n": 0, "draft_n_accepted": 0}},
        {"timings": {"draft_rounds": None}},
    ]
    mixed = [
        {"timings": {"draft_n": 40, "draft_n_accepted": 30}},
        {"timings": {"draft_n": 0, "draft_n_accepted": 0}},
    ]
    ev, p = fe.concurrent_check(
        None, "mlx-vlm-server", "dflash", "", conc, 0, probe_responses=drafting
    )
    assert p == [] and ev["engaged_mode"] == "dflash"
    ev, p = fe.concurrent_check(
        None, "mlx-vlm-server", "dflash", "", conc, 0, probe_responses=zero
    )
    assert p == [] and ev["engaged_mode"] == "ar" and ev["label"].startswith("ar (")
    ev, p = fe.concurrent_check(
        None, "mlx-vlm-server", "dflash", "", conc, 0, probe_responses=mixed
    )
    assert p and ev["engaged_mode"] == "unverified"
    # the streamed lines of a real run (chat/done lines of any kind) do not change the probe verdict
    streamed = "[mlx-vlm] Request completed: stream=True generated_tokens=64 decode=40.0 tok/s\n"
    ev, p = fe.concurrent_check(
        None, "mlx-vlm-server", "dflash", streamed, conc, 0, probe_responses=drafting
    )
    assert p == [] and ev["engaged_mode"] == "dflash"


def test_yunshu_sequential_wire_1105b_kapitsa_path_still_engages():
    """The sequential per-row path on the wire-1105b kapitsa raw (regression guard for the item selection)."""
    jls = [
        j
        for j in sorted(
            glob.glob(
                "/Volumes/Micron/yunshu-fullcmp/wire-1105b/kapitsa-0.2.26/*-r0.jsonl"
            )
        )
        if "telemetry" not in j
    ]
    if not jls:
        pytest.skip("no wire-1105b kapitsa raw")
    jl = jls[0]
    rows = [json.loads(x) for x in Path(jl).read_text().splitlines() if x.strip()]
    header = next(r for r in rows if r.get("part") == "session")
    decode = [r for r in rows if r.get("part") == "decode"]
    log = (
        Path(jl)
        .with_name(Path(jl).name[: -len(".jsonl")] + ".server.log")
        .read_text(errors="replace")
    )
    ev = fe.evaluate(
        "kapitsa-0.2.26",
        "dflash",
        log,
        decode,
        fe.load_table(),
        ready_offset=header.get("server_log_ready_offset"),
    )
    assert ev["engaged_mode"] == "dflash", ev["stats"]
