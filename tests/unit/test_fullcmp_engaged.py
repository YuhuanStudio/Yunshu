"""fullcmp engaged mode is derived from evidence only (scripts/research/fullcmp_engaged.py)."""

from __future__ import annotations

import copy
import glob
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_engaged as fe  # noqa: E402

MERGED = ROOT / "scripts" / "research" / "fullcmp_engaged_evidence.json"
READY_LINE = "Drafter ready; speculative decoding enabled.\n"
REQ_LOG = "spec: request drafted\n"


def _ev(kind, pat, per_request=False, startup_sufficient=False):
    return {
        "kind": kind,
        "pattern_or_path": pat,
        "source_ref": "synthetic",
        "real_example": "",
        "negative_example": "",
        "per_request": per_request,
        "startup_sufficient": startup_sufficient,
    }


def _table():
    return {
        "_meta": {"synthetic": True},
        "engines": {
            "eng": {
                "modes": {
                    "dflash": {
                        "evidence": [
                            _ev("log_regex", r"Drafter ready; speculative decoding enabled\."),
                            _ev("log_regex", r"spec: request drafted", per_request=True),
                            _ev("acceptance_stat", "attr.accepted", per_request=True),
                        ],
                        "fallback": [r"falls back to AR"],
                    },
                    "ar": {"evidence": [_ev("log_regex", r"Starting httpd", startup_sufficient=True)]},
                },
            },
            "metrics-eng": {
                "modes": {"mtp": {"evidence": [_ev("metrics", r"spec_enabled 1", per_request=True)]}}
            },
        },
    }


def _startup(text):
    return text, len(text.encode())


def test_per_request_log_line_after_ready_is_engaged():
    log = READY_LINE
    ready = len(log.encode())
    log += REQ_LOG
    res = fe.evaluate("eng", "dflash", log, [], _table(), ready_offset=ready)
    assert res["engaged_mode"] == "dflash"
    assert res["evidence_kind"] == "log_regex"
    assert res["source"] == "slice"
    assert res["matched_line"] == "spec: request drafted"


def test_startup_only_line_is_loaded_not_engaged():
    log = READY_LINE
    res = fe.evaluate("eng", "dflash", log, [], _table(), ready_offset=len(log.encode()))
    assert res["engaged_mode"] == "unverified"
    assert res["evidence_kind"] == "startup_only"
    assert res["source"] == "startup_log"
    assert res["matched_line"] == "Drafter ready; speculative decoding enabled."


def test_startup_sufficient_line_engages_from_startup_log():
    res = fe.evaluate("eng", "ar", "Starting httpd at 127.0.0.1 on port 1...\n", [], _table(), ready_offset=10**6)
    assert res["engaged_mode"] == "ar"
    assert res["source"] == "startup_log"


def test_no_evidence_is_unverified():
    res = fe.evaluate("eng", "dflash", "nothing here\n", [], _table(), ready_offset=0)
    assert res["engaged_mode"] == "unverified"
    assert res["evidence_kind"] is None


def test_fallback_line_is_ar_even_with_positive_evidence():
    log = REQ_LOG + "warning: falls back to AR\n"
    res = fe.evaluate("eng", "dflash", log, [], _table(), ready_offset=0)
    assert res["engaged_mode"] == "ar"
    assert res["evidence_kind"] == "fallback"


def test_acceptance_stat_engages_only_when_positive():
    ok = fe.evaluate("eng", "dflash", "", [{"attr": {"accepted": 2}}], _table(), ready_offset=0)
    assert ok["engaged_mode"] == "dflash" and ok["source"] == "acceptance_stat"
    zero = fe.evaluate("eng", "dflash", "", [{"attr": {"accepted": 0}}], _table(), ready_offset=0)
    assert zero["engaged_mode"] == "unverified"


def test_metrics_evidence_reads_the_probe_text():
    on = fe.evaluate("metrics-eng", "mtp", "", [], _table(), probe_text="spec_enabled 1\n", ready_offset=0)
    off = fe.evaluate("metrics-eng", "mtp", "", [], _table(), probe_text="spec_enabled 0\n", ready_offset=0)
    assert on["engaged_mode"] == "mtp" and on["source"] == "metrics"
    assert off["engaged_mode"] == "unverified"


def test_wrong_or_missing_mode_entry_is_unverified():
    assert fe.evaluate("eng", "mtp", REQ_LOG, [], _table(), ready_offset=0)["engaged_mode"] == "unverified"
    assert fe.evaluate("nope", "dflash", REQ_LOG, [], _table(), ready_offset=0)["engaged_mode"] == "unverified"


def test_acceptance_problem_when_all_stats_zero_or_absent():
    m = _table()["engines"]["eng"]["modes"]["dflash"]
    assert fe.acceptance_problem("dflash", m, [{"attr": {"accepted": 1}}]) is None
    assert fe.acceptance_problem("dflash", m, [{"attr": {"accepted": 0}}, {}]) is not None
    assert fe.acceptance_problem("dflash", m, []) is not None
    assert fe.acceptance_problem("ar", m, [{}]) is None


def test_validate_rejects_bad_entries():
    bad = copy.deepcopy(_table())
    bad["engines"]["eng"]["modes"]["dflash"]["evidence"][0]["source_ref"] = ""
    bad["engines"]["eng"]["modes"]["dflash"]["evidence"][1]["pattern_or_path"] = "(unclosed"
    bad["engines"]["metrics-eng"]["modes"]["mtp"]["evidence"][0]["kind"] = "prose"
    bad["engines"]["eng"]["modes"]["warp"] = {"evidence": [_ev("log_regex", "x")]}
    bad["engines"]["eng"]["modes"]["off"] = {"evidence": []}
    bad["engines"]["eng"]["modes"]["mtp"] = {"evidence": [_ev("response_field", "not a path!")]}
    problems = "\n".join(fe.validate_table(bad))
    for needle in ("source_ref", "bad regex", "kind", "not in", "non-empty", "dotted field path"):
        assert needle in problems, needle
    assert fe.validate_table(_table()) == []
    assert fe.validate_table({"engines": []})  # not an object of engines


def test_load_table_is_none_when_missing_or_malformed(tmp_path):
    assert fe.load_table(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert fe.load_table(broken) is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_table()))
    assert fe.load_table(good)["engines"]["eng"]["modes"]["dflash"]


def test_merged_table_is_the_default_and_validates():
    assert Path(fe.DEFAULT_TABLE) == MERGED
    table = fe.load_table()
    assert table is not None, "fullcmp_engaged_evidence.json must load through the schema"
    assert fe.validate_table(table) == []
    assert "engaged_evidence_tf_splash.json" in table["_meta"]["sources"]
    for engine, entry in table["engines"].items():
        for mode, m in entry["modes"].items():
            assert m["evidence"], f"{engine}/{mode} has no evidence"
            for ev in m["evidence"]:
                assert ev["source_ref"], f"{engine}/{mode} evidence without source_ref"


def test_merged_table_positive_regex_matches_a_real_splash_log():
    """Real-log check (skipped when the logs are not on disk): splash's per-request line is in a
    spec-on run's own log, so the splash dflash entry can engage on a request."""
    logs = glob.glob("/Volumes/Micron/yunshu-fullcmp/wire-1103/splash-1.5.0/*.server.log")
    if not logs:
        pytest.skip("no real splash server log on disk")
    table = fe.load_table()
    m = table["engines"]["splash-1.5.0"]["modes"]["dflash"]
    text = Path(logs[0]).read_text(errors="replace")
    res = fe.evaluate("splash-1.5.0", "dflash", text, [], table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", m


def test_srv_refuses_only_at_the_final_check(monkeypatch):
    """The startup check records; only the final check (after the requests) may refuse a cell."""
    import tfbench

    class FakeLog:
        def __init__(self, text):
            self.text = text

        def read_text(self, errors="replace"):
            return self.text

    def make(text):
        srv = tfbench.Srv.__new__(tfbench.Srv)
        srv.engine = "eng"
        srv.requested_spec_mode = "dflash"
        srv.probe = None
        srv.ready_log_offset = 0
        srv.log = FakeLog(text)
        return srv

    monkeypatch.setattr(fe, "load_table", _table)
    srv = make(READY_LINE)
    srv._verify_fullcmp_evidence(READY_LINE, final=False)  # startup: loaded only, no refusal
    assert srv.engaged_spec_mode == "unverified"
    with pytest.raises(RuntimeError, match="engaged=unverified"):
        srv._verify_fullcmp_evidence(READY_LINE, final=True)
    srv = make(REQ_LOG)
    srv._verify_fullcmp_evidence(REQ_LOG, final=True, rows=[])  # per-request line: engaged, no error
    assert srv.engaged_spec_mode == "dflash" and srv.engaged_evidence_source == "slice"
    monkeypatch.setattr(fe, "load_table", lambda: None)  # no table: no evidence at all
    with pytest.raises(RuntimeError, match="table: missing"):
        make(REQ_LOG)._verify_fullcmp_evidence(REQ_LOG, final=True)


def test_splash_startup_slice_is_not_empty(tmp_path):
    """Regression (2026-11-04): the slice began at readiness, so splash-1.5.0 cells had an empty
    .server.log slice. The cell slice is the whole server log from byte 0."""
    import tfbench

    import bench_engines as be

    launch = be.build_launch("splash-1.5.0", 18990, tmp_path / "home", {}, 140000, 1)
    assert "serve" in launch.cmd
    log = tmp_path / "server.log"
    log.write_bytes(READY_LINE.encode())  # startup only: printed before readiness
    cell = tfbench.cell_engine_log(log, log.stat().st_size)
    assert cell, "the cell's engine-log slice must not be empty"
    assert cell == READY_LINE.encode()


# Every engine has a configured mode (launch spec or audit pin); none is exempt.
UNCONFIGURED: set[str] = set()


def test_table_keys_equal_the_manifest_engine_set():
    import fullcmp_manifest as fm

    table = fe.load_table()
    manifest = set(fm.ENGINES) - {"ollama", "lmstudio", fm.PLACEHOLDER}
    assert set(table["engines"]) == manifest
    for engine, entry in table["engines"].items():
        assert "modes" in entry and entry["confidence"], engine
        if engine in UNCONFIGURED:
            assert entry["modes"] == {}
            continue
        assert entry["modes"], f"{engine}: no mode"
        for mode, m in entry["modes"].items():
            assert m["evidence"], f"{engine}/{mode}: no evidence item"


def test_every_entry_regex_runs_on_the_real_logs_and_prints_the_line():
    """Each log_regex / acceptance_regex item is run on the real server logs on disk. A real log
    that matches must give a non-null real_example (the matched line); an engine with no real log
    says so. Output: one line per engine (run pytest -s to see it)."""
    import fullcmp_engaged_merge as mg

    table = fe.load_table()
    for engine, entry in sorted(table["engines"].items()):
        logs = mg.real_logs_for(engine)
        if not logs:
            print(f"{engine}: no real log on disk (source-only)")
            continue
        texts = {f: Path(f).read_text(errors="replace") for f in logs}
        for mode, m in entry["modes"].items():
            for ev in m["evidence"]:
                if ev["kind"] not in ("log_regex", "acceptance_regex"):
                    continue
                rx = re.compile(ev["pattern_or_path"])
                matched = next(
                    (fe._line_at(t, h.start(), h.end()) for t in texts.values() if (h := rx.search(t))), None
                )
                print(f"{engine}/{mode} {ev['kind']}: {matched or 'NO MATCH'}")
                if matched is not None:
                    assert ev["real_example"], f"{engine}/{mode}: real log matches but real_example is null"
