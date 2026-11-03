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
    # the real splash log on disk holds only the two readiness pings (prompt 15): they never count
    # for a measured row, so a measured row with that prompt has no done line of its own
    done = [int(x) for x in re.findall(r"Done · input (\d+) ·", text)]
    assert len(done) == 2, "the real splash log holds the readiness pings only"
    rows = [{"pt": 15, "ct": 10}]
    res = fe.evaluate("splash-1.5.0", "dflash", text, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "unverified"
    assert "row 0" in res["stats"]["per_row_problem"], m
    # no measured rows: nothing to correlate, so no per-request evidence
    assert fe.evaluate("splash-1.5.0", "dflash", text, [], table, ready_offset=0)["engaged_mode"] == "unverified"


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


# ---- per measured request correlation (readiness pings never count for a measured row)
TF_LINE = "[tensorfold] done chatcmpl-{i} prompt={p} cached=0 thinking=False tokens=10 finish=stop rounds=1 accepted={a}/15 ttft=0.03\n"


def _tf_table():
    return {
        "_meta": {},
        "engines": {
            "tf": {
                "modes": {
                    "dflash": {
                        "evidence": [
                            _ev(
                                "acceptance_regex",
                                r"\[tensorfold\] done chatcmpl-\S+ prompt=(?P<prompt>\d+) .*? rounds=\d+ accepted=(?P<accepted>\d+)/\d+",
                                per_request=True,
                            )
                            | {"group": "accepted", "min": 0}
                        ],
                        "fallback": [],
                    }
                }
            }
        },
    }


def _log(*lines):
    return "".join(lines)


def test_readiness_ping_drafting_but_measured_request_not_drafting_fails():
    """Ping (prompt 15) accepted>0, the measured request (prompt 1024) accepted=0 -> unverified."""
    log = _log(
        TF_LINE.format(i="a", p=15, a=9),  # readiness ping
        TF_LINE.format(i="b", p=15, a=9),  # readiness ping
        TF_LINE.format(i="c", p=1024, a=0),  # measured: AR after warm-up
    )
    rows = [{"pt": 1024, "ct": 10}]
    res = fe.evaluate("tf", "dflash", log, rows, _tf_table(), ready_offset=0)
    assert res["engaged_mode"] == "unverified"
    assert "done line value" in res["stats"]["per_row_problem"]


def test_correct_per_request_lines_pass_and_report_each_matched_line():
    log = _log(
        TF_LINE.format(i="a", p=15, a=9),
        TF_LINE.format(i="b", p=15, a=9),
        TF_LINE.format(i="c", p=1024, a=5),
        TF_LINE.format(i="d", p=1024, a=3),
    )
    rows = [{"pt": 1024, "ct": 10}, {"pt": 1024, "ct": 10}]
    res = fe.evaluate("tf", "dflash", log, rows, _tf_table(), ready_offset=0)
    assert res["engaged_mode"] == "dflash"
    assert [r["row"] for r in res["stats"]["per_row"]] == [0, 1]
    assert "chatcmpl-c" in res["stats"]["per_row"][0]["line"]
    assert "chatcmpl-d" in res["stats"]["per_row"][1]["line"]


def test_missing_done_line_for_a_measured_request_fails():
    """The warm request has no done line (the server never logged it): unverified."""
    log = _log(
        TF_LINE.format(i="a", p=15, a=9),
        TF_LINE.format(i="b", p=15, a=9),
        TF_LINE.format(i="c", p=1024, a=5),  # cold
    )
    rows = [{"pt": 1024, "ct": 10}, {"pt": 1024, "ct": 10}]  # cold + warm
    res = fe.evaluate("tf", "dflash", log, rows, _tf_table(), ready_offset=0)
    assert res["engaged_mode"] == "unverified"
    assert "row 1" in res["stats"]["per_row_problem"]


# ---- probe-based evidence (mlx-vlm 0.7.11: per-request draft stats only in a non-streaming response)
def _probe_table():
    return {
        "_meta": {},
        "engines": {
            "vlm": {
                "modes": {
                    "dflash": {
                        "evidence": [
                            _ev("response_field_probe", "timings.draft_rounds", per_request=True)
                            | {"paths": ["timings.draft_rounds", "timings.draft_n_accepted"]}
                        ],
                        "fallback": [],
                    }
                }
            }
        },
    }


def test_stream_rows_without_draft_stats_pass_when_the_probe_has_them():
    """The measured requests are streamed (no draft stats in their rows); the one non-streaming probe
    carries timings.draft_rounds and draft_n_accepted > 0 -> engaged by the probe."""
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]  # streamed: no attr/draft fields
    probe = {"timings": {"draft_kind": "dflash", "draft_rounds": 12, "draft_n": 80, "draft_n_accepted": 51}}
    res = fe.evaluate("vlm", "dflash", "", rows, _probe_table(), ready_offset=0, probe_response=probe)
    assert res["engaged_mode"] == "dflash"
    assert res["source"] == "probe"
    assert res["stats"]["probe"] == [12, 51]


def test_probe_with_zero_drafting_is_unverified_dflash_not_applied():
    rows = [{"pt": 1024, "ct": 64}]
    probe = {"timings": {"draft_kind": "dflash", "draft_rounds": 0, "draft_n": 0, "draft_n_accepted": 0}}
    res = fe.evaluate("vlm", "dflash", "", rows, _probe_table(), ready_offset=0, probe_response=probe)
    assert res["engaged_mode"] == "unverified"
    null = fe.evaluate("vlm", "dflash", "", rows, _probe_table(), ready_offset=0,
                       probe_response={"timings": {"draft_rounds": None}})
    assert null["engaged_mode"] == "unverified"
    absent = fe.evaluate("vlm", "dflash", "", rows, _probe_table(), ready_offset=0, probe_response=None)
    assert absent["engaged_mode"] == "unverified"


def test_probe_schema_needs_dotted_paths():
    bad = _probe_table()
    bad["engines"]["vlm"]["modes"]["dflash"]["evidence"][0]["paths"] = ["not a path!"]
    assert any("response_field_probe" in p for p in fe.validate_table(bad))


def test_omlx_mtp_line_is_per_row_acceptance_and_matches_the_real_log(tmp_path):
    """The oMLX per-request line (_log_mtp_stats) is correlated per measured row; checked on the real
    conc log on disk when it exists."""
    table = fe.load_table()
    m = table["engines"]["omlx-0.7.3"]["modes"]["dflash"]
    item = next(ev for ev in m["evidence"] if ev["kind"] == "acceptance_regex" and "MTP" in ev["pattern_or_path"])
    assert item["group"] == "accepted"
    real = glob.glob("/Volumes/Micron/yunshu-fullcmp/smoke/*/out/server-omlx-0.7.3-conc-0.log")
    if not real:
        pytest.skip("no real oMLX conc log on disk")
    text = Path(real[0]).read_text(errors="replace")
    counts = [m_.group("accepted") for m_ in re.finditer(item["pattern_or_path"], text)]
    assert len(counts) >= 3, "the real oMLX log holds per-row MTP[uid] lines"
    # the first two lines are the readiness requests (READINESS_REQUESTS); the rest are measured rows
    rows = [{"pt": 1, "ct": 1}] * (len(counts) - 2)
    res = fe.evaluate("omlx-0.7.3", "dflash", text, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", res["stats"]


def test_completed_request_lines_give_the_streamed_decode_speed():
    """generation.py:397 'Request completed: ... decode=<x> tok/s' is the streamed rows' speed."""
    log = (
        "2026-11-04 00:02:58,314 - INFO - Request completed: endpoint=/chat/completions stream=True "
        "prompt_tokens=15 generated_tokens=10 elapsed=2.317s prefill=13.2 tok/s decode=9.4 tok/s finish_reason=stop\n"
        "2026-11-04 00:02:58,656 - INFO - Request completed: endpoint=/chat/completions stream=True "
        "prompt_tokens=15 generated_tokens=10 elapsed=0.340s prefill=126.1 tok/s decode=47.5 tok/s finish_reason=stop\n"
    )
    assert fe.completed_decode_tok_s(log) == [9.4, 47.5]
    assert fe.completed_decode_tok_s("nothing") == []


def test_probe_fields_keep_the_draft_and_speed_timings():
    probe = {"timings": {"draft_kind": "dflash", "draft_rounds": 12, "draft_n": 80,
                         "draft_n_accepted": 51, "predicted_per_second": 30.5, "extra": 1}}
    fields = fe.probe_fields(probe)
    assert fields["draft_rounds"] == 12 and fields["draft_n_accepted"] == 51
    assert fields["predicted_per_second"] == 30.5 and fields["predicted_ms"] is None
    assert "extra" not in fields
    assert fe.probe_fields(None)["draft_kind"] is None


# ---- splash per-row correlation: thousands separators and the optional dedup field (2026-11-04)
SPLASH_REAL = [
    "00:00:43:16 Done · input 15 · cached 0 · output 10 · TTFT 0.1s · 120.1 tok/s · 4.5 tok/cycle\n",
    "00:00:43:16 Done · input 15 · cached 0 · output 10 · TTFT 0.0s · 120.6 tok/s · 4.5 tok/cycle\n",
    "00:00:43:18 Done · input 1,024 · cached 0 · output 64 · TTFT 0.8s · 58.1 tok/s · 2.2 tok/cycle\n",
    "00:00:43:19 Done · input 1,024 · cached 992 · deduplicated 64 · output 64 · TTFT 0.1s · 67.5 tok/s · 2.6 tok/cycle\n",
]
SYNTH_128K = "00:01:02:03 Done · input 131,072 · cached 0 · output 64 · TTFT 9.5s · 40.2 tok/s · 2.1 tok/cycle\n"


def _splash_entry():
    table = fe.load_table()
    return table, table["engines"]["splash-1.5.0"]["modes"]["dflash"]


def test_splash_real_lines_correlate_per_measured_row_with_commas_and_dedup():
    """The two measured rows (1,024 prompt) match their own done lines; the two readiness pings do not
    count. Both real lines are fixtures (wire-1104-work/rest/tfout/out/server-splash-1.5.0-decode-0.log)."""
    table, _ = _splash_entry()
    log = "".join(SPLASH_REAL)
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]
    res = fe.evaluate("splash-1.5.0", "dflash", log, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", res["stats"]
    assert [r["value"] for r in res["stats"]["per_row"]] == [2.2, 2.6]


def test_splash_128k_comma_prompt_correlates():
    table, _ = _splash_entry()
    log = "".join(SPLASH_REAL[:2]) + SYNTH_128K
    res = fe.evaluate("splash-1.5.0", "dflash", log, [{"pt": 131072, "ct": 64}], table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", res["stats"]
    assert res["stats"]["per_row"][0]["prompt"] == 131072


def test_splash_real_wire_1104_log_on_disk_engages_per_measured_row():
    """The real splash decode log on disk: 2 readiness pings + 2 measured rows of 1,024 tokens."""
    logs = glob.glob("/Volumes/Micron/yunshu-fullcmp/wire-1104-work/*/tfout/out/server-splash-1.5.0-*.log")
    if not logs:
        pytest.skip("no real splash log on disk")
    table, _ = _splash_entry()
    text = Path(logs[0]).read_text(errors="replace")
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]
    res = fe.evaluate("splash-1.5.0", "dflash", text, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", res["stats"]


def test_evaluate_without_a_table_argument_uses_the_default_file():
    """Regression (wire-1104, omlx-0.7.3rc1): a caller that passed no table raised TypeError. Every call
    now resolves the table itself; the default file is used when none is passed."""
    log = "".join(SPLASH_REAL)
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]
    res = fe.evaluate("splash-1.5.0", "dflash", log, rows, ready_offset=0)
    assert res["engaged_mode"] == "dflash"


def test_every_manifest_engine_runs_the_tfbench_engaged_path_with_a_fake_log():
    """No engine skipped: every manifest id goes through tfbench.Srv._verify_fullcmp_evidence (the call
    the GPU driver makes) with a fake log and rows. Only shape/signature errors are asserted here."""
    import fullcmp_manifest as fm
    import tfbench

    class FakeLog:
        def __init__(self, text):
            self.text = text

        def read_text(self, errors="replace"):
            return self.text

    for engine in sorted(set(fm.ENGINES) - {"ollama", "lmstudio", fm.PLACEHOLDER}):
        srv = tfbench.Srv.__new__(tfbench.Srv)
        srv.engine = engine
        srv.requested_spec_mode = "dflash"
        srv.probe = None
        srv.ready_log_offset = 0
        srv.log = FakeLog("nothing useful\n")
        srv._verify_fullcmp_evidence("nothing useful\n", final=False, rows=[{"pt": 1, "ct": 1}])
        assert srv.engaged_spec_mode in ("unverified", "dflash", "ar", "mtp", "off"), engine


def test_mlx_vlm_probe_path_is_exercised_through_the_tfbench_engaged_path():
    """The pinned id mlx-vlm-server: its probe (non-streaming timings) reaches the same tfbench call."""
    import tfbench

    class FakeLog:
        def read_text(self, errors="replace"):
            return ""

    def run(probe):
        srv = tfbench.Srv.__new__(tfbench.Srv)
        srv.engine = "mlx-vlm-server"
        srv.requested_spec_mode = "dflash"
        srv.probe = None
        srv.ready_log_offset = 0
        srv.log = FakeLog()
        srv._verify_fullcmp_evidence("", final=True, rows=[{"pt": 1024, "ct": 64}], probe_response=probe)
        return srv.engaged_spec_mode

    ok = {"timings": {"draft_rounds": 9, "draft_n_accepted": 40}}
    assert run(ok) == "dflash"
    # the final check refuses a cell whose probe shows no drafting (dflash not applied)
    with pytest.raises(RuntimeError, match="engaged=unverified"):
        run({"timings": {"draft_rounds": 0, "draft_n_accepted": 0}})


def test_finalized_session_row_spec_mode_is_the_engaged_mode_not_the_startup_value():
    """Regression (wire-1104 tf-1.2.1): the top-level spec_mode came from META at startup, before the
    measured requests' evidence was verified, so an engaged cell's header said 'unverified'. The
    finalize fields now carry the final engaged mode; a cell with no evidence stays unverified."""
    import tfbench

    class FakeLog:
        def __init__(self, text):
            self.text = text

        def read_text(self, errors="replace"):
            return self.text

    def srv(log_text):
        s = tfbench.Srv.__new__(tfbench.Srv)
        s.engine = "tf-1.2.1"
        s.requested_spec_mode = "dflash"
        s.probe = None
        s.ready_log_offset = 0
        s.log = FakeLog(log_text)
        return s

    line = "[tensorfold] done chatcmpl-x prompt=1024 cached=0 finish=length rounds=19 accepted=45/285\n"
    engaged = srv(line)
    # startup: no request evidence yet (the request-time done line is not in the log)
    engaged._verify_fullcmp_evidence("", final=False, rows=())
    assert engaged.engaged_spec_mode == "unverified"
    # the final check, after the measured requests, verifies the engagement
    ping = "[tensorfold] done chatcmpl-p prompt=15 cached=0 finish=stop rounds=1 accepted=9/15\n"
    # two readiness pings first (READINESS_REQUESTS), then the measured request's done line
    engaged._verify_fullcmp_evidence(ping * 2 + line, final=True, rows=[{"pt": 1024, "ct": 64}])
    fields = tfbench.engaged_end_fields(engaged)
    assert fields["spec_mode"] == "dflash" == fields["engaged_spec_mode"]
    assert fields["engaged_evidence_source"] == "slice"

    never = srv("nothing\n")
    never._verify_fullcmp_evidence("nothing\n", final=False, rows=())
    assert tfbench.engaged_end_fields(never)["spec_mode"] == "unverified"


KAPITSA_REAL = [
    "[kapitsa] chat: prompt 15 (cached 0) in 100 ms (+1.1 ms template/tokenize; last chunk: encoded by 0.5 ms, GPU start +17.7 ms, GPU 81.9 ms, wake +0.1 ms), 10 tokens in 80 ms (112.4 tok/s), stop, 4.50 tok/cycle, KFlash (DFlash2): 2 cycles\n",
    "[kapitsa] chat: prompt 15 (cached 0) in 86 ms (+0.9 ms template/tokenize; last chunk: encoded by 0.2 ms, GPU start +4.0 ms, GPU 81.2 ms, wake +0.1 ms), 10 tokens in 78 ms (115.2 tok/s), stop, 4.50 tok/cycle, KFlash (DFlash2): 2 cycles\n",
    "[kapitsa] chat: prompt 1024 (cached 0) in 995 ms (+1.3 ms template/tokenize; last chunk: encoded by 936.1 ms, GPU start +3.7 ms, GPU 55.0 ms, wake +0.1 ms), 64 tokens in 856 ms (73.6 tok/s), length, 2.86 tok/cycle, KFlash (DFlash2): 22 cycles\n",
    "[kapitsa] chat: prompt 1,024 (cached 1017) in 120 ms (+1.2 ms template/tokenize), 64 tokens in 700 ms (91.4 tok/s), length, 2.86 tok/cycle, KFlash (DFlash2): 22 cycles\n",
]


def test_kapitsa_per_row_kflash_evidence_with_label():
    """kapitsa runs KFlash tree drafting on the DFlash2 lattice: per measured row the chat line's
    tok/cycle (> 1.0) and 'KFlash (DFlash2)' cycle count; the label is reported with the base mode."""
    table = fe.load_table()
    entry = table["engines"]["kapitsa-0.2.26"]
    assert entry["engaged_label"] == "dflash (kflash tree)"
    log = "".join(KAPITSA_REAL)
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]
    res = fe.evaluate("kapitsa-0.2.26", "dflash", log, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", res["stats"]  # the base mode; the guard compares it
    assert res["label"] == "dflash (kflash tree)"
    assert [r["value"] for r in res["stats"]["per_row"]] == [2.86, 2.86]


def test_kapitsa_real_wire_1105_log_engages_per_measured_row():
    logs = glob.glob("/Volumes/Micron/yunshu-fullcmp/wire-1105-work/*/tfout/out/server-kapitsa-0.2.26-*.log")
    if not logs:
        pytest.skip("no real kapitsa log on disk")
    table = fe.load_table()
    text = Path(logs[0]).read_text(errors="replace")
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]
    res = fe.evaluate("kapitsa-0.2.26", "dflash", text, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "dflash", res["stats"]


LLAMA_BLOCK = (
    "0.02.942.612 I slot print_timing: id  0 | task {t} | prompt eval time =     219.78 ms /  {p} tokens (   14.65 ms per token,    68.25 tokens per second)\n"
    "0.02.942.614 I slot print_timing: id  0 | task {t} |        eval time =     173.54 ms /    10 tokens (   19.28 ms per token,    51.86 tokens per second)\n"
    "0.02.942.615 I slot print_timing: id  0 | task {t} |   graphs reused =          2\n"
    "0.02.942.617 I slot print_timing: id  0 | task {t} | draft acceptance = 0.54900 (   {a} accepted /    {g} generated), mean len =  3.33\n"
    "0.02.942.631 I slot      release: id  0 | task {t} | stop processing: n_tokens = 25, truncated = 0\n"
)


def test_llama_per_row_draft_acceptance_blocks():
    """llama-cpp b12061: one print_timing block per request (prompt-eval line, then 'draft acceptance =
    X (N accepted / M generated)'). Measured rows match by order after the readiness requests."""
    table = fe.load_table()
    blocks = [LLAMA_BLOCK.format(t=i, p=p, a=a, g=g) for i, (p, a, g) in enumerate(
        [(15, 7, 9), (4, 7, 9), (1024, 37, 60), (4, 37, 60)])]
    log = "".join(blocks)
    rows = [{"pt": 1024, "ct": 64}, {"pt": 1024, "ct": 64}]
    res = fe.evaluate("llama-cpp-b12061", "mtp", log, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "mtp", res["stats"]
    assert [r["value"] for r in res["stats"]["per_row"]] == [37, 37]


def test_llama_real_wire_1105_log_engages_per_measured_row():
    logs = glob.glob("/Volumes/Micron/yunshu-fullcmp/wire-1105-work/*/tfout/out/server-llama-cpp-b12061-*.log")
    if not logs:
        pytest.skip("no real llama log on disk")
    table = fe.load_table()
    text = Path(logs[0]).read_text(errors="replace")
    rows = [{"pt": 1034, "ct": 10}, {"pt": 1034, "ct": 10}]
    res = fe.evaluate("llama-cpp-b12061", "mtp", text, rows, table, ready_offset=0)
    assert res["engaged_mode"] == "mtp", res["stats"]
    assert [r["value"] for r in res["stats"]["per_row"]] == [39.0, 39.0]


WIRE_1105_ENGINES = ("tf-1.2.1", "tf-1.1.8", "omlx-0.7.3rc1", "mlx-serve-26.11.1", "splash-1.5.0", "llama-cpp-b12061")


def test_audit_acceptance_verdict_uses_the_evidence_not_the_attr_fields_on_real_wire_1105_records():
    """Regression (wire-1105): the guard read attr.accepted from the record, which these engines never
    populate, while the evidence showed accepted > 0 per measured row. On the real records and logs on
    disk, no engine with per-row evidence may fail 'without per-request acceptance'."""
    import types

    import fullcmp_audit as fa

    checked = []
    for engine in WIRE_1105_ENGINES:
        for jl in sorted(glob.glob(f"/Volumes/Micron/yunshu-fullcmp/wire-1105/{engine}/*.jsonl")):
            if jl.endswith(".telemetry.jsonl"):
                continue
            rows = [json.loads(x) for x in Path(jl).read_text().splitlines() if x.strip()]
            header = next((r for r in rows if r.get("part") == "session"), None)
            if header is None:
                continue
            decode = [r for r in rows if r.get("part") == "decode"]
            cell = {"cell": Path(jl).stem, "engine": engine, "spec_mode": header.get("spec_mode_expected")}
            cf = types.SimpleNamespace(header=header, cell=cell, rep=0, rows=decode, path=Path(jl))
            res = fa.check_engaged_mode(types.SimpleNamespace(files=[cf]))
            checked.append(engine)
            bad = [v for v in res.violations if "without per-request acceptance" in v or "zero or absent" in v]
            assert not bad, (engine, bad)
    if not checked:
        pytest.skip("no wire-1105 records on disk")


def test_attr_rows_are_filled_from_the_matched_evidence():
    per_row = [
        {"row": 0, "prompt": 1024, "line": "[tensorfold] done ... rounds=19 accepted=45/285 ...", "value": 45.0,
         "group": "accepted"},
        {"row": 1, "prompt": 1024, "line": "Done · input 1,024 · ... 2.6 tok/cycle", "value": 2.6, "group": "tpc"},
    ]
    rows = [{"ct": 64}, {"ct": 64}]
    a0, a1 = fe.attr_rows_from(per_row, rows)
    assert a0["accepted"] == 45 and a0["drafted"] == 285 and a0["rounds"] == 19
    assert a0["tokens_per_round"] == round(64 / 19, 3)
    # a tokens-per-cycle group is not an accepted count: unavailable, never a made-up number
    assert str(a1["accepted"]).startswith("unavailable")
    assert str(a1["drafted"]).startswith("unavailable") and str(a1["rounds"]).startswith("unavailable")


def test_attr_rows_on_the_real_wire_1105_records_are_filled_for_every_measured_row():
    """Failing-before: the records written before the final evidence carry no attribution. On the real
    wire-1105 logs, the matched evidence fills accepted (and drafted/rounds where the line states them)
    for every measured row, from the same evaluation the audit uses."""
    table = fe.load_table()
    checked = []
    for engine in WIRE_1105_ENGINES:
        for jl in sorted(glob.glob(f"/Volumes/Micron/yunshu-fullcmp/wire-1105/{engine}/*.jsonl")):
            if jl.endswith(".telemetry.jsonl"):
                continue
            lines = [json.loads(x) for x in Path(jl).read_text().splitlines() if x.strip()]
            measured = [r for r in lines if r.get("part") == "decode"]
            log_path = Path(jl).with_name(Path(jl).name[: -len(".jsonl")] + ".server.log")
            if not measured or not log_path.exists():
                continue
            mode = next((r.get("spec_mode_expected") for r in lines if r.get("part") == "session"), None)
            ev = fe.evaluate(engine, mode, log_path.read_text(errors="replace"), measured, table, ready_offset=0)
            per_row = ev["stats"].get("per_row") or []
            if not per_row:
                continue  # no per-row evidence in this engine's log: nothing to fill
            attrs = fe.attr_rows_from(per_row, measured)
            assert len(attrs) == len(measured), (engine, jl)
            # a count where the matched line states one (accepted > 0); otherwise 'unavailable (reason)'
            for a in attrs:
                if isinstance(a["accepted"], (int, float)):
                    assert a["accepted"] > 0, (engine, a)
                else:
                    assert str(a["accepted"]).startswith("unavailable"), (engine, a)
            if any(isinstance(a["accepted"], (int, float)) for a in attrs):
                checked.append(engine)
    if not checked:
        pytest.skip("no wire-1105 engine with per-row evidence on disk")


def _guard_violations_for(jl):
    """The audit's engaged_mode guard on one real cell file (read only)."""
    import types

    import fullcmp_audit as fa

    rows = [json.loads(x) for x in Path(jl).read_text().splitlines() if x.strip()]
    header = next(r for r in rows if r.get("part") == "session")
    decode = [r for r in rows if r.get("part") == "decode"]
    cell = {"cell": Path(jl).stem, "engine": header["engine"], "spec_mode": header.get("spec_mode_expected")}
    cf = types.SimpleNamespace(header=header, cell=cell, rep=0, rows=decode, path=Path(jl))
    return fa.check_engaged_mode(types.SimpleNamespace(files=[cf])).violations


def test_kapitsa_tok_cycle_only_engine_is_proven_per_row_on_the_real_raw():
    """Regression (wire-1105b): kapitsa's chat line has tok/cycle and 'KFlash (DFlash2): N cycles' but
    no accepted count. The acceptance verdict is the per-row evidence (tok/cycle > 1.0), not attr.accepted.
    A stale per-row problem from another evidence item must not fail it."""
    jls = sorted(glob.glob("/Volumes/Micron/yunshu-fullcmp/wire-1105b/kapitsa-0.2.26/*-r0.jsonl"))
    jls = [j for j in jls if "telemetry" not in j]
    if not jls:
        pytest.skip("no real kapitsa raw on disk")
    for jl in jls:
        bad = [v for v in _guard_violations_for(jl) if "without per-request acceptance" in v]
        assert not bad, bad


def test_mlx_vlm_probe_is_re_evaluated_by_the_audit_on_the_real_raw():
    """Regression (wire-1105b): the audit did not pass the probe response; the header's engine_probe is None
    for this engine. The recorded part=probe row must reach the evaluation: no 'unverified' violation."""
    jls = sorted(glob.glob("/Volumes/Micron/yunshu-fullcmp/wire-1105b/mlx-vlm-server/*-r0.jsonl"))
    jls = [j for j in jls if "telemetry" not in j]
    if not jls:
        pytest.skip("no real mlx-vlm raw on disk")
    for jl in jls:
        bad = [v for v in _guard_violations_for(jl) if "unverified" in v]
        assert not bad, bad


TF_CONC_REAL = (
    '[tensorfold] done chatcmpl-a7f9d981ca391dec3977210d34767534 prompt=15 cached=0 thinking=False effort=none tokens=10 sha=e94588627640 finish=stop rounds=1 accepted=9/15 ttft=0.037 decode=0.041 elapsed=0.078 queued=0.000 tok/s=217.6\n',
    '[tensorfold] done chatcmpl-a3aec464a36e4639539599572c1fb9bc prompt=15 cached=0 thinking=False effort=none tokens=10 sha=e94588627640 finish=stop rounds=1 accepted=9/15 ttft=0.039 decode=0.040 elapsed=0.079 queued=0.000 tok/s=224.9\n',
    '[tensorfold] done chatcmpl-3ff4edad0a3ea27a41bbcd7ff40c5f26 prompt=1024 cached=0 thinking=False effort=none tokens=64 sha=8f4c72f50c71 finish=length rounds=18 accepted=49/267 ttft=1.059 decode=0.752 elapsed=1.811 queued=0.000 tok/s=83.8\n',
    '[tensorfold] done chatcmpl-26cbb5abe07d3a983509ef8ede8d2680 prompt=1024 cached=0 thinking=False effort=none tokens=64 sha=af45034bcfc4 finish=length rounds=12 accepted=52/172 ttft=2.863 decode=0.501 elapsed=3.365 queued=1.810 tok/s=125.7\n',
    '[tensorfold] done chatcmpl-c6ebead6f8bf4aa8ba6fe46459569129 prompt=1024 cached=0 thinking=False effort=none tokens=64 sha=0de688d4dfa9 finish=length rounds=17 accepted=46/246 ttft=1.078 decode=0.703 elapsed=1.781 queued=0.024 tok/s=89.6\n',
    '[tensorfold] done chatcmpl-742a911e5f4f5c063a04fcaa6f56645a prompt=1024 cached=0 thinking=False effort=none tokens=64 sha=79079b3f5c62 finish=length rounds=12 accepted=51/174 ttft=2.837 decode=0.514 elapsed=3.352 queued=1.781 tok/s=122.5\n',
)


def _tf_item():
    entry = fe.load_table()["engines"]["tf-1.2.1"]["modes"]["dflash"]
    return next(e for e in entry["evidence"] if e["kind"] == "acceptance_regex" and e.get("group"))


def _conc_records(*sizes):
    """One conc record per entry of sizes (requests of 1024 prompt tokens, 64 completion tokens)."""
    return [{"part": "conc", "n": n, "requests": [{"rid": i, "pt": 1024, "ct": 64} for i in range(n)]}
            for n in sizes]


def test_tf_concurrent_real_lines_engage_each_request_with_queued_time():
    """Real wire-1106 tf-1.2.1 conc cell: 2 readiness pings, then two n=2 conc records (4 done lines).
    Each request matches its own line; the queued= field is read where present (serialized queue)."""
    log = "".join(TF_CONC_REAL)
    ev, problems = fe.concurrent_check(fe.load_table(), "tf-1.2.1", "dflash", log, _conc_records(2, 2), 0)
    assert problems == [], problems
    assert ev["engaged_mode"] == "dflash"
    per_row = ev["stats"]["per_row"]
    assert [r["value"] for r in per_row] == [49.0, 52.0, 46.0, 51.0]
    assert [round(r["queued_s"], 3) if r["queued_s"] is not None else None for r in per_row] == [0.0, 1.81, 0.024, 1.781]


def test_tf_concurrent_missing_done_line_fails_closed():
    """Mutation: one concurrent request has no done line (the server did not log it): the cell fails."""
    lines = list(TF_CONC_REAL)
    del lines[4]  # the third measured request's line
    ev, problems = fe.concurrent_check(fe.load_table(), "tf-1.2.1", "dflash", "".join(lines), _conc_records(2, 2), 0)
    assert problems and ev["engaged_mode"] == "unverified"


def test_tf_concurrent_extra_done_line_fails_closed():
    """Mutation: one more done line than requests (a request the record does not account for)."""
    lines = list(TF_CONC_REAL) + [TF_CONC_REAL[-1]]
    ev, problems = fe.concurrent_check(fe.load_table(), "tf-1.2.1", "dflash", "".join(lines), _conc_records(2, 2), 0)
    assert problems and ev["engaged_mode"] == "unverified"


def test_tf_concurrent_unaccepted_request_fails_closed():
    """Mutation: a measured request's done line has accepted=0: that request is not drafting."""
    lines = list(TF_CONC_REAL)
    lines[2] = lines[2].replace("accepted=49/267", "accepted=0/267")
    ev, problems = fe.concurrent_check(fe.load_table(), "tf-1.2.1", "dflash", "".join(lines), _conc_records(2, 2), 0)
    assert problems and ev["engaged_mode"] == "unverified"


def test_yunshu_concurrent_verdict_is_each_requests_own_counters():
    """Yunshu conc/mixed: no done line; each request's x_yunshu.speculative accepted must be > 0."""
    ok = [{"part": "conc", "n": 2, "requests": [
        {"rid": 0, "pt": 1024, "ct": 16, "spec": {"accepted": 3, "drafted": 4, "rounds": 2}},
        {"rid": 1, "pt": 1024, "ct": 16, "spec": {"accepted": 5, "drafted": 6, "rounds": 3}}]}]
    ev, problems = fe.concurrent_check(None, "yunshu", "dflash", "", ok, 0)
    assert problems == [] and ev["engaged_mode"] == "dflash"
    assert [r["value"] for r in ev["stats"]["per_row"]] == [3.0, 5.0]
    zero = [{"part": "conc", "n": 2, "requests": [
        {"rid": 0, "pt": 1024, "ct": 16, "spec": {"accepted": 3, "drafted": 4, "rounds": 2}},
        {"rid": 1, "pt": 1024, "ct": 16, "spec": {"accepted": 0, "drafted": 4, "rounds": 2}}]}]
    ev, problems = fe.concurrent_check(None, "yunshu", "dflash", "", zero, 0)
    assert problems and ev["engaged_mode"] == "unverified"
    absent = [{"part": "conc", "n": 1, "requests": [{"rid": 0, "pt": 1024, "ct": 16,
               "spec": "unavailable (response carries no per-request speculative counters)"}]}]
    assert fe.concurrent_check(None, "yunshu", "dflash", "", absent, 0)[1]


def test_concurrent_attr_rows_carry_rid_and_part():
    log = "".join(TF_CONC_REAL)
    rows = fe.concurrent_attr_rows(fe.load_table(), "tf-1.2.1", "dflash", log, _conc_records(2, 2), 0)
    assert [r["rid"] for r in rows] == [0, 1, 0, 1]
    assert rows[0]["accepted"] == 49 and rows[0]["rec_part"] == "conc"
    assert rows[1]["queued_s"] == 1.81
