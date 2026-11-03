"""fullcmp engaged mode is derived from evidence only (scripts/research/fullcmp_engaged.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_engaged as fe  # noqa: E402

SPEC_LOG = (
    "2026-11-04 loading target\n"
    "Loading speculative drafter (dflash): /m/drafter\n"
    "Drafter ready; speculative decoding enabled.\n"
)
TABLE = {
    "tf-1.1.8": {
        "mode": "dflash",
        "evidence": [
            {
                "kind": "log_regex",
                "pattern_or_path": r"Drafter ready; speculative decoding enabled\.",
                "source_ref": "synthetic",
                "real_example": "Drafter ready; speculative decoding enabled.",
                "negative_example": "Loading speculative drafter (dflash): /m/drafter",
                "per_request": False,
            },
            {
                "kind": "acceptance_stat",
                "pattern_or_path": "attr.accepted",
                "source_ref": "synthetic",
                "real_example": "attr.accepted=3",
                "negative_example": "attr.accepted=0",
                "per_request": True,
            },
        ],
        "fallback": [r"falls back to AR"],
        "log_source": "server log",
        "confidence": "high",
        "notes": "synthetic",
    },
    "mlx-vlm-server": {
        "mode": "dflash",
        "evidence": [
            {
                "kind": "metrics",
                "pattern_or_path": r"speculative_decoding_enabled 1",
                "source_ref": "synthetic",
                "real_example": "speculative_decoding_enabled 1",
                "negative_example": "speculative_decoding_enabled 0",
                "per_request": False,
            }
        ],
        "log_source": "status probe",
        "confidence": "medium",
        "notes": "synthetic",
    },
}


def _table():
    return json.loads(json.dumps(TABLE))


def test_log_regex_engages_with_the_matched_line():
    res = fe.evaluate("tf-1.1.8", SPEC_LOG, [], _table())
    assert res["engaged_mode"] == "dflash"
    assert res["evidence_kind"] == "log_regex"
    assert res["matched_line"] == "Drafter ready; speculative decoding enabled."


def test_no_engaged_line_is_unverified_not_the_request():
    log = "Loading speculative drafter (dflash): /m/drafter\n"  # loaded, never ready
    res = fe.evaluate("tf-1.1.8", log, [], _table())
    assert res["engaged_mode"] == "unverified"
    assert res["matched_line"] is None


def test_empty_slice_is_unverified():
    assert fe.evaluate("tf-1.1.8", "", [], _table())["engaged_mode"] == "unverified"


def test_unknown_engine_is_unverified():
    assert fe.evaluate("not-an-engine", SPEC_LOG, [], _table())["engaged_mode"] == "unverified"


def test_fallback_line_is_ar_even_with_positive_evidence():
    log = SPEC_LOG + "warning: drafter incompatible, falls back to AR\n"
    res = fe.evaluate("tf-1.1.8", log, [], _table())
    assert res["engaged_mode"] == "ar"
    assert res["evidence_kind"] == "fallback"
    assert "falls back to AR" in res["matched_line"]


def test_metrics_evidence_reads_the_probe_text():
    on = fe.evaluate("mlx-vlm-server", "", [], _table(), probe_text="speculative_decoding_enabled 1\n")
    off = fe.evaluate("mlx-vlm-server", "", [], _table(), probe_text="speculative_decoding_enabled 0\n")
    assert on["engaged_mode"] == "dflash" and on["evidence_kind"] == "metrics"
    assert off["engaged_mode"] == "unverified"
    assert fe.evaluate("mlx-vlm-server", "", [], _table(), probe_text=None)["engaged_mode"] == "unverified"


def test_acceptance_stat_counts_as_evidence_only_when_positive():
    rows = [{"attr": {"accepted": 0}}, {"attr": {"accepted": 2}}]
    res = fe.evaluate("tf-1.1.8", "", rows, _table())
    assert res["engaged_mode"] == "dflash"
    assert res["evidence_kind"] == "acceptance_stat"
    zero = fe.evaluate("tf-1.1.8", "", [{"attr": {"accepted": 0}}], _table())
    assert zero["engaged_mode"] == "unverified"


def test_acceptance_problem_when_all_stats_zero_or_absent():
    entry = TABLE["tf-1.1.8"]
    assert fe.acceptance_problem("dflash", entry, [{"attr": {"accepted": 1}}]) is None
    assert fe.acceptance_problem("dflash", entry, [{"attr": {"accepted": 0}}, {}]) is not None
    assert fe.acceptance_problem("dflash", entry, []) is not None
    assert fe.acceptance_problem("ar", entry, [{}]) is None
    assert fe.acceptance_problem("off", entry, []) is None


def test_validate_rejects_missing_source_ref_and_bad_regex():
    bad = _table()
    bad["tf-1.1.8"]["evidence"][0]["source_ref"] = ""
    bad["mlx-vlm-server"]["evidence"][0]["pattern_or_path"] = "(unclosed"
    problems = fe.validate_table(bad)
    assert any("source_ref" in p for p in problems)
    assert any("bad regex" in p for p in problems)
    assert fe.validate_table(_table()) == []


def test_load_table_is_none_when_missing_or_malformed(tmp_path):
    assert fe.load_table(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert fe.load_table(broken) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"x": {"mode": "warp", "evidence": []}}))
    assert fe.load_table(wrong) is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_table()))
    assert fe.load_table(good)["tf-1.1.8"]["mode"] == "dflash"


def test_default_table_path_is_beside_the_module():
    assert Path(fe.DEFAULT_TABLE).name == "fullcmp_engaged_evidence.json"
    assert Path(fe.DEFAULT_TABLE).parent == ROOT / "scripts" / "research"


@pytest.mark.parametrize("mode", ["off", "ar"])
def test_ar_modes_never_need_acceptance(mode):
    assert fe.acceptance_problem(mode, TABLE["tf-1.1.8"], []) is None


SPLASH_STARTUP = (
    "Loading DFlash2 draft /m/draft\n"
    "Decode cycles verify every row of the draft's block\n"
)
SPLASH_TABLE = {
    "splash-1.5.0": {
        "mode": "dflash",
        "evidence": [
            {
                "kind": "log_regex",
                "pattern_or_path": r"Decode cycles verify every row of the draft's block",
                "source_ref": "synthetic",
                "real_example": SPLASH_STARTUP.splitlines()[1],
                "negative_example": "Loading DFlash2 draft /m/draft",
                "per_request": False,
            }
        ],
        "log_source": "server log",
        "confidence": "high",
        "notes": "synthetic",
    }
}


def test_splash_cell_slice_keeps_the_startup_evidence(tmp_path):
    """Regression (2026-11-04): the slice began at readiness, after Splash printed its drafter line
    and with no per-request log lines, so splash-1.5.0 cells had an empty .server.log slice."""
    import tfbench

    import bench_engines as be

    launch = be.build_launch("splash-1.5.0", 18990, tmp_path / "home", {}, 140000, 1)
    assert "serve" in launch.cmd
    log = tmp_path / "server.log"
    log.write_bytes(SPLASH_STARTUP.encode())  # startup only: printed before readiness
    cell = tfbench.cell_engine_log(log, log.stat().st_size)
    assert cell, "the cell's engine-log slice must not be empty"
    assert fe.evaluate("splash-1.5.0", cell.decode(), [], SPLASH_TABLE)["engaged_mode"] == "dflash"


def test_srv_header_takes_the_mode_from_evidence_not_the_request(monkeypatch):
    """The Srv header engaged mode: evidence only. Request copied, no evidence -> refused."""

    import tfbench

    class FakeLog:
        def __init__(self, text):
            self.text = text

        def read_text(self, errors="replace"):
            return self.text

    def make(text):
        srv = tfbench.Srv.__new__(tfbench.Srv)
        srv.engine = "splash-1.5.0"
        srv.requested_spec_mode = "dflash"
        srv.probe = '{"draft": "local DFlash2"}'  # the config echo the old copy read as evidence
        srv.log = FakeLog(text)
        return srv

    monkeypatch.setattr(fe, "load_table", lambda: SPLASH_TABLE)
    srv = make(SPLASH_STARTUP)
    srv._verify_fullcmp_evidence(SPLASH_STARTUP)
    assert srv.engaged_spec_mode == "dflash"
    assert srv.engaged_log_line == SPLASH_STARTUP.splitlines()[1]
    assert srv.engaged_evidence_kind == "log_regex"

    srv = make("Loading DFlash2 draft /m/draft\n")  # the probe says draft, the log has no evidence
    with pytest.raises(RuntimeError, match="engaged=unverified"):
        srv._verify_fullcmp_evidence("Loading DFlash2 draft /m/draft\n")
    monkeypatch.setattr(fe, "load_table", lambda: None)  # no table: no evidence at all
    with pytest.raises(RuntimeError, match="table: missing"):
        make(SPLASH_STARTUP)._verify_fullcmp_evidence(SPLASH_STARTUP)
