"""Per-request evidence lines tolerate fields the engine adds (tool calls, grammar, thinking, images, cached/dedup).

wire-1108rest, Splash p3, kind/tool-chat: the done line carries 'tools 2·c5b1f9df' and the per-row regex, written for
'input · cached [· deduplicated] · output · TTFT · tok/s · tok/cycle', rejected it, as it had rejected 'deduplicated'
before. The class is a fixed field order: the patterns now require only the fields the verdict needs (Splash: input,
output, tok/cycle; key=value engines: each key anywhere on the line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import fullcmp_engaged as fe  # noqa: E402

# the real wire-1108rest Splash done line of the kind/tool-chat cell
SPLASH_TOOLS = "Done · input 8,573 · cached 0 · output 27 · tools 2·c5b1f9df · TTFT 7.0s · 135.8 tok/s · 5.2 tok/cycle"


def _item(engine, mode, nth=-1):
    table = fe.load_table()
    items = [i for i in table["engines"][engine]["modes"][mode]["evidence"] if i["kind"] == "acceptance_regex" and i.get("per_request")]
    return items[nth]


def _rows(*pts):
    return [{"pt": p, "ct": 16, "cached": 0} for p in pts]


def test_splash_done_line_with_a_tools_segment_is_per_row_evidence():
    rx = re.compile(_item("splash-1.5.0", "dflash")["pattern_or_path"])
    m = rx.search(SPLASH_TOOLS)
    assert m and m.group("prompt") == "8,573" and m.group("tpc") == "5.2"


@pytest.mark.parametrize(
    "line,prompt,tpc",
    [
        ("Done · input 15 · cached 0 · output 10 · TTFT 0.2s · 118.5 tok/s · 4.5 tok/cycle", "15", "4.5"),
        ("Done · input 1,024 · cached 992 · deduplicated 64 · output 64 · TTFT 0.1s · 67.5 tok/s · 2.6 tok/cycle", "1,024", "2.6"),
        (SPLASH_TOOLS, "8,573", "5.2"),
        ("Done · input 900 · cached 0 · output 40 · grammar json-schema · TTFT 1.0s · 90.0 tok/s · 3.1 tok/cycle", "900", "3.1"),
        ("Done · input 900 · images 1 · cached 0 · thinking 120 · output 40 · TTFT 1.0s · 90.0 tok/s · 3.1 tok/cycle", "900", "3.1"),
        ("Done · input 7 · output 3 · 2.0 tok/cycle", "7", "2.0"),  # only the required segments
    ],
)
def test_splash_pattern_requires_only_input_output_and_tok_per_cycle(line, prompt, tpc):
    m = re.search(_item("splash-1.5.0", "dflash")["pattern_or_path"], line)
    assert m and m.group("prompt") == prompt and m.group("tpc") == tpc


@pytest.mark.parametrize("line", ["Done · output 10 · 2.0 tok/cycle", "Done · input 10 · 2.0 tok/cycle", "Done · input 10 · output 3 · TTFT 1s"])
def test_splash_pattern_still_rejects_a_line_missing_a_required_segment(line):
    assert not re.search(_item("splash-1.5.0", "dflash")["pattern_or_path"], line)


def test_the_real_tool_chat_cell_correlates_end_to_end_through_the_verdict():
    """Failing-before: the old pattern found no done line for the 8,573-token row ('no done line': startup_only)."""
    ready = [
        "00:00:01 Done · input 15 · cached 0 · output 10 · TTFT 0.2s · 118.5 tok/s · 4.5 tok/cycle",
        "00:00:02 Done · input 15 · cached 0 · output 10 · TTFT 0.1s · 120.1 tok/s · 4.5 tok/cycle",
    ]
    log = "Decode cycles verify every row of the draft's block (--verify-rows auto)\n" + "\n".join([*ready, SPLASH_TOOLS]) + "\n"
    ev = fe.evaluate("splash-1.5.0", "dflash", log, _rows(8573), fe.load_table(), ready_offset=0)
    assert ev["engaged_mode"] == "dflash", ev


# --- the other engines' per-request patterns, with an extra field the engine could add ----------------------------


def _perturb(engine, line):
    if engine.startswith("kapitsa"):
        return line.replace(", KFlash (DFlash2)", ", tools 2, KFlash (DFlash2)").replace(", stop,", ", stop, grammar json,")
    if engine.startswith("omlx"):
        return line.replace(" tok/cycle=", " newfield=7 tok/cycle=").replace(" accept=", " extra=1 accept=")
    if engine.startswith("mlx-serve"):
        return line.replace(" accepts=", " newfield=3 accepts=").replace(" drafted=", " zz=1 drafted=")
    if engine.startswith("tf-"):
        return line.replace(" cached=", " tools=2 grammar=json cached=")
    return line


@pytest.mark.parametrize(
    "engine,mode",
    [("kapitsa-0.2.26", "dflash"), ("omlx-0.7.3", "dflash"), ("omlx-0.7.3rc1", "dflash"), ("mlx-serve-26.11.1", "mtp"), ("tf-1.2.1", "dflash"), ("tf-1.1.8", "dflash")],
)
def test_per_request_patterns_survive_an_extra_field(engine, mode):
    item = _item(engine, mode)
    example = item["real_example"]
    rx = re.compile(item["pattern_or_path"])
    base = rx.search(example)
    assert base, "the real example must match"
    grown = _perturb(engine, example)
    assert grown != example
    m = rx.search(grown)
    assert m, f"{engine}: an extra field broke the pattern"
    for g in base.groupdict():
        if base.group(g) is not None:
            assert m.group(g) == base.group(g), (engine, g)


# the real Splash kind cells of wire-1108rest (done lines copied from their server logs): two readiness pings, then
# the cell's one request; rows' prompt counts are the usage values the cell recorded
SPLASH_KIND_CELLS = {
    "cjk": ("Done · input 8,211 · cached 0 · output 512 · TTFT 6.7s · 59.1 tok/s · 2.3 tok/cycle", 8211),
    "json-schema": ("Done · input 8,216 · cached 0 · output 263 · TTFT 6.7s · 156.3 tok/s · 6.1 tok/cycle", 8216),
    "tool-chat": (SPLASH_TOOLS, 8573),
}


@pytest.mark.parametrize("kind", sorted(SPLASH_KIND_CELLS))
def test_each_real_splash_kind_cell_verdict_is_dflash(kind):
    line, pt = SPLASH_KIND_CELLS[kind]
    ready = [
        "13:44:42 Done · input 15 · cached 0 · output 10 · TTFT 0.1s · 124.4 tok/s · 4.5 tok/cycle",
        "13:44:43 Done · input 15 · cached 0 · output 10 · TTFT 0.0s · 124.5 tok/s · 4.5 tok/cycle",
    ]
    log = "Decode cycles verify every row of the draft's block (--verify-rows auto)\n" + "\n".join([*ready, "13:44:58 " + line]) + "\n"
    ev = fe.evaluate("splash-1.5.0", "dflash", log, _rows(pt), fe.load_table(), ready_offset=0)
    assert ev["engaged_mode"] == "dflash", (kind, ev)


# --- constrained kind cells an engine decodes without drafting (wire-1108rest mlx-serve, kind/json-schema) ------------

MLX_SERVE_JSON = (Path(__file__).parent / "fixtures" / "mlx_serve_json_schema_cell.server.log").read_text()
MLX_SERVE_READY = MLX_SERVE_JSON.index("POST /v1/chat/completions")  # the log offset after readiness


def _json_row(pt=8310):
    return {"pt": pt, "ct": 190, "cached": 0, "family": "json-schema", "kind": "json-schema"}


def test_mlx_serve_json_schema_cell_is_ar_with_the_constrained_label():
    """Failing-before: the cell ended 'requested mtp, engaged=unverified (startup_only)' and aborted the chunk. The real
    log has '[grammar] enforcing JSON schema from token 0', decodes at 30.1 tok/s and prints no [spec-stats] line for
    the request: the verdict is a labelled 'ar'."""
    ev = fe.evaluate("mlx-serve-26.11.1", "mtp", MLX_SERVE_JSON, [_json_row()], fe.load_table(), ready_offset=MLX_SERVE_READY)
    assert ev["engaged_mode"] == "ar" and ev["evidence_kind"] == "constrained_ar", ev
    assert ev["label"].startswith("ar (constrained decoding: mlx-serve does not draft under a JSON schema")
    assert "no per-request spec line + grammar line" in ev["label"]
    assert "[grammar] enforcing JSON schema" in ev["matched_line"]


def test_without_the_engines_feature_line_the_same_cell_stays_unverified():
    log = MLX_SERVE_JSON.replace("[grammar] enforcing JSON schema from token 0", "[grammar] other message")
    ev = fe.evaluate("mlx-serve-26.11.1", "mtp", log, [_json_row()], fe.load_table(), ready_offset=MLX_SERVE_READY)
    assert ev["engaged_mode"] == fe.UNVERIFIED  # absent + no explanation: fail


def test_per_request_spec_evidence_beats_the_constrained_label():
    spec = (
        "  [spec-stats] mode=mtp attempts=40 accepts=60 avg_per_round=1.50 per_draft_pct=60.0% depth=6 drafted=100 "
        "ext_rounds=0 partial_rounds=0 runtime_disabled=false reason=none adaptive=undecided\n"
    )
    log = MLX_SERVE_JSON.replace("  <- 8310+190 tokens streamed", spec + "  <- 8310+190 tokens streamed")
    ev = fe.evaluate("mlx-serve-26.11.1", "mtp", log, [_json_row()], fe.load_table(), ready_offset=MLX_SERVE_READY)
    assert ev["engaged_mode"] == "mtp" and ev["evidence_kind"] != "constrained_ar", ev


def test_a_plain_kind_cell_never_gets_the_constrained_label():
    row = {"pt": 8310, "ct": 190, "cached": 0, "family": "cjk", "kind": "cjk"}
    ev = fe.evaluate("mlx-serve-26.11.1", "mtp", MLX_SERVE_JSON, [row], fe.load_table(), ready_offset=MLX_SERVE_READY)
    assert ev["engaged_mode"] == fe.UNVERIFIED


def test_the_in_job_check_accepts_the_labelled_ar_instead_of_aborting_the_chunk(tmp_path, monkeypatch):
    import types

    import tfbench as tb

    monkeypatch.setenv("TFB_FULLCMP", "1")
    srv = object.__new__(tb.Srv)
    srv.engine = "mlx-serve-26.11.1"
    srv.requested_spec_mode = "mtp"
    srv.log = tmp_path / "live.log"
    srv.log.write_text(MLX_SERVE_JSON)
    srv.ready_log_offset = MLX_SERVE_READY
    srv.probe_text = lambda: ""
    srv.verify_spec_mode(final=True, rows=[_json_row()])  # no RuntimeError
    assert srv.engaged_spec_mode == "ar" and srv.engaged_evidence_kind == "constrained_ar"
    assert srv.engaged_label.startswith("ar (constrained decoding")
    assert types  # keep the import local


def test_the_audit_passes_a_constrained_ar_cell_with_its_label(tmp_path):
    """The audit re-derives the verdict from the cell's slice: the labelled 'ar' of a constrained kind cell passes
    (configured 'mtp'), a plain cell with the same log does not."""
    import json

    import fullcmp_audit as fa

    ev = fe.evaluate("mlx-serve-26.11.1", "mtp", MLX_SERVE_JSON, [_json_row()], fe.load_table(), ready_offset=MLX_SERVE_READY)
    path = tmp_path / "c-r0.jsonl"
    path.write_text(json.dumps({"part": "session"}) + "\n")
    (tmp_path / "c-r0.server.log").write_text(MLX_SERVE_JSON)
    header = {
        "part": "session", "server_log_ready_offset": MLX_SERVE_READY, "engaged_spec_mode": "ar",
        "engaged_log_line": ev["matched_line"], "engaged_evidence_source": ev["source"],
        "requested_spec_mode": "mtp", "spec_mode_expected": "mtp",
    }

    def run(row):
        cell = {"cell": "B2.kind.mlx-serve.json-schema", "engine": "mlx-serve-26.11.1", "spec_mode": "mtp", "group": "g"}
        cf = fa.CellFile(cell, 0, path, header, [row])
        return fa.check_engaged_mode(fa.Run(manifest={}, files=[cf], missing=[], extra=[], parse_errors=[]))

    assert run(_json_row()).violations == []
    plain = {"pt": 8310, "ct": 190, "cached": 0, "family": "cjk", "kind": "cjk"}
    assert run(plain).violations  # not a constrained kind: the same log is unverified


def test_the_table_validates_and_a_malformed_no_draft_entry_is_rejected():
    table = fe.load_table()
    assert table is not None
    bad = {"engines": {"x": {"modes": {"mtp": {"evidence": [{"kind": "log_regex", "pattern_or_path": "a", "source_ref": "s",
                                                           "per_request": False, "startup_sufficient": True,
                                                           "confidence": "c"}], "no_draft": [{"kinds": ["json-schema"]}]}}}}}
    assert any("no_draft[0]" in p for p in fe.validate_table(bad))


# --- concurrent cells: a request that drafted with 0 accepted is still speculative decoding (wire-1108c8 mlx-serve c8) ----

MLX_SERVE_C8 = (Path(__file__).parent / "fixtures" / "mlx_serve_conc_c8.server.log").read_text()
MLX_SERVE_C8_READY = 3702  # the cell's recorded server_log_ready_offset


def _c8_records():
    req = {"pt": 1024, "ct": 512}
    return [{"part": "conc", "n": 8, "requests": [dict(rid=i, **req) for i in range(8)]} for _ in range(2)]


def _spec_lines(log):
    rx = re.compile(r"\[spec-stats\] mode=mtp attempts=(\d+) accepts=(\d+) .*? drafted=(\d+)")
    return [(int(a), int(c), int(d)) for a, c, d in rx.findall(log[MLX_SERVE_C8_READY:])][2:]  # past the readiness pings


def test_mlx_serve_c8_cell_passes_with_the_drafted_share_label():
    """Failing-before: 'concurrent conc n=8: request 3: value 0.0 not above 0.0' aborted the cell: under --max-concurrent 8
    the per-request spec-stats are tiny ('attempts=1 accepts=0 drafted=1'). Every request matched its line, so the cell
    passes and the label carries the drafted share."""
    lines = _spec_lines(MLX_SERVE_C8)
    assert len(lines) == 16 and any(c == 0 for _a, c, _d in lines)  # the real log does contain accepts=0 requests
    ev, problems = fe.concurrent_check(fe.load_table(), "mlx-serve-26.11.1", "mtp", MLX_SERVE_C8, _c8_records(), MLX_SERVE_C8_READY)
    assert problems == [], problems
    drafted = [x for x in lines if x[2] > 0]
    pct = round(100.0 * sum(c for _a, c, _d in drafted) / sum(d for _a, _c, d in drafted))
    assert ev["engaged_mode"] == "mtp"
    assert ev["label"] == f"mtp (drafted on {len(drafted)} of 16 requests; {pct}% of tokens accepted)"
    assert fe.acceptance_proven(ev) is True


def test_a_request_with_drafted_zero_is_a_no_draft_request_not_a_failure():
    base = (
        "  [spec-stats] mode=mtp attempts={a} accepts={c} avg_per_round=0.50 per_draft_pct=50.0% depth=6 drafted={d} "
        "ext_rounds=0 partial_rounds=0 runtime_disabled=false reason=none adaptive=undecided\n"
    )
    warm = base.format(a=5, c=5, d=5) * 2
    reqs = [{"rid": i, "pt": 1024, "ct": 512} for i in range(4)]
    recs = [{"part": "conc", "n": 4, "requests": reqs}]
    log = warm + base.format(a=2, c=1, d=2) + base.format(a=0, c=0, d=0) + base.format(a=1, c=0, d=1) + base.format(a=3, c=3, d=3)
    ev, problems = fe.concurrent_check(fe.load_table(), "mlx-serve-26.11.1", "mtp", log, recs, 0)
    assert problems == [] and ev["engaged_mode"] == "mtp"
    assert ev["label"] == "mtp (drafted on 3 of 4 requests; 67% of tokens accepted)"  # (1+0+3) / (2+1+3)
    none = warm + base.format(a=0, c=0, d=0) * 4
    ev, problems = fe.concurrent_check(fe.load_table(), "mlx-serve-26.11.1", "mtp", none, recs, 0)
    assert problems == [] and ev["engaged_mode"] == "ar" and ev["label"] == "ar (concurrent path does not draft)"


def test_an_unmatched_line_or_a_count_mismatch_still_fails_the_concurrent_cell():
    log = MLX_SERVE_C8[:MLX_SERVE_C8_READY] + "".join(
        ln + "\n" for ln in MLX_SERVE_C8[MLX_SERVE_C8_READY:].splitlines() if "[spec-stats] mode=mtp" in ln
    )
    short = "\n".join(log.splitlines()[:-3]) + "\n"  # three requests lost their line
    ev, problems = fe.concurrent_check(fe.load_table(), "mlx-serve-26.11.1", "mtp", short, _c8_records(), MLX_SERVE_C8_READY)
    assert problems and ev["engaged_mode"] == fe.UNVERIFIED


def test_tensorfold_done_lines_expose_the_drafted_count():
    rx = re.compile(_item("tf-1.2.1", "dflash")["pattern_or_path"])
    m = rx.search("[tensorfold] done chatcmpl-a prompt=15 cached=0 tokens=10 rounds=1 accepted=9/15 ttft=0.037")
    assert m and m.group("accepted") == "9" and m.group("drafted") == "15"


KAPITSA_C8 = (Path(__file__).parent / "fixtures" / "kapitsa_conc_c8.server.log").read_text()
KAPITSA_C8_READY = 2825  # the cell's recorded server_log_ready_offset (launched --slots 8 --context 3072 --private)


def test_kapitsa_c8_cell_where_almost_no_slot_drafts_passes_with_the_share_label():
    """Failing-before (wire-1108c8): 'request 0: value 1.0 not above 1.0' aborted the cell. With 8 slots busy 15 of the 16
    requests log '1.00 tok/cycle' (KFlash does not draft) and one logs 1.01: no-draft requests, and the cell passes with
    its drafted share. A cell where all 16 were 1.00 is the labelled ar (next test)."""
    ev, problems = fe.concurrent_check(fe.load_table(), "kapitsa-0.2.26", "dflash", KAPITSA_C8, _c8_records(), KAPITSA_C8_READY)
    assert problems == [], problems
    assert ev["engaged_mode"] == "dflash" and ev["label"] == "dflash (drafted on 1 of 16 requests)"
    assert sum(1 for r in ev["stats"]["per_row"] if r["drafted_request"]) == 1 and len(ev["stats"]["per_row"]) == 16


def test_kapitsa_c8_cell_where_no_slot_drafts_is_the_labelled_ar():
    log = re.sub(r"1\.01 tok/cycle", "1.00 tok/cycle", KAPITSA_C8)
    ev, problems = fe.concurrent_check(fe.load_table(), "kapitsa-0.2.26", "dflash", log, _c8_records(), KAPITSA_C8_READY)
    assert problems == [], problems
    assert ev["engaged_mode"] == "ar" and ev["label"] == "ar (concurrent path does not draft)"
    assert all(r["drafted_request"] is False for r in ev["stats"]["per_row"])


def test_kapitsa_requests_above_one_token_per_cycle_are_drafted_requests():
    line = (
        "[kapitsa] chat: prompt 1024 (cached 0) in 900 ms (x), 512 tokens in 9000 ms (56.9 tok/s), length, {tpc} tok/cycle, "
        "KFlash (DFlash2): 100 cycles\n"
    )
    warm = line.format(tpc="4.50") * 2
    recs = [{"part": "conc", "n": 4, "requests": [{"rid": i, "pt": 1024, "ct": 512} for i in range(4)]}]
    log = warm + line.format(tpc="1.00") + line.format(tpc="2.30") + line.format(tpc="1.00") + line.format(tpc="3.10")
    ev, problems = fe.concurrent_check(fe.load_table(), "kapitsa-0.2.26", "dflash", log, recs, 0)
    assert problems == [] and ev["engaged_mode"] == "dflash"
    assert ev["label"] == "dflash (drafted on 2 of 4 requests)"  # no drafted count on this line: no accepted percentage
    assert fe.acceptance_proven(ev) is True
