"""CPU-only tests for the final-table harness cleanups.

Covers: the report keeps kapitsa out of the main table and lists it in a footnoted appendix; tool-chat
cells report end-to-end latency and completion tokens instead of decode tok/s; the audit manifest
expects only the cells this run planned (an engine left out of the run, such as lmstudio, expects no file).
Every input is a synthetic file under tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


report = _load("fullcmp_report", "scripts/dev/fullcmp_report.py")
run = _load("fullcmp_run_final", "scripts/research/fullcmp_run.py")


def _cell(engine: str, kind: str, cell_id: str = "1K.c1") -> str:
    return f"B2.core.{engine}.{cell_id}.{kind}.o512.single.th-off.none.greedy"


def _write(raw: Path, engine: str, name: str, records: list[dict]) -> None:
    d = raw / engine
    d.mkdir(parents=True, exist_ok=True)
    lines = [{"part": "session", "version": "1", "git_sha": "abc"}] + records
    (d / f"{name}-r0.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_kapitsa_is_appendix_not_main_table(tmp_path):
    raw = tmp_path / "raw"
    _write(
        raw,
        "yunshu",
        _cell("yunshu", "code"),
        [{"phase": "cold", "ttft_s": 1.0, "dec_tps": 50.0, "ct": 512}],
    )
    _write(
        raw,
        "kapitsa-0.2.31",
        _cell("kapitsa-0.2.31", "code"),
        [{"phase": "cold", "ttft_s": 1.2, "dec_tps": 40.0}],
    )
    rows, _meta = report.collect(str(raw))
    assert {r["engine"] for r in rows} == {"yunshu", "kapitsa-0.2.31"}
    main = [r for r in rows if not report.is_appendix(r["engine"])]
    appendix = [r for r in rows if report.is_appendix(r["engine"])]
    assert [r["engine"] for r in main] == ["yunshu"]
    assert [r["engine"] for r in appendix] == ["kapitsa-0.2.31"]
    assert all(not e.startswith("kapitsa") for e in report.ENGINE_ORDER)
    assert (
        "0.2.26 -> 0.2.31" in report.APPENDIX_NOTE
        and "not comparable" in report.APPENDIX_NOTE
    )


def test_tool_chat_reports_end_to_end_and_tokens_not_decode(tmp_path):
    raw = tmp_path / "raw"
    attr = {"t_receive": 100.0, "t_last_token": 102.5}
    _write(
        raw,
        "yunshu",
        _cell("yunshu", "tool-chat", "8K.c1"),
        [{"phase": "cold", "ttft_s": 0.4, "dec_tps": 9999.0, "ct": 40, "attr": attr}],
    )
    # a tool-chat record without client timestamps has no end-to-end value: it is not reported
    _write(
        raw,
        "tf-1.2.1",
        _cell("tf-1.2.1", "tool-chat", "8K.c1"),
        [{"phase": "cold", "ttft_s": 0.5, "dec_tps": 80.0, "ct": 30}],
    )
    rows, _ = report.collect(str(raw))
    by_engine = {r["engine"]: r for r in rows}
    assert "tf-1.2.1" not in by_engine
    row = by_engine["yunshu"]
    assert row["e2e_s"] == 2.5
    assert row["ct"] == 40
    assert "dec_tps" not in row
    # the page renders tool-chat cells from e2e_s and ct, never from dec_tps
    assert "端到端" in report.HTML and "r.kind==='tool-chat'" in report.HTML


def test_tool_chat_e2e_is_median_over_reps(tmp_path):
    raw = tmp_path / "raw"
    name = _cell("yunshu", "tool-chat", "8K.c1")
    _write(
        raw,
        "yunshu",
        name,
        [{"phase": "cold", "ct": 40, "attr": {"t_receive": 0.0, "t_last_token": 2.0}}],
    )
    d = raw / "yunshu"
    lines = [
        {"part": "session"},
        {"phase": "cold", "ct": 44, "attr": {"t_receive": 10.0, "t_last_token": 13.0}},
    ]
    (d / f"{name}-r1.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows, _ = report.collect(str(raw))
    row = next(r for r in rows if r["engine"] == "yunshu")
    assert row["n"] == 2 and row["e2e_s"] == 2.5 and row["ct"] == 42


def test_audit_scope_drops_engines_not_planned(tmp_path):
    rows = [
        {
            "batch": "B2",
            "engine": "yunshu",
            "id": "B2.core.yunshu.1K",
            "status": "unverified",
        },
        {
            "batch": "B2",
            "engine": "lmstudio",
            "id": "B2.core.lmstudio.1K",
            "status": "unverified",
        },
        {
            "batch": "B2",
            "engine": "yunshu",
            "id": "B2.core.yunshu.8K",
            "status": "unverified",
        },
        {
            "batch": "B1",
            "engine": "yunshu",
            "id": "B1.core.yunshu.1K",
            "status": "unverified",
        },
        {
            "batch": "B1",
            "engine": "lmstudio",
            "id": "B1.core.lmstudio.1K",
            "status": "unverified",
        },
    ]
    # the run planned only the 1K yunshu cell of B2 (--cells / --engines cut the rest)
    plan = SimpleNamespace(runnable=[rows[0]])
    kept = run.audit_scope_rows(rows, "B2", plan)
    ids = [r["id"] for r in kept]
    assert (
        "B2.core.lmstudio.1K" not in ids
    )  # not planned by the run, and no file exists
    assert "B2.core.yunshu.8K" not in ids  # planned batch: only what this run planned
    assert "B1.core.yunshu.1K" in ids  # an earlier audited batch keeps its rows
    assert (
        "B1.core.lmstudio.1K" not in ids
    )  # a harness stub engine never expects a file


def test_audit_scope_for_an_engine_filtered_run_expects_only_that_engine():
    rows = [
        {
            "batch": "B2",
            "engine": "yunshu",
            "id": "B2.core.yunshu.1K",
            "status": "unverified",
        },
        {
            "batch": "B2",
            "engine": "lmstudio",
            "id": "B2.core.lmstudio.1K",
            "status": "unverified",
        },
    ]
    plan = SimpleNamespace(runnable=[rows[0]])
    assert [r["engine"] for r in run.audit_scope_rows(rows, "B2", plan)] == ["yunshu"]
