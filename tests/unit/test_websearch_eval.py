"""Frozen evaluation preflight and mechanical citation/paired gate checks, no network."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/research/websearch_eval.py"
    spec = importlib.util.spec_from_file_location("websearch_eval", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture():
    return {
        "id": "fixture-1",
        "category": "docs",
        "query": "What is the fixture code?",
        "captured_at": "2026-10-07",
        "expected_answers": ["FOURTWO"],
        "known_good_urls": ["https://example.org/docs"],
        "results": [
            {
                "title": "Docs",
                "url": "https://example.org/docs",
                "snippet": "General docs",
            }
        ],
        "pages": {
            "https://example.org/docs": "<article><h1>Fixture code</h1><p>The fixture code is FOURTWO.</p></article>"
        },
    }


def test_snapshot_validation_and_score(tmp_path):
    m = module()
    p = tmp_path / "snapshot.jsonl"
    row = fixture()
    p.write_text(json.dumps(row) + "\n")
    assert len(m.load_snapshot(p)) == 1
    with pytest.raises(ValueError, match="design set"):
        m.load_snapshot(p, True)
    citation = {"url": "https://example.org/docs", "cited_text": "FOURTWO"}
    assert (
        m.score("FOURTWO", [citation], row, {citation["url"]: "code FOURTWO"})[
            "valid_citations"
        ]
        == 1
    )
    assert (
        m.score("wrong", [citation], row, {citation["url"]: "other"})["valid_citations"]
        == 0
    )
    row["expected_answers"] = []
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="curate"):
        m.load_snapshot(p)


def test_paired_gate_requires_200_and_net_one():
    m = module()
    equal = {"research": {"correct": True}, "snippets": {"correct": True}}
    assert not m.quality_verdict([equal] * 199)["pass"]
    assert m.quality_verdict([equal] * 200)["pass"]
    worse = {"research": {"correct": False}, "snippets": {"correct": True}}
    assert not m.quality_verdict([worse] * 2 + [equal] * 198)["pass"]


async def test_dry_run_real_extraction_no_server(tmp_path):
    m = module()
    snapshot, output = tmp_path / "snapshot.jsonl", tmp_path / "output.jsonl"
    snapshot.write_text(json.dumps(fixture()))
    a = m.parser().parse_args(
        ["--snapshot", str(snapshot), "--out", str(output), "--dry-run"]
    )
    assert await m.run(a) == 0
    final = json.loads(output.read_text().splitlines()[-1])
    assert final["complete"] and final["quality_gate"] is None


def test_design_query_set_counts_and_no_fabricated_gold():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/data/websearch_queries.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({r["id"] for r in rows}) == 130
    assert all("expected_answers" not in r for r in rows)
    assert {
        c: sum(r["category"] == c for r in rows)
        for c in ("news", "docs", "code", "adversarial")
    } == {"news": 40, "docs": 40, "code": 40, "adversarial": 10}


async def test_capture_dry_run_never_opens_network(tmp_path, monkeypatch):
    from yunshu_gateway.server_tools import search

    async def no_network(*args, **kwargs):
        pytest.fail("capture dry run opened network")

    monkeypatch.setattr(search, "run_search", no_network)
    m = module()
    source, out = tmp_path / "queries.jsonl", tmp_path / "capture.jsonl"
    source.write_text(json.dumps({"id": "one", "query": "fixture", "category": "docs"}))
    args = m.parser().parse_args(
        ["--capture", str(source), "--out", str(out), "--dry-run"]
    )
    assert await m.run(args) == 0
    assert json.loads(out.read_text())["mode"] == "capture-dry"
