"""scripts/research/fullcmp_manifest.py: the expected-cell manifest is unique, honest and deterministic."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fullcmp_manifest", ROOT / "scripts/research/fullcmp_manifest.py"
)
fm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fm)

SLICE = dict(
    engines=["yunshu", "tf-1.2.0", "mlx-lm-server", "ollama"],
    models=["q27-4bit", "qwen35-9b-4bit"],
    ctx_labels={"1K", "32K", "128K"},
)


def _slice():
    rows, stats = fm.build_rows(**SLICE)
    return rows, stats


def test_ids_are_unique():
    rows, _ = _slice()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert len(ids) > 100


def test_runnable_cells_have_positive_estimate():
    rows, _ = _slice()
    runnable = [r for r in rows if r["status"] == fm.RUNNABLE]
    assert runnable, "the slice must contain runnable cells"
    assert all(r["est_gpu_min"] > 0 for r in runnable)


def test_unsupported_and_unverified_always_have_reason_and_unsupported_costs_nothing():
    rows, _ = _slice()
    for r in rows:
        assert r["status"] in (fm.RUNNABLE, fm.UNVERIFIED, fm.UNSUPPORTED)
        if r["status"] != fm.RUNNABLE:
            assert r["reason"], r["id"]
        if r["status"] == fm.UNSUPPORTED:
            assert r["est_gpu_min"] == 0.0


def test_engine_gaps_are_rows_not_drops():
    rows, _ = _slice()
    ollama = [r for r in rows if r["engine"] == "ollama"]
    assert ollama and all(r["status"] == fm.UNSUPPORTED for r in ollama)
    assert all("n/a" in r["reason"] for r in ollama)
    text_only = [
        r
        for r in rows
        if r["engine"] == "mlx-lm-server" and r.get("media", "none") != "none"
    ]
    assert text_only and all(r["status"] == fm.UNSUPPORTED for r in text_only)


def test_pruning_rules_are_documented_and_enforced():
    assert set(fm.PRUNE_RULES) == {f"P{i}" for i in range(1, 13)}
    assert all(text.strip() for text in fm.PRUNE_RULES.values())
    rows, stats = _slice()
    assert set(stats) <= set(fm.PRUNE_RULES)
    for r in rows:
        if r["table"] != "decode":
            continue
        if r["prompt_kind"] == "json-schema":
            assert r["out_len"] <= 2048, r["id"]
        if r["out_len"] == 8192 and r["ctx_tokens"] > 32768:
            assert (r["ctx"], r["prompt_kind"]) in {
                ("128K", "code"),
                ("128K", "prose"),
                ("256K", "code"),
            }, r["id"]
        if r["variant"] == "distinct":
            assert r["cache_state"] not in ("warm_repeat", "prefix_hit"), r["id"]
        if r["cache_state"] in ("cold_process", "restart_disk"):
            assert r["conc"] == 1, r["id"]


def test_output_is_deterministic():
    a = [json.dumps(r, sort_keys=True) for r in _slice()[0]]
    b = [json.dumps(r, sort_keys=True) for r in _slice()[0]]
    assert a == b


def test_summary_prints_every_section():
    rows, stats = _slice()
    buf = io.StringIO()
    fm.summarize(rows, stats, out=buf)
    text = buf.getvalue()
    for needle in (
        "Per engine",
        "Per dimension",
        "Pruned",
        "GPU hours",
        "Proposed batch order",
    ):
        assert needle in text
