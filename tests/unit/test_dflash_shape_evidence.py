"""Performance decisions reject partial, mismatched, or diagnostic evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "shape_evidence",
    Path(__file__).parents[2] / "scripts/research/analyze_dflash_shape.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize = _module.summarize


def evidence():
    rows = [
        dict(
            part="snapshot",
            mode="dflash",
            barrier=False,
            arms=["main", "copy8"],
            contexts=[1024],
            tasks=["code"],
            reps=3,
            python_sha256="source",
            harness_sha256="harness",
        ),
        dict(part="reference", context=1024, task="code", sha="tokens"),
    ]
    for rep in range(3):
        for arm, tps in (("main", 100.0), ("copy8", 102.0)):
            rows.append(
                dict(
                    part="result",
                    context=1024,
                    task="code",
                    arm=arm,
                    rep=rep,
                    sha="tokens",
                    parity=True,
                    tps=tps,
                    round_ms=50.0,
                    commits_per_round=5.0,
                    tokens=6,
                    rounds=[dict(committed=5)],
                )
            )
    rows.append(dict(complete=True, success=True))
    return rows, dict(state="done", rc=0, quiet=True, contended=False)


def test_complete_clean_matrix_counts_three_runs():
    rows, receipt = evidence()
    result = summarize(rows, receipt)
    winner = next(r for r in result if r["arm"] == "copy8")
    assert winner["gain_pct"] == pytest.approx(2.0)
    assert winner["reps"] == 3


def test_explicit_baseline_uses_current_copy_default():
    rows, receipt = evidence()
    rows[0]["arms"][0] = "nativecopy8"
    for row in rows:
        if row.get("arm") == "main":
            row["arm"] = "nativecopy8"
    result = summarize(rows, receipt, baseline="nativecopy8")
    winner = next(r for r in result if r["arm"] == "copy8")
    assert winner["gain_pct"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="main baseline required"):
        summarize(rows, receipt)
    with pytest.raises(ValueError, match="missing baseline required"):
        summarize(rows, receipt, baseline="missing")


@pytest.mark.parametrize(
    "field,value",
    [("rc", 1), ("state", "running"), ("quiet", False), ("contended", True)],
)
def test_receipt_overrides_harness_timing_claim(field, value):
    rows, receipt = evidence()
    receipt[field] = value
    with pytest.raises(ValueError):
        summarize(rows, receipt)


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "duplicate",
        "digest",
        "trace",
        "barrier",
        "capture",
        "reps",
        "mode",
        "missing_source",
        "unfinished",
    ],
)
def test_reject_invalid_evidence(damage):
    rows, receipt = evidence()
    if damage == "incomplete":
        rows.pop(-2)
    elif damage == "duplicate":
        rows.insert(-1, copy.deepcopy(rows[2]))
    elif damage == "digest":
        rows[3]["sha"] = "different"
    elif damage == "trace":
        rows[3]["rounds"][0]["committed"] = 4
    elif damage == "barrier":
        rows[0]["barrier"] = True
    elif damage == "capture":
        rows[0]["capture_proposals"] = True
    elif damage == "reps":
        rows[0]["reps"] = 1
    elif damage == "mode":
        rows[0]["mode"] = "mtp"
    elif damage == "missing_source":
        rows[0].pop("python_sha256")
    elif damage == "unfinished":
        rows.pop()
    with pytest.raises(ValueError):
        summarize(rows, receipt)


def test_first_bonus_eos_does_not_require_a_speculative_round():
    spec = importlib.util.spec_from_file_location(
        "shape_benchmark",
        Path(__file__).parents[2] / "scripts/research/bench_dflash_shape.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.trim_trace_budget([], 1, speculative=True)
    with pytest.raises(RuntimeError):
        module.trim_trace_budget([], 2, speculative=True)
    records = [dict(committed=8)]
    module.trim_trace_budget(records, 4, speculative=True)
    assert records == [dict(committed=3, committed_uncut=8)]
