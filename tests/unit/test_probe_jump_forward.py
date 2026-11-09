"""CPU admission and fail-closed evidence checks for the forced-window probe."""

import copy

import pytest
from scripts.research.probe_jump_forward import parser, validate_records


def records():
    return [
        dict(
            phase="decode",
            arm=a,
            case=case,
            rep=r,
            dec_tps=100.0,
            ct=100,
            sha="identical",
            schema_valid=True,
            engaged=a == "cand",
        )
        for r in range(3)
        for case in ("schema", "forced_literal")
        for a in ("base", "cand")
    ] + [dict(complete=True, success=True)]


def test_jump_probe_arguments():
    a = parser().parse_args(
        [
            "--base-src",
            "/base",
            "--cand-src",
            "/cand",
            "--model",
            "/model",
            "--out",
            "/out",
            "--dry-run",
        ]
    )
    assert a.reps == 3 and a.tokens == 512 and a.dry_run


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "identity",
        "tokens",
        "schema",
        "engagement",
        "rate",
        "complete",
    ],
)
def test_jump_probe_fails_closed(fault):
    rows = copy.deepcopy(records())
    assert validate_records(rows, 3) == (True, "")
    if fault == "missing":
        del rows[0]
    elif fault == "duplicate":
        rows[0] = rows[1]
    elif fault == "identity":
        rows[1]["sha"] = "different"
    elif fault == "tokens":
        rows[1]["ct"] += 1
    elif fault == "schema":
        rows[1]["schema_valid"] = False
    elif fault == "engagement":
        rows[3]["engaged"] = False
    elif fault == "rate":
        rows[1]["dec_tps"] = float("nan")
    else:
        rows[-1]["complete"] = False
    assert not validate_records(rows, 3)[0]


def test_jump_stage_refuses_missing_cell_or_evidence(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts.verify import stages

    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    for results in ({}, {"paired": SimpleNamespace(ok=True, evidence=None)}):
        ctx = SimpleNamespace(
            cand=SimpleNamespace(path=tmp_path),
            base=SimpleNamespace(path=tmp_path),
            py="python",
            model="model",
            mem_gb=8,
            suite={"reps": 3},
            exe=SimpleNamespace(run_cells=lambda cells: results),
        )
        result = stages.stage_jump_forward(ctx)
        assert not result.passed
        assert "missing paired JSON-schema evidence" in result.reasons
