"""CPU admission and fail-closed evidence checks for the draft-LM / assistant probe."""

import copy

import pytest
from scripts.research.probe_draft import (
    CASES,
    parse_row_log,
    parser,
    validate_records,
)


def records():
    rows = []
    for r in range(2):
        for case in CASES:
            for arm in ("off", "on"):
                rows.append(
                    dict(
                        phase="decode",
                        arm=arm,
                        case=case,
                        rep=r,
                        dec_tps=50.0,
                        ct=100,
                        pt=40,
                        sha="x",
                        engaged=arm == "on" and not case.endswith("_long"),
                        match=500,
                    )
                )
    return rows + [dict(complete=True, success=True)]


def test_arguments():
    a = parser().parse_args(
        ["--src", "/s", "--model", "/m", "--drafter", "assistant", "--out", "/o"]
    )
    assert a.reps == 3 and not a.dry_run


def test_good_records_pass():
    assert validate_records(records(), 2) == (True, "")


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "complete",
        "never",
        "diverged",
        "offengaged",
        "longengaged",
        "longdiff",
        "nopt",
    ],
)
def test_fails_closed(fault):
    rows = copy.deepcopy(records())
    short = next(r for r in rows if r.get("arm") == "on" and r["case"] == "prose_short")
    long_on = next(
        r for r in rows if r.get("arm") == "on" and r["case"] == "prose_long"
    )
    if fault == "missing":
        rows.pop(0)
    elif fault == "complete":
        rows[-1]["success"] = False
    elif fault == "never":
        short["engaged"] = False
    elif fault == "diverged":
        short["match"] = 3
    elif fault == "offengaged":
        next(r for r in rows if r.get("arm") == "off")["engaged"] = True
    elif fault == "longengaged":
        long_on["engaged"] = True
    elif fault == "longdiff":
        long_on["sha"] = "y"
    else:
        short.pop("pt")
    assert validate_records(rows, 2)[0] is False


def test_row_log_parse():
    log = "x\nRound driver gemma4-assistant row: 12 windows, 60 drafted, 51 accepted, 120 tokens\n"
    assert parse_row_log(log) == (12, 60, 51, 120)
    assert parse_row_log("nothing") is None
