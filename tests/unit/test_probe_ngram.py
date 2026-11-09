"""CPU admission and fail-closed evidence checks for the n-gram probe."""

import copy

import pytest
from scripts.research.probe_ngram import CASES, parser, validate_records


def records(expect="engaged"):
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
                        sha="x",
                        engaged=arm == "on"
                        and expect == "engaged"
                        and case != "prose_short",
                        match=500,
                    )
                )
    return rows + [dict(complete=True, success=True)]


def test_arguments():
    a = parser().parse_args(
        [
            "--src",
            "/s",
            "--model",
            "/m",
            "--expect",
            "plain",
            "--out",
            "/o",
            "--dry-run",
        ]
    )
    assert a.reps == 3 and a.dry_run


def test_good_records_pass():
    assert validate_records(records(), 2, "engaged") == (True, "")
    assert validate_records(records("plain"), 2, "plain") == (True, "")


@pytest.mark.parametrize(
    "fault", ["missing", "complete", "rate", "never", "diverged", "offengaged"]
)
def test_engaged_fails_closed(fault):
    rows = copy.deepcopy(records())
    on = next(r for r in rows if r.get("arm") == "on" and r["case"] == "repeat_short")
    if fault == "missing":
        rows.pop(0)
    elif fault == "complete":
        rows[-1]["success"] = False
    elif fault == "rate":
        rows[0]["dec_tps"] = 0
    elif fault == "never":
        on["engaged"] = False
    elif fault == "diverged":
        on["match"] = 3
    else:
        next(r for r in rows if r.get("arm") == "off")["engaged"] = True
    assert validate_records(rows, 2, "engaged")[0] is False


def test_plain_requires_identity_and_no_engagement():
    rows = records("plain")
    next(r for r in rows if r.get("arm") == "on")["sha"] = "y"
    assert validate_records(rows, 2, "plain")[0] is False
    rows = records("plain")
    next(r for r in rows if r.get("arm") == "on")["engaged"] = True
    assert validate_records(rows, 2, "plain")[0] is False
