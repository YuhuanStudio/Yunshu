"""yv preflight runs every related test (no -x), names every FAILED id in full, and reruns only
the failed tests once in isolation: a test that passes there is flaky, not a failure
(2026-11-02: 24 preflight failures, most of them known-flaky tests, and the -x stopped the run
before the real failures were visible)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify import stages  # noqa: E402
from verify.core import RunDir  # noqa: E402

FLAKY = "tests/unit/test_flaky.py::test_timing"
REAL = "tests/unit/test_real.py::test_broken[4-3072]"
OTHER = "tests/unit/test_real.py::test_also_broken"


class FakePopen:
    def __init__(self, rc, stdout):
        self.returncode, self.stdout, self.stderr = rc, stdout, ""


def _patch(monkeypatch, tests, runs):
    """runs: list of (rc, stdout) answers for the pytest calls, in order."""
    calls = []

    def fake_run(cmd, *a, **kw):
        if "-c" in cmd:  # import checks pass; no xdist, so one pytest run per arm
            return FakePopen(1 if "xdist" in cmd[-1] else 0, "ok")
        calls.append(list(cmd))
        rc, out = runs.pop(0)
        return FakePopen(rc, out)

    monkeypatch.setattr(stages.subprocess, "run", fake_run)
    monkeypatch.setattr(stages, "changed_files", lambda base, cand: ["python/x.py"])
    monkeypatch.setattr(stages, "related_tests", lambda changed, path: list(tests))
    return calls


def _ctx(tmp_path):
    base = stages.Arm("base", "b", "a" * 40, tmp_path / "base", "")
    cand = stages.Arm("cand", "c", "b" * 40, tmp_path / "cand", "")
    return SimpleNamespace(
        run=RunDir(tmp_path / "run"),
        exe=SimpleNamespace(jobs=[]),
        base=base,
        cand=cand,
        py=sys.executable,
        suite={},
    )


def _first_run(*failed):
    lines = [
        "collected 3 items",
        *[f"FAILED {t} - AssertionError: nope" for t in failed],
    ]
    lines += ["=== short test summary info ===", "1 failed, 2 passed in 3.00s"]
    return 1, "\n".join(lines)


def test_flaky_test_passing_on_isolated_rerun_does_not_fail(monkeypatch, tmp_path):
    calls = _patch(
        monkeypatch,
        [FLAKY, REAL],
        [
            _first_run(FLAKY),
            (0, f"PASSED {FLAKY}\n1 passed in 0.5s"),
        ],
    )
    res = stages.stage_preflight(_ctx(tmp_path))
    assert res.passed, res.reasons
    assert FLAKY in res.numbers["flaky"]
    assert all("-x" not in c for c in calls), calls
    # the rerun names the failed test only, once
    assert FLAKY in calls[1] and REAL not in calls[1]


def test_test_failing_again_on_isolated_rerun_fails_with_full_names(
    monkeypatch, tmp_path
):
    _patch(
        monkeypatch,
        [FLAKY, REAL, OTHER],
        [
            _first_run(REAL, OTHER),
            (1, f"FAILED {REAL} - AssertionError: nope\n1 failed, 1 passed"),
        ],
    )
    res = stages.stage_preflight(_ctx(tmp_path))
    assert not res.passed
    text = "\n".join(res.reasons)
    assert REAL in text
    assert OTHER in text  # not rerun-passed: it is still a failure, named in full
    assert "flaky" not in text


def test_no_failed_ids_but_nonzero_rc_still_fails(monkeypatch, tmp_path):
    _patch(monkeypatch, [REAL], [(2, "INTERNALERROR> boom\n")])
    res = stages.stage_preflight(_ctx(tmp_path))
    assert not res.passed
    assert "rc 2" in "\n".join(res.reasons)


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            ["FAILED a.py::t[1 - 2] - x"],
            ["a.py::t[1"],
        ),  # split on the message separator
        (["ERROR b.py - KeyError: 'X'", "FAILED c.py::t"], ["b.py", "c.py::t"]),
        (["FAILED d.py::t", "FAILED d.py::t"], ["d.py::t"]),
    ],
)
def test_failed_ids_are_parsed_once_and_untruncated(lines, expected):
    assert stages._failed_test_ids(lines) == expected
