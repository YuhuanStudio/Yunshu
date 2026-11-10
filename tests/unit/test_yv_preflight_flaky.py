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


def test_preflight_tests_use_marked_candidate_python(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    own = ctx.cand.path / ".venv/bin/python"
    own.parent.mkdir(parents=True)
    own.write_text("fake candidate interpreter")
    (ctx.cand.path / ".yv-own-venv").touch()
    calls = _patch(monkeypatch, [REAL], [(0, "1 passed")])
    assert stages.stage_preflight(ctx).passed
    assert str(own) in calls[0]


def test_mlx_preflight_queues_tests_and_requires_complete_evidence(
    monkeypatch, tmp_path
):
    import json

    ctx = _ctx(tmp_path)
    submitted = []

    class Queue:
        def submit(self, label, argv, **kwargs):
            from scripts.dev.gpuq_preflight import preflight

            ctx.cand.path.mkdir(exist_ok=True)
            assert preflight(argv, str(ctx.cand.path), {}) == []
            submitted.append((argv, kwargs))
            kwargs["out"].write_text(
                json.dumps(
                    {"complete": True, "rc": 0, "stdout": "1 passed", "stderr": ""}
                )
            )
            return "preflight-job"

        def wait(self, jid):
            return SimpleNamespace(rc=0)

    ctx.exe.gq = Queue()
    ctx.exe.label = "deps-test"
    ctx.exe.priority = 0
    monkeypatch.setenv("YV_PREFLIGHT_GPUQ", "1")
    calls = _patch(monkeypatch, [REAL], [])
    result = stages.stage_preflight(ctx)
    assert result.passed
    assert calls == []  # no pytest process outside the queue
    assert submitted[0][1]["expect_complete"] is True
    assert submitted[0][1]["cwd"] == ctx.cand.path
    assert result.jobs == [("preflight", "tests-0", "preflight-job")]


def test_queued_preflight_rejects_missing_completion(monkeypatch, tmp_path):
    import json

    ctx = _ctx(tmp_path)

    class Queue:
        def submit(self, label, argv, **kwargs):
            kwargs["out"].write_text(json.dumps({"rc": 0}))
            return "incomplete-job"

        def wait(self, jid):
            return SimpleNamespace(rc=0)

    ctx.exe.gq = Queue()
    ctx.exe.label = "deps-test"
    ctx.exe.priority = 0
    monkeypatch.setenv("YV_PREFLIGHT_GPUQ", "1")
    with pytest.raises(stages.InfraError, match="incomplete test evidence"):
        stages._run_preflight_tests(ctx, ["python", "-m", "pytest"], {}, "tests")


def test_queued_child_does_not_requeue_nested_verifier_tests(monkeypatch, tmp_path):
    import json
    import os
    import subprocess

    spec, out = tmp_path / "command.json", tmp_path / "result.json"
    spec.write_text(
        json.dumps(
            {
                "cmd": [
                    sys.executable,
                    "-c",
                    "import os; print(os.getenv('YV_PREFLIGHT_GPUQ', 'unset'))",
                ],
                "env": {},
            }
        )
    )
    subprocess.run(
        [sys.executable, "-c", stages._PREFLIGHT_CHILD, str(spec), str(out)],
        env=dict(os.environ, YV_PREFLIGHT_GPUQ="1"),
        check=True,
    )
    assert json.loads(out.read_text())["stdout"].strip() == "unset"
