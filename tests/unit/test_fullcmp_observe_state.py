"""Observe state of fullcmp Yunshu cells: timed reps observe OFF, profile reps observe ON.

Pure unit tests: no server, no GPU, no gpuq. The audit mutation cases live in test_fullcmp_audit.py
(BAD_CASES, guard 'observe_state'); this file covers the runner env, the tfbench refusal and the
guard's profile-root path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import fullcmp_audit as fa  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_run_observe", ROOT / "scripts" / "research" / "fullcmp_run.py"
)
run = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_run_observe"] = run
_SPEC.loader.exec_module(run)

PIN = {
    "sha": "0f2960047855546a9f13b2631b680352b1c57cfe",
    "python_executable": "/venv/bin/python",
    "launch": {
        "harness_env": {
            "TFB_YUNSHU_BIN": "/venv/bin/yunshu",
            "TFB_YUNSHU_SRC": "/tree/python",
            "PYTHONPATH": "/tree/python",
        }
    },
}


# --- runner: the cell env decides observe -------------------------------------------------


def test_timed_cell_env_exports_observe_off_and_no_profile_marker():
    env = run.cell_env("yunshu", PIN, "lbl", group="g", profile=False)
    assert "TFB_OBSERVE=0" in env
    assert "TFB_OBSERVE=1" not in env
    assert not any(t.startswith("TFB_PROFILE") for t in env)


def test_profile_cell_env_exports_observe_on_with_profile_marker():
    env = run.cell_env("yunshu", PIN, "lbl", group="g", profile=True)
    assert "TFB_OBSERVE=1" in env and "TFB_PROFILE=1" in env
    assert "TFB_OBSERVE=0" not in env


def test_every_cell_env_states_observe_explicitly():
    for engine in ("yunshu", "mlx-serve"):
        for prof in (False, True):
            env = run.cell_env(engine, PIN, "lbl", group="g", profile=prof)
            assert sum(t.startswith("TFB_OBSERVE=") for t in env) == 1


# --- provenance: fail closed on the observe state -----------------------------------------


@pytest.mark.parametrize("raw", [None, "1", "", "true"])
def test_timed_yunshu_cell_refuses_unset_or_on(raw):
    environ = {} if raw is None else {"TFB_OBSERVE": raw}
    with pytest.raises(
        fp.ProvenanceError, match="timed Yunshu cell requires TFB_OBSERVE=0"
    ):
        fp.observe_state(environ, True)


def test_timed_yunshu_cell_accepts_observe_off():
    assert fp.observe_state({"TFB_OBSERVE": "0"}, True) is False


@pytest.mark.parametrize("raw", [None, "0", ""])
def test_profile_yunshu_cell_refuses_observe_off_or_unset(raw):
    environ = {"TFB_PROFILE": "1"}
    if raw is not None:
        environ["TFB_OBSERVE"] = raw
    with pytest.raises(
        fp.ProvenanceError, match="profile Yunshu cell requires TFB_OBSERVE=1"
    ):
        fp.observe_state(environ, True)


def test_profile_yunshu_cell_accepts_observe_on():
    assert fp.observe_state({"TFB_PROFILE": "1", "TFB_OBSERVE": "1"}, True) is True


def test_competitor_cells_have_no_observe_state():
    assert fp.observe_state({}, False) is None
    assert fp.observe_state({"TFB_OBSERVE": "1"}, False) is None


# --- audit: the profile-root path of guard observe_state ----------------------------------


def _manifest():
    cell = {"engine": "yunshu", "cell": "q27_decode-1024-code-p0", "group": "g"}
    return cell, {"run_id": "r", "reps": 1, "cells": [cell], "groups": {}, "pins": {}}


def test_profile_rep_with_observe_off_fails_the_guard(tmp_path):
    cell, manifest = _manifest()
    prof_dir = tmp_path / "profile" / "yunshu"
    prof_dir.mkdir(parents=True)
    session = {
        "part": "session",
        "observe": False,
        "yunshu_observe": None,
        "profile": True,
    }
    (prof_dir / f"{cell['cell']}-prof.jsonl").write_text(json.dumps(session) + "\n")
    head = {
        "part": "session",
        "observe": False,
        "yunshu_observe": None,
        "profile": False,
    }
    run_obj = fa.Run(
        manifest,
        [fa.CellFile(cell, 0, Path("x"), head, [])],
        [],
        [],
        [],
        profile_root=tmp_path / "profile",
    )
    res = fa.check_observe_state(run_obj)
    assert any(
        "profile rep: observe=False but profile=True" in v for v in res.violations
    )


def test_profile_rep_without_session_header_fails_the_guard(tmp_path):
    cell, manifest = _manifest()
    prof_dir = tmp_path / "profile" / "yunshu"
    prof_dir.mkdir(parents=True)
    (prof_dir / f"{cell['cell']}-prof.jsonl").write_text(
        json.dumps({"part": "profile"}) + "\n"
    )
    head = {
        "part": "session",
        "observe": False,
        "yunshu_observe": None,
        "profile": False,
    }
    run_obj = fa.Run(
        manifest,
        [fa.CellFile(cell, 0, Path("x"), head, [])],
        [],
        [],
        [],
        profile_root=tmp_path / "profile",
    )
    res = fa.check_observe_state(run_obj)
    assert any("no session header" in v for v in res.violations)


def test_good_timed_and_profile_headers_pass_the_guard(tmp_path):
    cell, manifest = _manifest()
    prof_dir = tmp_path / "profile" / "yunshu"
    prof_dir.mkdir(parents=True)
    session = {
        "part": "session",
        "observe": True,
        "yunshu_observe": "1",
        "profile": True,
    }
    (prof_dir / f"{cell['cell']}-prof.jsonl").write_text(json.dumps(session) + "\n")
    head = {
        "part": "session",
        "observe": False,
        "yunshu_observe": None,
        "profile": False,
    }
    run_obj = fa.Run(
        manifest,
        [fa.CellFile(cell, 0, Path("x"), head, [])],
        [],
        [],
        [],
        profile_root=tmp_path / "profile",
    )
    assert fa.check_observe_state(run_obj).violations == []


# --- runner: preflight label unique per invocation; a refused submit fails loudly ---------


def _label_of(argv):
    return argv[argv.index("--label") + 1]


def test_two_preflight_invocations_of_one_batch_get_distinct_labels(monkeypatch):
    seen = []

    class Done:
        returncode = 0

    def fake_run(argv, **kw):
        seen.append(_label_of(argv))
        return Done()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(run, "PREFLIGHT_DIR", Path("/nonexistent-ok/x"), raising=False)
    monkeypatch.setattr(run.Path, "mkdir", lambda self, **k: None)
    assert run.swap_gate("SMOKE") == 0
    assert run.swap_gate("SMOKE") == 0
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(lbl.startswith("fullcmp-preflight-SMOKE-") for lbl in seen)


def test_refused_submit_exits_nonzero_and_names_the_label(monkeypatch, capsys):
    calls = []

    class Res:
        def __init__(self, rc, out):
            self.returncode, self.stdout, self.stderr = (
                rc,
                out,
                "duplicate active label",
            )

    def fake_run(argv, **kw):
        calls.append(argv)
        return Res(1, "")

    class Job:
        label = "fullcmp-x-yunshu-g-r0"

        def submit_argv(self):
            return ["gpuq", "submit"]

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(run, "write_wrapper", lambda job: None)
    ids, bad = run.submit_all([Job(), Job()])
    assert bad == 1 and ids == []
    assert len(calls) == 1  # stops at the refusal: the second job is not submitted
    assert "SUBMIT REFUSED label=fullcmp-x-yunshu-g-r0" in capsys.readouterr().err
