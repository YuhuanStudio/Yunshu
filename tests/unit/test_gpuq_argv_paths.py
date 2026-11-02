"""gpuq submit refuses a job whose absolute input path does not exist, before its queue wait
(2026-11-02: 15 jobs died in 0 s on a missing prompt file; output paths must still be allowed)."""

import importlib.util
import sys
from pathlib import Path

DEV = Path(__file__).resolve().parents[2] / "scripts/dev"
spec = importlib.util.spec_from_file_location(
    "gpuq_preflight", DEV / "gpuq_preflight.py"
)
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

ECHO = [sys.executable, "-c", "pass"]  # a non-script exe: only the argv checks run


def test_missing_absolute_input_path_is_refused(tmp_path):
    missing = tmp_path / "prompts" / "code-32768.txt"
    problems = pf.preflight([*ECHO, "--prompts", str(missing)], str(tmp_path), {})
    assert any("path does not exist" in p and str(missing) in p for p in problems), (
        problems
    )


def test_missing_path_after_flag_equals_is_refused(tmp_path):
    missing = tmp_path / "corpus"
    problems = pf.preflight([*ECHO, f"--corpus={missing}"], str(tmp_path), {})
    assert any(str(missing) in p for p in problems), problems


def test_existing_input_and_new_output_paths_are_allowed(tmp_path):
    real = tmp_path / "in.txt"
    real.write_text("x")
    new_out = tmp_path / "results" / "new.jsonl"  # not created yet: it is an output
    problems = pf.preflight(
        [*ECHO, str(real), "--out", str(new_out), "--log", str(tmp_path / "x.log")],
        str(tmp_path),
        {},
    )
    assert not problems, problems


def test_env_assignments_and_relative_args_are_not_path_checked(tmp_path):
    problems = pf.preflight(
        ["TFB_OUT=/nonexistent/out", *ECHO, "relative/missing.txt", "--n=3"],
        str(tmp_path),
        {},
    )
    assert not problems, problems
