"""gpuq submit refuses a label with whitespace and an argv element that is several arguments
glued together (2026-11-12: zsh does not word-split $VAR, so `--engine "yunshu-tf4 1024"` reached
the queue as one argument and three 27B census jobs died in 0 s with rc=2)."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts/dev"
spec = importlib.util.spec_from_file_location(
    "gpuq_preflight", DEV / "gpuq_preflight.py"
)
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

HINT = "zsh does not word-split $VAR"


@pytest.fixture
def gpuq(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(DEV))
    import gpuq as module

    module = importlib.reload(module)
    # a unit test must not start the queue daemon
    monkeypatch.setattr(module, "_ensure_daemon", lambda: None)
    return module


def test_label_with_whitespace_is_refused():
    problems = pf.label_problems("1112-220145-00-gap27b-yunshu-tf4 1024-")
    assert problems and "whitespace" in problems[0], problems


def test_plain_label_is_allowed():
    assert pf.label_problems("1112-220145-00-gap27b-yunshu-tf4-1024") == []


def test_submit_refuses_label_with_whitespace(gpuq):
    with pytest.raises(ValueError, match="whitespace"):
        gpuq.submit([sys.executable, "-c", "pass"], "gap 27b", 5, 0)
    assert not gpuq.JOBS.exists() or not any(gpuq.JOBS.glob("*.json"))


def test_glued_option_and_number_is_refused(tmp_path):
    cmd = ["python3", "census.py", "--engine", "yunshu-tf4 1024"]
    problems = pf.spaced_arg_problems(cmd, str(tmp_path))
    assert len(problems) == 1, problems
    assert "yunshu-tf4 1024" in problems[0]
    assert HINT in problems[0]
    assert "${=VAR}" in problems[0]


def test_glued_option_flag_is_refused(tmp_path):
    problems = pf.spaced_arg_problems(["python3", "x.py", "a --flag"], str(tmp_path))
    assert problems and "a --flag" in problems[0], problems


def test_space_joined_existing_paths_are_refused(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    problems = pf.spaced_arg_problems(["python3", "x.py", "a.txt b.txt"], str(tmp_path))
    assert problems and "a.txt b.txt" in problems[0], problems


def test_spaced_prompt_is_allowed(tmp_path):
    cmd = ["python3", "x.py", "--prompt", "hello world", "--n", "3"]
    assert pf.spaced_arg_problems(cmd, str(tmp_path)) == []


def test_single_token_and_unspaced_args_are_allowed(tmp_path):
    cmd = ["python3", "x.py", "--engine=yunshu-tf4", "--out", "/nonexistent/o.json"]
    assert pf.spaced_arg_problems(cmd, str(tmp_path)) == []


def test_env_assignment_with_spaces_is_allowed(tmp_path):
    cmd = ["TFB_NOTE=hello 1", "python3", "x.py"]
    assert pf.spaced_arg_problems(cmd, str(tmp_path)) == []


def test_submit_refuses_glued_argument(gpuq, tmp_path):
    cmd = [sys.executable, "-c", "pass", "--engine", "yunshu-tf4 1024"]
    with pytest.raises(ValueError, match="yunshu-tf4 1024"):
        gpuq.submit(cmd, "gap-ok", 5, 0)


def test_submit_allow_spaced_args_escape(gpuq):
    cmd = [sys.executable, "-c", "pass", "--engine", "yunshu-tf4 1024"]
    jid = gpuq.submit(cmd, "gap-ok", 5, 0, allow_spaced_args=True)
    assert jid.endswith("gap-ok")


@pytest.mark.parametrize(
    "cmd",
    [
        ["bash", "-lc", "cd /Volumes/P5Plus/x && uv run pytest tests/unit -q"],
        ["python3", "-c", "import sys; print(sys.argv[1:])"],
        ["bash", "-c", "export A=1; /usr/bin/true --flag 2"],
    ],
)
def test_scripts_for_an_interpreter_are_allowed(cmd, tmp_path):
    # 81 of the last 4000 real jobs were shell command lines like these; they must still queue.
    assert pf.spaced_arg_problems(cmd, str(tmp_path)) == []
