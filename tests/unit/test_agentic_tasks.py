"""Agentic benchmark: task fixtures are valid and the summary statistics are right."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "agentic")
)
import run_agentic  # noqa: E402
import tasks as tasklib  # noqa: E402


def test_wilson():
    lo, hi = run_agentic.wilson(5, 10)
    assert 0.23 < lo < 0.25 and 0.75 < hi < 0.77
    assert run_agentic.wilson(0, 0) == (0.0, 0.0)
    lo, hi = run_agentic.wilson(10, 10)
    assert hi == 1.0 and lo > 0.69


def test_summarize(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    row = dict(
        type="run",
        engine="e",
        agent="opencode",
        task="t",
        repeat=1,
        passed=True,
        wall_s=10.0,
        requests=3,
        prompt_tokens=1000,
        cached_tokens=600,
        completion_tokens=50,
        api_errors=0,
        malformed_tool_calls=1,
        leaked_tool_markup=0,
        server_peak_gib=20.0,
        request_log=[{"decode_tok_s": 30.0, "ttft_s": 0.5}],
    )
    p.write_text(json.dumps({"type": "meta"}) + "\n" + json.dumps(row) + "\n")
    run_agentic.cmd_summarize(type("A", (), {"files": [str(p)]}))
    out = capsys.readouterr().out
    assert "| e | opencode | 1 | 1 |" in out and "60.0%" in out


@pytest.mark.skipif(not Path(tasklib.PYTHON).exists(), reason="agentic pyenv missing")
def test_custom_tasks_reference_passes(tmp_path):
    import shutil
    import subprocess

    for t in tasklib.custom_tasks():
        w = tmp_path / t.id
        t.stage(w)
        assert not t.grade(w)[0], f"{t.id}: starting repo must fail"
        ref = t.src / "reference"
        shutil.copytree(ref, w, dirs_exist_ok=True)
        if (w / "apply.py").exists():
            (w / "apply.py").unlink()
            subprocess.run([tasklib.PYTHON, str(ref / "apply.py"), str(w)], check=True)
        ok, tail = t.grade(w)
        assert ok, f"{t.id}: reference must pass\n{tail}"
