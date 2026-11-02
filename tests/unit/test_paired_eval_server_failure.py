"""paired-eval names the real reason a server did not start: the log path and its last lines,
with the ImportError first (2026-11-02: the paired-eval venv lacked xxhash, and the job said only
"server failed to start")."""

import importlib.util
from pathlib import Path

_p = Path(__file__).parents[2] / "scripts/research/accuracy/paired_eval.py"
_spec = importlib.util.spec_from_file_location("paired_eval", _p)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


def test_failure_report_shows_the_import_error_and_the_log(tmp_path):
    log = tmp_path / "mmlu_pro-base.log"
    lines = [f"booting line {i}" for i in range(60)]
    lines.append("Traceback (most recent call last):")
    lines.append("ModuleNotFoundError: No module named 'xxhash'")
    log.write_text("\n".join(lines) + "\n")
    text = pe.server_failure_report(log, rc=1)
    assert str(log) in text
    assert "ModuleNotFoundError: No module named 'xxhash'" in text
    assert "rc=1" in text
    assert "booting line 0" not in text  # only the tail


def test_failure_report_without_a_log(tmp_path):
    text = pe.server_failure_report(tmp_path / "missing.log", rc=None)
    assert "no server log" in text
