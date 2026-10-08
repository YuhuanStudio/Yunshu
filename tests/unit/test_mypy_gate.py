"""A missing checker is an infrastructure failure, never a zero-error pass."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mypy_gate", ROOT / "scripts/dev/mypy_gate.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_missing_mypy_fails_closed(monkeypatch, capsys):
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="No module named mypy\n"
        ),
    )
    assert gate.main([]) == 2
    assert "No module named mypy" in capsys.readouterr().err


def test_actual_type_error_still_uses_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "BASELINE", tmp_path / "missing")
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1,
            stdout="python/new.py:1: error: Bad type [arg-type]\n",
            stderr="",
        ),
    )
    assert gate.main([]) == 1
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="Success: no issues found\n", stderr=""
        ),
    )
    assert gate.main([]) == 0
