"""The real-checkpoint gate rejects the two-token garbage the lead observed."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "flashnext_loadcheck",
    Path(__file__).resolve().parents[2] / "scripts/research/flashnext_loadcheck.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_sanity_rejects_garbage_and_api_error():
    with pytest.raises(ValueError, match="sanity"):
        probe.validate({"choices": [{"message": {"content": "r"}}]})
    with pytest.raises(ValueError, match="crash"):
        probe.validate({"error": "crash"})
    assert probe.validate(
        {"choices": [{"message": {"content": "def fib(n): return n"}}]}
    )


def test_dry_run(capsys, tmp_path):
    assert (
        probe.main(
            [
                "--model",
                "fake",
                "--draft",
                "mtp",
                "--out",
                str(tmp_path / "a.json"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"complete": true' in capsys.readouterr().out
