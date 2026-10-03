"""Serving preflight failures uphold the global JSON contract for agents."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["--port", "65536"], "Cannot listen"),
        (["org/first", "-m", "org/second"], "disagree"),
        (["--set", "NOT_A_SETTING=1"], "unknown setting"),
        (["--set", "MAX_CONCURRENT=not-an-integer"], "YUNSHU_MAX_CONCURRENT"),
    ],
)
def test_serve_preflight_json_error(tmp_path, arguments, expected):
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("YUNSHU_")
    }
    env.update(
        HOME=str(tmp_path), PYTHONPATH=str(ROOT / "python"), PYTHONDONTWRITEBYTECODE="1"
    )
    result = subprocess.run(
        [sys.executable, "-m", "yunshu_cli", "--json", "serve", *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2, result.stderr
    body = json.loads(result.stdout)
    assert expected in body["error"]
    assert set(body) == {"error"}
