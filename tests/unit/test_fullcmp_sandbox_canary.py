"""Canary: unit tests cannot reach the real GPU queue, the telemetry dir, or the Yunshu raw roots.
A failure here means the sandbox in conftest.py stopped covering a path (2026-11-04 leak)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REAL_QUEUE = "/Volumes/P5Plus/yunshu-gpuq"
REAL_RAW = "/Volumes/Micron/yunshu-fullcmp"


def test_env_points_every_result_dir_into_tmp(tmp_path):
    for var in ("GPUQ_DIR", "TFB_TELEMETRY_DIR", "TFB_OUT", "QUIET_DIR"):
        value = os.environ[var]
        assert value.startswith(str(tmp_path)), (var, value)
        assert not value.startswith(REAL_QUEUE) and not value.startswith(REAL_RAW), var


def test_a_gpuq_submit_on_the_real_queue_is_refused(monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", REAL_QUEUE)
    with pytest.raises(RuntimeError, match="real queue"):
        subprocess.run(["/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq", "submit", "--", "true"])


def test_a_popen_on_the_real_queue_is_refused(monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", REAL_QUEUE)
    with pytest.raises(RuntimeError, match="real queue"):
        subprocess.Popen(["/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq", "run", "--", "true"])
