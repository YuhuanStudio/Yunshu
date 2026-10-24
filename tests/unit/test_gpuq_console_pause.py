"""gpuq writes the console pause file while a timing job runs, and removes it otherwise."""

import importlib
import json
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts" / "dev"
NOW = 1_000_000.0


@pytest.fixture
def gpuq(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    sys.path.insert(0, str(DEV))
    import gpuq as module

    module = importlib.reload(module)
    monkeypatch.setattr(module, "_now", lambda: NOW)
    return module


def job(label, state="running", **kw):
    return {
        "id": label,
        "label": label,
        "state": state,
        "started": NOW - 10,
        "timeout_s": 600,
        **kw,
    }


def read(gpuq):
    return json.loads((gpuq.ROOT / "console-pause.json").read_text())


@pytest.mark.parametrize("priority", [-1, 0, 1])
@pytest.mark.parametrize(
    "extra,label", [({"quiet": True}, "ab-arm1"), ({}, "q-speed-27b")]
)
def test_timing_job_writes_the_pause_file(gpuq, priority, extra, label):
    gpuq.sync_console_pause([job(label, priority=priority, **extra)])
    assert read(gpuq) == {"pause": True, "until": NOW - 10 + 600, "reason": label}


def test_non_timing_pending_paused_or_finished_jobs_do_not(gpuq):
    path = gpuq.ROOT / "console-pause.json"
    gpuq.sync_console_pause([job("smoke")])
    assert not path.exists()
    gpuq.sync_console_pause([job("a-speed", state="pending", quiet=True)])
    assert not path.exists()
    gpuq.sync_console_pause([job("a-speed", paused=True)])
    assert not path.exists()
    gpuq.sync_console_pause([job("a-speed", state="done")])
    assert not path.exists()


def test_file_is_removed_when_the_timing_job_ends_and_stale_ones_on_start(gpuq):
    gpuq.ROOT.mkdir(parents=True, exist_ok=True)
    path = gpuq.ROOT / "console-pause.json"
    path.write_text('{"pause": true, "until": null, "reason": "dead daemon"}')
    gpuq.sync_console_pause([job("x-speed")])
    assert read(gpuq)["reason"] == "x-speed"
    gpuq.sync_console_pause([job("x-speed", state="done")])
    assert not path.exists()
    gpuq.sync_console_pause([])  # nothing to remove is fine
