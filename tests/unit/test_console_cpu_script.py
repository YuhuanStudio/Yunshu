"""CPU checks of scripts/research/console_cpu.py before its first GPU job."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "console_cpu.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("console_cpu", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_summary_statistics():
    mod = load()
    s = mod.summarize(
        [{"cpu": c, "rss_mib": 60.0 + i} for i, c in enumerate([0.1, 0.2, 0.3, 4.0])]
    )
    assert s["n"] == 4 and s["cpu_max_pct"] == 4.0 and s["cpu_median_pct"] == 0.25
    assert (
        s["rss_mib_start"] == 60.0
        and s["rss_mib_end"] == 63.0
        and s["rss_growth_mib"] == 3.0
    )
    assert mod.summarize([]) == {"n": 0}


def test_the_sampler_measures_a_busy_process_and_an_idle_one():
    mod = load()
    busy = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    idle = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.5)
        b = mod.sampler(busy.pid, 2 / 60, every=0.5)
        i = mod.sampler(idle.pid, 2 / 60, every=0.5)
    finally:
        busy.kill()
        idle.kill()
    assert b and i
    assert mod.summarize(b)["cpu_mean_pct"] > 50
    assert mod.summarize(i)["cpu_mean_pct"] < 5


def test_help_and_no_pkill():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0 and "--mode" in out.stdout
    assert "pkill" not in SCRIPT.read_text() and "stop_tree(proc)" in SCRIPT.read_text()
