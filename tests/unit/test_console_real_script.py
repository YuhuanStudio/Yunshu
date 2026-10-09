"""CPU checks of scripts/research/console_real_server.py before its first GPU job."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "console_real_server.py"


def load():
    spec = importlib.util.spec_from_file_location("console_real_server", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


GOOD_ROW = {
    "request_id": "req_1",
    "t": 1.0,
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "ttft_ms": 120.0,
    "model": "m",
}


def good(n: int = 2):
    history = {"data": [dict(GOOD_ROW, request_id=f"r{i}") for i in range(n)]}
    metrics = {"series": {"t": list(range(30)), "active_gb": [2.0] * 30}, "gaps": []}
    stream = {"first_event_before_end": True}
    after = {
        "console_up": True,
        "state_up": False,
        "event_kinds": ["engine_unreachable"],
        "history_rows": n,
    }
    return n, history, metrics, 40.0, stream, after


def test_a_complete_run_passes():
    assert load().evaluate(*good()) == []


def test_each_failure_is_reported_not_swallowed():
    mod = load()
    n, history, metrics, secs, stream, after = good()
    assert any(
        "request log has" in p
        for p in mod.evaluate(5, history, metrics, secs, stream, after)
    )
    bad_row = {
        "data": [dict(GOOD_ROW, completion_tokens=0), dict(GOOD_ROW, ttft_ms=None)]
    }
    out = mod.evaluate(2, bad_row, metrics, secs, stream, after)
    assert any("token counts" in p for p in out) and any("TTFT" in p for p in out)
    leaky = {"data": [dict(GOOD_ROW, messages="x"), dict(GOOD_ROW)]}
    assert any(
        "content" in p for p in mod.evaluate(2, leaky, metrics, secs, stream, after)
    )
    few = {"series": {"t": [1, 2], "active_gb": [2.0, 2.0]}, "gaps": []}
    assert any(
        "metrics history has" in p
        for p in mod.evaluate(n, history, few, secs, stream, after)
    )
    nomem = {"series": {"t": list(range(30)), "active_gb": [None] * 30}, "gaps": []}
    assert any(
        "Metal memory" in p
        for p in mod.evaluate(n, history, nomem, secs, stream, after)
    )
    gappy = dict(metrics, gaps=[[1, 2]])
    assert any(
        "gaps" in p for p in mod.evaluate(n, history, gappy, secs, stream, after)
    )
    assert any(
        "did not stream" in p
        for p in mod.evaluate(
            n, history, metrics, secs, {"first_event_before_end": False}, after
        )
    )
    dead = dict(after, console_up=False)
    assert any(
        "did not survive" in p
        for p in mod.evaluate(n, history, metrics, secs, stream, dead)
    )
    unaware = dict(after, state_up=True, event_kinds=[])
    out = mod.evaluate(n, history, metrics, secs, stream, unaware)
    assert any("did not notice" in p for p in out) and any(
        "engine_unreachable" in p for p in out
    )
    assert any(
        "stopped answering" in p
        for p in mod.evaluate(
            n, history, metrics, secs, stream, dict(after, history_rows=0)
        )
    )


def test_the_script_parses_its_arguments_and_has_help():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0 and "--console-port" in out.stdout


def test_the_script_never_uses_pkill_and_kills_its_process_group():
    text = SCRIPT.read_text()
    assert (
        "pkill" not in text
        and "killpg(proc.pid" in text
        and "start_new_session=True" in text
    )
