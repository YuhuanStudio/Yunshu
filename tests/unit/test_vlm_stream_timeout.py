"""two HIGH bugs in the VLM streaming timeout path.
(1) it read kwargs.get('timeout') but the gateway passes timeout_seconds=req.timeout —
    so a user timeout was silently ignored (hardcoded 300s).
(2) the timeout branch did NOT set cancel_event, so the executor thread kept decoding to
    max_tokens (stream_task.cancel() can't stop the executor) — the class."""

from __future__ import annotations

import asyncio
import pathlib
import threading


def _src():
    root = pathlib.Path(__file__).resolve().parents[2]
    return (root / "python/yunshu_engine/vlm_engine.py").read_text()


def test_timeout_reads_timeout_seconds_key():
    s = _src()
    # The streaming consumer must read the gateway's 'timeout_seconds' key (with the old
    # 'timeout' as a fallback), not 'timeout' alone.
    # ruff normalises string quotes to double quotes.
    assert 'kwargs.get("timeout_seconds") or kwargs.get("timeout") or 300' in s
    # the old wrong-key-only read is gone
    assert '_timeout_seconds = kwargs.get("timeout", 300)' not in s


def test_timeout_branch_sets_cancel_event():
    s = _src()
    # In the VLM stream timeout branch, cancel_event must be set so the GPU loop stops.
    # ruff may split logger.warning( across lines; anchor on the warning message text.
    to = s.index("VLM stream timeout: no token")
    region = s[to : to + 1400]
    assert "cancel_event.set()" in region
    assert "if cancel_event is not None" in region


def test_cancel_event_set_works_for_both_event_types():
    # The fix calls cancel_event.set(); both asyncio.Event and threading.Event support it,
    # and the GPU loop's _is_cancelled reads asyncio.Event._value (thread-safe read).
    ae = asyncio.Event()
    te = threading.Event()
    ae.set()
    te.set()
    assert ae._value is True  # what _is_cancelled checks for asyncio.Event
    assert te.is_set() is True
