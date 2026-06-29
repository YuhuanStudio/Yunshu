"""the streaming fast path must enforce a TOTAL-generation deadline, not just
the consumer's per-token inactivity timeout. timeout means 'max total wall time' for the
non-streaming path; without a total deadline a steady stream ran to max_tokens, blowing
past the user's timeout. Mirror the non-streaming gen_t0 + timeout_seconds deadline in
the streaming GPU loop."""

from __future__ import annotations

import pathlib


def _src():
    root = pathlib.Path(__file__).resolve().parents[2]
    return (root / "python/yunshu_engine/batched_engine.py").read_text()


def test_streaming_total_deadline_computed():
    s = _src()
    assert "_stream_timeout_deadline = _stream_gen_t0 + timeout_seconds" in s


def test_streaming_gpu_loop_checks_total_deadline():
    s = _src()
    # The GPU-loop timeout branch must fire on EITHER the consumer inactivity cancel OR
    # the total deadline.
    i = s.index("_stream_timeout_deadline = _stream_gen_t0")
    # find the loop check after the deadline is set
    check = s.index("_timeout_cancel.is_set() or (", i)
    region = s[check : check + 200]
    assert "_stream_timeout_deadline is not None" in region
    assert "time.perf_counter() > _stream_timeout_deadline" in region


def test_deadline_disabled_when_no_timeout():
    # `timeout_seconds if timeout_seconds else None` → a falsy timeout yields no deadline
    # (the check is guarded by `is not None`), so timeout_seconds=0 doesn't instantly fire.
    s = _src()
    assert "_stream_gen_t0 + timeout_seconds if timeout_seconds else None" in s
