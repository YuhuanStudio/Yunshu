"""The text fast path has no default generation deadline.

BatchedEngine turned the gateway's ``timeout_seconds=None`` into ``or 300.0``, so every
text-only request without a client timeout was cut at 300 s total (a long thinking reply
ended with finish_reason="timeout"; the stream's inactivity wait also counted prefill).
The VLM path dropped its fixed default for the same reason (test_vlm_stream_timeout).
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2] / "python/yunshu_engine"


def _src(name):
    return (ROOT / name).read_text()


def test_batched_engine_passes_the_client_timeout_through():
    assert "timeout_seconds or 300.0" not in _src("batched_engine.py")


def test_fast_paths_default_to_no_deadline():
    for name in (
        "engine_fast.py",
        "engine_stream.py",
        "engine_ngram.py",
        "engine_speculative.py",
    ):
        s = _src(name)
        assert "timeout_seconds: float = 300.0" not in s, name
        assert "timeout_seconds: float | None = None" in s, name


def test_total_deadline_is_skipped_without_a_timeout():
    for name in ("engine_fast.py", "engine_ngram.py"):
        s = _src(name)
        assert "gen_t0 + timeout_seconds if timeout_seconds else None" in s, name
        assert "_timeout_deadline is not None" in s, name
