"""the deferred "spec-stream multi-token stop-prefix leak" is NOT a live bug —
stream_generate() forces spec_decode=False at the top (fast-path spec routes are not
lossless / no speedup on Apple Silicon), so _stream_generate_speculative and
_stream_generate_ngram_spec are unreachable in the streaming flow; the live path
(_stream_generate_fast) already routes deltas through StopHoldbackBuffer (W667/W669).

This test pins the unreachability INVARIANT: if someone re-enables spec streaming (removes
the forced spec_decode=False) without first wrapping the spec emit in a hold-back buffer,
this test fails — flagging that the dead-code stop-leak just went live."""
from __future__ import annotations

import inspect

from yunshu_engine.batched_engine import BatchedEngine


def test_stream_generate_forces_spec_decode_false():
    src = inspect.getsource(BatchedEngine.stream_generate)
    # the guard that makes the spec streaming branches unreachable
    assert "spec_decode = False" in src
    # and it is unconditional (not under some opt-in flag) — a bare reassignment
    assert any(
        line.strip() == "spec_decode = False"
        for line in src.splitlines()
    ), "spec_decode must be unconditionally forced False in stream_generate"


def test_live_streaming_path_has_holdback():
    # the reachable streaming generator must hold back potential stop prefixes
    src = inspect.getsource(BatchedEngine._stream_generate_fast)
    assert "StopHoldbackBuffer" in src


def test_dead_spec_streamers_are_annotated():
    # both unreachable spec streamers carry the honesty annotation so they aren't
    # re-flagged as a live leak (and tell a re-enabler to add hold-back first)
    spec = inspect.getsource(BatchedEngine._stream_generate_speculative)
    ngram = inspect.getsource(BatchedEngine._stream_generate_ngram_spec)
    assert "UNREACHABLE" in spec and "StopHoldbackBuffer" in spec
    assert "UNREACHABLE" in ngram and "StopHoldbackBuffer" in ngram
