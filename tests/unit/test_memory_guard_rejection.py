"""A request the memory guard refuses is an error to the client, never an empty completion."""

from __future__ import annotations

import pytest

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.exceptions import MemoryGuardRejectedError


class _Guard:
    def __init__(self, ok):
        self.ok = ok

    def preflight_check(self, num_prompt_tokens, max_tokens):
        return (self.ok, "" if self.ok else "estimated=10, usable=1")


def _engine(ok):
    eng = BatchedEngine()
    eng._ensure_memory_guard = lambda: _Guard(ok)  # type: ignore[method-assign]
    return eng


def test_non_streaming_rejection_raises_a_memory_error():
    with pytest.raises(MemoryError, match="Memory guard rejected"):
        _engine(False)._check_memory_guard("hello", 100, raise_on_reject=True)
    assert issubclass(MemoryGuardRejectedError, MemoryError)


def test_streaming_rejection_still_carries_the_error_message():
    out = _engine(False)._check_memory_guard("hello", 100)
    assert out is not None
    assert out.finish_reason == "memory_limit" and out.completion_tokens == 0
    assert "Memory guard rejected" in out.error


def test_admitted_request_passes_both_ways():
    assert _engine(True)._check_memory_guard("hello", 100) is None
    assert _engine(True)._check_memory_guard("hello", 100, raise_on_reject=True) is None
