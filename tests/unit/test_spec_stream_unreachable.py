"""Speculative routes in BatchedEngine come from one table (``_spec_route``).

Streaming speculation is limited to the experimental MTP route, whose streamer
holds back stop prefixes; the EAGLE and n-gram streamers have no hold-back
(a multi-token stop would leak), so the table never routes a stream to them.
"""

from __future__ import annotations

import inspect

import pytest

from yunshu_engine.batched_engine import BatchedEngine


def _engine(**attrs):
    eng = object.__new__(BatchedEngine)
    base = dict(
        _spec_enabled=True,
        _spec_decoder=object(),
        _mtp_decoder=object(),
        _ngram_proposer=object(),
        _ngram_greedy_default=False,
    )
    base.update(attrs)
    for k, v in base.items():
        setattr(eng, k, v)
    return eng


def _route(eng, **kw):
    args = dict(
        spec_decode=True,
        stream=False,
        temperature=0.0,
        logprobs=False,
        use_engine_loop=False,
    )
    args.update(kw)
    return eng._spec_route(**args)


def test_default_routes(monkeypatch):
    monkeypatch.delenv("YUNSHU_SPEC_UNVERIFIED", raising=False)
    eng = _engine()
    assert _route(eng) == "ngram"  # per-request spec_decode reaches n-gram
    assert _route(eng, spec_decode=False) is None
    assert _route(_engine(_ngram_greedy_default=True), spec_decode=False) == "ngram"
    assert _route(eng, temperature=0.7) is None  # n-gram is greedy-only
    assert _route(eng, logprobs=True) is None
    assert _route(eng, use_engine_loop=True) is None
    assert _route(eng, gemma4_eligible=lambda: True) == "gemma4_assistant"


def test_streams_never_take_unsafe_routes(monkeypatch):
    eng = _engine()
    for flag in ("", "eagle"):
        monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", flag)
        assert _route(eng, stream=True, gemma4_eligible=lambda: True) is None
    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", "mtp")
    assert _route(eng, stream=True) == "mtp"


@pytest.mark.parametrize("flag,expected", [("eagle", "eagle"), ("mtp", "mtp")])
def test_unverified_routes_are_opt_in(monkeypatch, flag, expected):
    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", flag)
    assert _route(_engine()) == expected
    assert _route(_engine(), temperature=0.5) is None


def test_live_streaming_path_has_holdback():
    src = inspect.getsource(BatchedEngine._stream_generate_fast)
    assert "StopHoldbackBuffer" in src
    assert "StopHoldbackBuffer(" in inspect.getsource(
        BatchedEngine._stream_generate_mtp
    )


def test_unrouted_streamers_stay_annotated():
    spec = inspect.getsource(BatchedEngine._stream_generate_speculative)
    ngram = inspect.getsource(BatchedEngine._stream_generate_ngram_spec)
    assert "UNREACHABLE" in spec and "UNREACHABLE" in ngram
