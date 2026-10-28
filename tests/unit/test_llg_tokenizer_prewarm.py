"""The llguidance token tables (about a second of vocabulary walking) are built in the
background at model load, once per mask width, without blocking the caller."""

import threading
import time
from unittest.mock import patch

import pytest

from yunshu_engine import settings
from yunshu_engine import tool_call_grammar as tcg

pytest.importorskip("llguidance")


class _Hf:
    """Stands in for a Hugging Face tokenizer (has ``backend_tokenizer``)."""

    backend_tokenizer = object()

    def __len__(self):
        return 1000


class _Engine:
    def __init__(self, tok, config=None):
        self._tokenizer = tok
        self._config = config or {}
        self._model = None


@pytest.fixture
def slow_build():
    calls: list[int] = []

    def fake_from_tokenizer(hf, n_vocab):
        calls.append(n_vocab)
        time.sleep(0.3)
        return ("llt", n_vocab)

    with patch("llguidance.hf.from_tokenizer", fake_from_tokenizer):
        yield calls
    for key in [k for k in tcg._TOKENIZERS if k[1] in (1000, 1024)]:
        tcg._TOKENIZERS.pop(key, None)
        tcg._TOKENIZER_BYTES.pop(key, None)
        tcg._BUILD_LOCKS.pop(key, None)


def _wait_cached(tok, width, timeout=5.0):
    key = (id(tcg._hf_tokenizer(tok)), width)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if key in tcg._TOKENIZERS:
            return True
        time.sleep(0.01)
    return False


def test_prewarm_returns_at_once_and_builds_only_the_tokenizer_width(slow_build):
    tok = _Hf()
    t0 = time.perf_counter()
    assert tcg.prewarm_for_engine(_Engine(tok, {"vocab_size": 1024})) is True
    assert time.perf_counter() - t0 < 0.1  # nothing waits for the build
    assert _wait_cached(tok, 1000)
    time.sleep(0.5)  # a second (padded-width) table would have been built by now
    assert slow_build == [1000]  # was [1000, 1024]: two ~0.45 GiB tables at every load
    # a later request finds the cached table: no further build
    assert tcg.llg_tokenizer(tok, 1000) == ("llt", 1000)
    assert slow_build == [1000]
    # the forced-tool width is built on first use, once
    assert tcg.llg_tokenizer(tok, 1024) == ("llt", 1024)
    assert tcg.llg_tokenizer(tok, 1024) == ("llt", 1024)
    assert sorted(slow_build) == [1000, 1024]


def test_a_request_during_the_prewarm_waits_for_it_instead_of_building_again(
    slow_build,
):
    tok = _Hf()
    assert tcg.prewarm_llg_tokenizer(tok) is True
    while not slow_build:  # prewarm is inside from_tokenizer now
        time.sleep(0.005)
    got = []
    t = threading.Thread(target=lambda: got.append(tcg.llg_tokenizer(tok, 1000)))
    t.start()
    t.join(5)
    assert got == [("llt", 1000)]
    assert slow_build == [1000]


def test_skipped_without_a_hf_tokenizer_llguidance_or_when_disabled(
    slow_build, monkeypatch
):
    class NoHf:
        def __len__(self):
            return 1000

    assert tcg.prewarm_llg_tokenizer(NoHf()) is False
    assert tcg.prewarm_llg_tokenizer(None) is False
    with patch("importlib.util.find_spec", return_value=None):
        assert tcg.prewarm_llg_tokenizer(_Hf()) is False
    monkeypatch.setenv("YUNSHU_GRAMMAR_PREWARM", "0")
    assert settings.get("YUNSHU_GRAMMAR_PREWARM") is False
    assert tcg.prewarm_llg_tokenizer(_Hf()) is False
    assert slow_build == []


def test_a_failing_build_is_swallowed(slow_build):
    with patch.object(tcg, "llg_tokenizer", side_effect=RuntimeError("boom")):
        tok = _Hf()
        assert tcg.prewarm_llg_tokenizer(tok) is True
        time.sleep(0.2)  # the worker logs and moves on; later requests build normally
    assert tcg.llg_tokenizer(tok, 1000) == ("llt", 1000)
