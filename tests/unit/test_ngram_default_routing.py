"""N-gram spec routing gate. The path is DEFAULT-ON for greedy
(YUNSHU_NGRAM_DEFAULT=1, the default; 0 opts out) now that the prefill/base loop
drives the KV cache with direct model() forwards — speculation is exact
(full-spec == no-spec output) and byte-identical to the fast path on short/medium
greedy gen across gemma-4-e4b / Qwen2.5-3B / Qwen3.5-2B. These tests exercise the
routing GATE given the flag, independent of the env default: greedy + flag-on →
n-gram; flag-off / no-proposer / temp>0 → fast path."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput


def _engine(greedy_default: bool, has_proposer: bool = True):
    eng = BatchedEngine.__new__(BatchedEngine)
    eng._loaded = True
    eng._model = MagicMock()
    eng._tokenizer = MagicMock()
    eng._model_name = "test"
    eng._kv_prefix_cache = MagicMock()
    eng._kv_prefix_cache.get.return_value = (None, None, 0)
    eng._kv_prefix_cache.add = MagicMock()
    eng._mem_pressure_threshold = 0
    eng._spec_enabled = False
    eng._spec_decoder = None
    eng._ngram_proposer = MagicMock() if has_proposer else None
    eng._ngram_greedy_default = greedy_default
    eng._spec_prefill_enabled = False
    eng._kv_quant_bits = None
    eng._check_memory_guard = MagicMock(return_value=None)
    eng._engine_core = None
    return eng


async def _route(eng, **kw):
    """Run generate() with both paths mocked; return which one was hit."""
    eng._generate_fast = AsyncMock(
        return_value=GenerationOutput(text="f", finished=True)
    )
    eng._generate_ngram_spec = AsyncMock(
        return_value=GenerationOutput(text="n", finished=True)
    )
    await eng.generate(prompt="hi", max_tokens=8, **kw)
    return (
        "ngram"
        if eng._generate_ngram_spec.called
        else "fast"
        if eng._generate_fast.called
        else "none"
    )


@pytest.mark.asyncio
async def test_greedy_default_on_routes_to_ngram_without_spec_decode():
    eng = _engine(greedy_default=True)
    assert await _route(eng, temperature=0.0, spec_decode=False) == "ngram"


@pytest.mark.asyncio
async def test_greedy_default_off_stays_on_fast_path():
    eng = _engine(greedy_default=False)
    assert await _route(eng, temperature=0.0, spec_decode=False) == "fast"


@pytest.mark.asyncio
async def test_no_proposer_stays_on_fast_path_even_if_default_on():
    # __new__ engines without _init_spec_decode must not crash on the gate
    eng = _engine(greedy_default=True, has_proposer=False)
    assert await _route(eng, temperature=0.0, spec_decode=False) == "fast"


@pytest.mark.asyncio
async def test_nonzero_temperature_never_takes_ngram():
    # n-gram verify is only lossless at temp<=0
    eng = _engine(greedy_default=True)
    assert await _route(eng, temperature=0.7, spec_decode=False) == "fast"
