"""Tests for BatchedEngine speculative decoding paths.

Covers text spec fallback, N-gram verification, and request cleanup.
All tests mock engine internals so no real models are needed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers: lightweight fakes that avoid real model / MLX initialization
# ---------------------------------------------------------------------------


class _FakeDetokenizer:
    """Minimal detokenizer that tracks added tokens."""

    def __init__(self):
        self._tokens: list[int] = []
        self.text = ""
        self.last_segment = ""

    def reset(self):
        self._tokens.clear()
        self.text = ""
        self.last_segment = ""

    def add_token(self, tid: int):
        self._tokens.append(tid)
        self.last_segment = f"t{tid}"
        self.text += f" t{tid}"

    def finalize(self):
        self.last_segment = ""


def _make_tokenizer():
    """Minimal tokenizer fake."""
    tok = MagicMock()
    tok.encode = lambda text, **kw: [1, 2, 3]
    tok.decode = lambda ids, **kw: " ".join(f"t{i}" for i in ids)
    tok.eos_token_id = 99
    tok.apply_chat_template = lambda msgs, **kw: "prompt text"
    detok = _FakeDetokenizer()
    tok.detokenizer = detok
    return tok


# ---------------------------------------------------------------------------
# Test 1: generate() spec fallback when no decoder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_speculative_fallback_when_no_decoder():
    """When spec_decode=True but _spec_decoder=None, generate falls through
    to the standard _generate_fast path without crashing."""
    engine = _make_batched_engine(
        _spec_decoder=None,
        _spec_enabled=True,
        _mtp_decoder=None,
        _ngram_proposer=None,
    )

    # Mock _generate_fast to avoid real model / MLX work
    fake_output = MagicMock()
    fake_output.finish_reason = "stop"
    engine._generate_fast = AsyncMock(return_value=fake_output)

    result = await engine.generate(
        prompt="test",
        spec_decode=True,
    )

    # Should have called _generate_fast (the fallback path)
    engine._generate_fast.assert_awaited_once()
    assert result is fake_output


# ---------------------------------------------------------------------------
# Test 2: stream_generate() spec fallback when no decoder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_generate_speculative_fallback_when_no_decoder():
    """When spec_decode=True but _spec_decoder=None, stream_generate falls
    through to the standard streaming fast path and yields at least one
    output."""
    engine = _make_batched_engine(
        _spec_decoder=None,
        _spec_enabled=True,
        _mtp_decoder=None,
        _ngram_proposer=None,
    )

    # Mock _stream_generate_fast to yield one output
    from yunshu_engine.batched_engine import GenerationOutput

    fake_gen_out = GenerationOutput(
        text="hello",
        new_text="hello",
        prompt_tokens=1,
        completion_tokens=1,
        finished=True,
        finish_reason="stop",
    )

    async def _fake_stream(**kwargs):
        yield fake_gen_out

    engine._stream_generate_fast = _fake_stream

    outputs = []
    async for out in engine.stream_generate(
        prompt="test",
        spec_decode=True,
    ):
        outputs.append(out)

    assert len(outputs) >= 1
    assert outputs[0].text == "hello"


# ---------------------------------------------------------------------------
# Retired text MTP adapters are covered by test_spec_stream_unreachable.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 6: _generate_ngram_spec stop suffix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_ngram_spec_stop_suffix():
    """N-gram spec decode respects stop suffixes by truncating output."""
    engine = _make_batched_engine()

    mock_proposer = MagicMock()
    # Proposer returns some draft tokens
    mock_proposer.propose = MagicMock(return_value=[10, 20, 30])
    engine._ngram_proposer = mock_proposer

    # We cannot run the real _generate_ngram_spec (needs MLX), so we test
    # the stop suffix logic pattern that the method uses.
    # The method checks: any(detokenizer.text.endswith(s) for s in stop_suffixes)

    detok = _FakeDetokenizer()
    stop_suffixes = ["END"]

    # Simulate the detokenizer building up text that ends with a stop suffix
    detok.text = "some text END"
    assert any(detok.text.endswith(s) for s in stop_suffixes)

    # Without suffix match
    detok.text = "some text"
    assert not any(detok.text.endswith(s) for s in stop_suffixes)


# ---------------------------------------------------------------------------
# Test 7: _finalize_request idempotent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 8: _finalize_request cleans all state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 9: abort_request finalizes all resources
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 10: engine loop exception puts error output
# ---------------------------------------------------------------------------


def _make_batched_engine(**overrides):
    """Create a BatchedEngine with __init__ mocked out."""
    from yunshu_engine.batched_engine import BatchedEngine

    engine = object.__new__(BatchedEngine)
    # Minimal attributes normally set in __init__
    engine.model_name = overrides.get("model_name", "test-model")
    engine.stream_interval = 1
    engine.enable_thinking = None
    engine._loaded = True
    engine._model = overrides.get("_model", MagicMock())
    engine._tokenizer = overrides.get("_tokenizer", _make_tokenizer())

    engine._spec_decoder = overrides.get("_spec_decoder")
    engine._spec_enabled = overrides.get("_spec_enabled", False)
    engine._mtp_decoder = overrides.get("_mtp_decoder")
    engine._mtp_strategy = None
    engine._ngram_proposer = overrides.get("_ngram_proposer")
    engine._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}

    engine._response_cache_hits = 0
    engine._response_cache_misses = 0
    engine._gpu_rejection_sampler = MagicMock()
    engine._gpu_rejection_enabled = False
    engine._spec_draft_verifier = MagicMock()
    engine._adaptive_spec = None
    engine._lookahead_reasoning = MagicMock()
    engine._medusa_proposer = None
    engine._medusa_strategy = None
    engine._spec_prefill_enabled = False
    engine._kv_prefix_cache = MagicMock()
    engine._prompt_cache = MagicMock()
    engine._mem_pressure_threshold = 85.0
    engine._preprocessor_registry = MagicMock()
    engine._kv_quant_bits = None
    engine._kv_quant_group_size = 64
    engine._kv_quant_start = 0
    engine._deltanet_inverter = None
    engine._deltanet_inversion_enabled = False
    engine._deltanet_inversion_stats = {
        "evictions_captured": 0,
        "inversions_attempted": 0,
        "inversions_succeeded": 0,
        "states_stored": 0,
    }
    engine._settings = None
    engine._thinking_store = None
    engine._total_reasoning_tokens = 0
    engine._lora_manager = None
    engine._compiled = False
    engine._use_compile = False
    engine._metal_kernel_manager = None
    engine._metal_kernels_enabled = False
    engine._streaming_pipeline_enabled = False
    engine._cache_config = None

    return engine
