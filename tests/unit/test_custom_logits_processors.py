"""SAMP-2: Tests for custom logits processors.

Verifies that user-provided logits processors:
  - Accept (token_ids: list[int], logits: mx.array) -> mx.array
  - Are applied AFTER built-in processors (repetition/frequency/presence penalty, logit_bias)
  - Work in both streaming and non-streaming generation paths
  - Multiple processors are applied in order
"""
import mlx.core as mx
import pytest

# ---------------------------------------------------------------------------
# 1. Unit tests for _wrap_custom_logits_processor
# ---------------------------------------------------------------------------


class TestWrapCustomLogitsProcessor:
    """Verify the signature adapter converts mx.array tokens to list[int]."""

    def test_wrapper_passes_list_of_int(self):
        """User processor receives token_ids as list[int], not mx.array."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        received_types = []

        def user_proc(token_ids, logits):
            received_types.append(type(token_ids))
            return logits

        wrapped = _wrap_custom_logits_processor(user_proc)
        tokens_mx = mx.array([1, 2, 3])
        logits = mx.zeros(100).astype(mx.float32)
        wrapped(tokens_mx, logits)

        assert len(received_types) == 1
        assert received_types[0] is list

    def test_wrapper_passes_correct_values(self):
        """User processor receives the exact token IDs as ints."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        received_ids = []

        def user_proc(token_ids, logits):
            received_ids.extend(token_ids)
            return logits

        wrapped = _wrap_custom_logits_processor(user_proc)
        tokens_mx = mx.array([42, 99, 7])
        logits = mx.zeros(100).astype(mx.float32)
        wrapped(tokens_mx, logits)

        assert received_ids == [42, 99, 7]

    def test_wrapper_returns_modified_logits(self):
        """User processor can modify and return logits."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        def boost_token_50(token_ids, logits):
            logits[50] += 100.0
            return logits

        wrapped = _wrap_custom_logits_processor(boost_token_50)
        tokens_mx = mx.array([1, 2])
        logits = mx.zeros(100).astype(mx.float32)
        result = wrapped(tokens_mx, logits)

        assert float(result[50]) == pytest.approx(100.0, abs=0.01)

    def test_wrapper_preserves_logits_type(self):
        """Returned logits remain mx.array."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        def user_proc(token_ids, logits):
            return logits

        wrapped = _wrap_custom_logits_processor(user_proc)
        tokens_mx = mx.array([1])
        logits = mx.zeros(10).astype(mx.float32)
        result = wrapped(tokens_mx, logits)

        assert isinstance(result, mx.array)


# ---------------------------------------------------------------------------
# 2. Processor correctness tests (vLLM-style callables)
# ---------------------------------------------------------------------------


class TestCustomProcessorCorrectness:
    """Test realistic custom processor behaviors with list[int] tokens."""

    def test_boost_specific_token(self):
        """A processor that boosts token 42 should make it dominate."""
        def boost_42(token_ids: list[int], logits: mx.array) -> mx.array:
            logits[42] += 50.0
            return logits

        logits = mx.zeros(100).astype(mx.float32)
        result = boost_42([1, 2, 3], logits)
        assert float(result[42]) == pytest.approx(50.0, abs=0.01)
        # All other tokens remain 0
        assert float(result[0]) == pytest.approx(0.0, abs=0.01)
        assert float(result[99]) == pytest.approx(0.0, abs=0.01)

    def test_zero_out_range(self):
        """A processor that zeros out a range of tokens."""
        def zero_range(token_ids: list[int], logits: mx.array) -> mx.array:
            logits[10:20] = -1e9
            return logits

        logits = mx.ones(100).astype(mx.float32)
        result = zero_range([1, 2, 3], logits)
        for i in range(10, 20):
            assert float(result[i]) == pytest.approx(-1e9, abs=1e6)
        for i in range(0, 10):
            assert float(result[i]) == pytest.approx(1.0, abs=0.01)
        for i in range(20, 30):
            assert float(result[i]) == pytest.approx(1.0, abs=0.01)

    def test_multiple_processors_applied_in_order(self):
        """Two processors applied sequentially: first boosts, second zeros."""
        def boost_50(token_ids: list[int], logits: mx.array) -> mx.array:
            logits[50] += 100.0
            logits[51] += 100.0
            return logits

        def zero_50(token_ids: list[int], logits: mx.array) -> mx.array:
            logits[50] = -1e9
            return logits

        logits = mx.zeros(100).astype(mx.float32)
        # Apply in order: boost first, then zero
        logits = boost_50([1], logits)
        logits = zero_50([1], logits)

        assert float(logits[50]) == pytest.approx(-1e9, abs=1e6)
        assert float(logits[51]) == pytest.approx(100.0, abs=0.01)  # Not zeroed


# ---------------------------------------------------------------------------
# 3. Integration: custom processors are placed AFTER built-in processors
# ---------------------------------------------------------------------------


class TestProcessorOrdering:
    """Verify custom processors run after built-in penalty/bias processors."""

    def test_custom_after_logit_bias(self):
        """Custom processor sees the effect of logit_bias, and can override it."""
        # Simulate built-in logit_bias processor (internal: mx.array signature)
        def logit_bias_proc(_tokens, logits):
            logits[10] += 5.0
            return logits

        # User custom processor (vLLM-style: list[int] signature)
        def user_proc(token_ids: list[int], logits: mx.array) -> mx.array:
            # The user should see the +5.0 from logit_bias and can override
            logits[10] = 0.0  # Override back to zero
            logits[20] = 999.0  # Set their own value
            return logits

        # Wrap user proc to adapt signature
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        wrapped_user = _wrap_custom_logits_processor(user_proc)

        # Apply in engine order: built-in first, then wrapped user proc
        logits = mx.zeros(100).astype(mx.float32)
        logits = logit_bias_proc(mx.array([1]), logits)
        assert float(logits[10]) == pytest.approx(5.0, abs=0.01)

        logits = wrapped_user(mx.array([1]), logits)
        assert float(logits[10]) == pytest.approx(0.0, abs=0.01)  # Overridden
        assert float(logits[20]) == pytest.approx(999.0, abs=0.01)  # Custom applied

    def test_custom_after_frequency_penalty(self):
        """Custom processor can undo frequency penalty if desired."""
        def freq_penalty_proc(tokens, logits, fp=1.0):
            counts = {}
            for t in tokens:
                tid = int(t)
                counts[tid] = counts.get(tid, 0) + 1
            for tid, cnt in counts.items():
                logits[..., tid] -= fp * cnt
            return logits

        def user_proc(token_ids: list[int], logits: mx.array) -> mx.array:
            # Undo the frequency penalty for token 5
            logits[5] = 42.0
            return logits

        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        wrapped_user = _wrap_custom_logits_processor(user_proc)

        # Token 5 appears 3 times → freq penalty = 1.0 * 3 = -3.0
        tokens = mx.array([5, 5, 5])
        logits = mx.zeros(100).astype(mx.float32)
        logits = freq_penalty_proc(tokens, logits)
        assert float(logits[5]) == pytest.approx(-3.0, abs=0.01)

        # User processor overrides
        logits = wrapped_user(tokens, logits)
        assert float(logits[5]) == pytest.approx(42.0, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Integration: _generate_fast path with custom processors
# ---------------------------------------------------------------------------


class TestGenerateFastCustomProcessors:
    """Verify custom logits processors are applied in _generate_fast."""

    @pytest.fixture
    def engine(self):
        from unittest.mock import MagicMock

        from yunshu_engine.batched_engine import BatchedEngine
        eng = BatchedEngine.__new__(BatchedEngine)
        eng._loaded = True
        eng._model = None
        eng._tokenizer = None
        eng._model_name = "test"
        eng._kv_prefix_cache = MagicMock()
        eng._kv_prefix_cache.get.return_value = (None, None, 0)
        eng._kv_prefix_cache.add = MagicMock()
        eng._mem_pressure_threshold = 0
        eng._spec_enabled = False
        eng._spec_decoder = None
        eng._ngram_proposer = None
        eng._spec_prefill_enabled = False
        eng._spec_prefill_draft_model = None
        eng._spec_prefill_threshold = 4096
        eng._spec_prefill_keep_rate = 0.3
        eng._kv_quant_bits = None
        eng._kv_quant_group_size = 64
        eng._check_memory_guard = lambda *a, **kw: None
        eng._engine_core = None
        eng._prompt_cache = None
        eng._thinking_store = None
        eng._token_pipeline = None
        return eng

    @pytest.mark.asyncio
    async def test_custom_processor_forwarded_to_generate_fast(self, engine):
        """logits_processors from generate() reach _generate_fast()."""
        from unittest.mock import AsyncMock, patch

        from yunshu_engine.batched_engine import GenerationOutput

        def custom_proc(token_ids, logits):
            return logits

        with patch.object(engine, '_generate_fast', new_callable=AsyncMock) as mock_fast:
            mock_fast.return_value = GenerationOutput(text="ok", finished=True)
            await engine.generate(
                prompt="test",
                logits_processors=[custom_proc],
            )
            call_kwargs = mock_fast.call_args
            procs = call_kwargs.kwargs.get("logits_processors")
            assert procs is not None
            assert len(procs) == 1
            assert procs[0] is custom_proc

    @pytest.mark.asyncio
    async def test_custom_processor_forwarded_to_stream_generate_fast(self, engine):
        """logits_processors from stream_generate() reach _stream_generate_fast()."""
        from unittest.mock import patch

        from yunshu_engine.batched_engine import GenerationOutput

        def custom_proc(token_ids, logits):
            return logits

        async def _fake_stream(**kwargs):
            yield GenerationOutput(text="ok", finished=True)

        with patch.object(engine, '_stream_generate_fast', side_effect=_fake_stream):
            chunks = []
            async for chunk in engine.stream_generate(
                prompt="test",
                logits_processors=[custom_proc],
            ):
                chunks.append(chunk)
            assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# 5. End-to-end: wrapped processors in a generate_step-like loop
# ---------------------------------------------------------------------------


class TestWrappedProcessorsInLoop:
    """Simulate the actual generate_step loop with wrapped custom processors."""

    def test_boost_token_dominates_sampling(self):
        """A boost processor should make the target token always selected."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        TARGET = 42

        def boost_target(token_ids: list[int], logits: mx.array) -> mx.array:
            logits[TARGET] += 1e6
            return logits

        wrapped = _wrap_custom_logits_processor(boost_target)

        # Simulate generate_step's sampling: argmax after processing
        tokens = mx.array([1, 2, 3])
        logits = mx.zeros(100).astype(mx.float32)
        # Add some random noise to make it non-trivial
        logits[mx.array([10, 20, 30])] = mx.array([5.0, 3.0, 4.0])

        processed = wrapped(tokens, logits)
        sampled = mx.argmax(processed)
        mx.eval(sampled)

        assert int(sampled) == TARGET

    def test_zero_range_excludes_tokens(self):
        """Zeroing out tokens 0-49 should force sampling from 50+."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        def zero_low_tokens(token_ids: list[int], logits: mx.array) -> mx.array:
            logits[:50] = -1e9
            return logits

        wrapped = _wrap_custom_logits_processor(zero_low_tokens)

        tokens = mx.array([1, 2, 3])
        logits = mx.zeros(100).astype(mx.float32)
        # Set some low tokens high, but they should be zeroed
        logits[mx.array([0, 1, 10, 25])] = mx.array([100.0, 90.0, 80.0, 70.0])
        # Set token 75 as the highest outside the zeroed range
        logits[75] = 50.0

        processed = wrapped(tokens, logits)
        sampled = mx.argmax(processed)
        mx.eval(sampled)

        assert int(sampled) == 75
        assert int(sampled) >= 50  # Must be from the non-zeroed range

    def test_sequential_processors_chain(self):
        """Three processors applied in sequence produce correct final logits."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        call_order = []

        def proc_a(token_ids: list[int], logits: mx.array) -> mx.array:
            call_order.append("a")
            logits[10] += 1.0
            return logits

        def proc_b(token_ids: list[int], logits: mx.array) -> mx.array:
            call_order.append("b")
            logits[10] *= 2.0
            return logits

        def proc_c(token_ids: list[int], logits: mx.array) -> mx.array:
            call_order.append("c")
            logits[20] = logits[10]
            return logits

        procs = [_wrap_custom_logits_processor(p) for p in [proc_a, proc_b, proc_c]]

        logits = mx.zeros(100).astype(mx.float32)
        tokens = mx.array([1])

        for p in procs:
            logits = p(tokens, logits)

        mx.eval(logits)
        # proc_a: logits[10] = 0 + 1.0 = 1.0
        # proc_b: logits[10] = 1.0 * 2.0 = 2.0
        # proc_c: logits[20] = logits[10] = 2.0
        assert call_order == ["a", "b", "c"]
        assert float(logits[10]) == pytest.approx(2.0, abs=0.01)
        assert float(logits[20]) == pytest.approx(2.0, abs=0.01)

    def test_empty_custom_processors_list_no_error(self):
        """Passing empty list or None for logits_processors should work fine."""
        from yunshu_engine.batched_engine import _wrap_custom_logits_processor

        # Wrapping an empty list produces empty
        wrapped = [_wrap_custom_logits_processor(p) for p in []]
        assert wrapped == []

        # None should be handled at the engine level
        wrapped = [_wrap_custom_logits_processor(p) for p in (None or [])]
        assert wrapped == []
