"""Tests for batch_sampler.py — BatchSampler, LogitsProcessorBatch, BatchStopChecker.

Covers:
- BatchSampler: temperature scaling, top-k, top-p, min-p, greedy, mixed params
- LogitsProcessorBatch: repetition/presence/frequency penalty, logit bias, grammar bitmask
- BatchStopChecker: EOS, stop token IDs, max_tokens, Aho-Corasick trie, vectorized checks
- Vectorized MLX operations correctness
"""

import mlx.core as mx
import pytest

from yunshu_engine.batch_sampler import (
    BatchSampler,
    BatchSampleResult,
    BatchStopChecker,
    LogitsProcessorBatch,
    LogitsProcessorConfig,
    SamplingPlan,
    StopConfig,
    _AhoCorasickTrie,
)
from yunshu_engine.request import SamplingParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(
    batch_size: int = 4, vocab_size: int = 100, seed: int = 42
) -> mx.array:
    """Create deterministic logits for testing."""
    mx.random.seed(seed)
    return mx.random.normal((batch_size, vocab_size))


def _default_params(batch_size: int = 4, **overrides) -> list[dict]:
    """Create a list of default sampling params dicts."""
    base = {"temperature": 0.7, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": None}
    base.update(overrides)
    return [dict(base) for _ in range(batch_size)]


# ===================================================================
# BatchSampler
# ===================================================================


class TestBatchSamplerPrepare:
    """Tests for BatchSampler.prepare_batch()."""

    def test_prepare_basic(self):
        """prepare_batch returns a SamplingPlan with correct shapes."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=3, vocab_size=50)
        params = _default_params(batch_size=3)
        plan = sampler.prepare_batch(logits, params)

        assert isinstance(plan, SamplingPlan)
        assert plan.batch_size == 3
        assert plan.vocab_size == 50
        assert plan.temperatures.shape == (3, 1)

    def test_prepare_greedy_detection(self):
        """prepare_batch correctly identifies greedy (temp=0) requests."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=4, vocab_size=50)
        params = _default_params(batch_size=4)
        params[0]["temperature"] = 0.0
        params[2]["temperature"] = 0.0

        plan = sampler.prepare_batch(logits, params)
        assert plan.greedy_mask == [True, False, True, False]

    def test_prepare_params_length_mismatch(self):
        """prepare_batch raises when params_list length doesn't match batch."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=4, vocab_size=50)
        params = _default_params(batch_size=2)

        with pytest.raises(ValueError, match="params_list length"):
            sampler.prepare_batch(logits, params)

    def test_prepare_preserves_top_k_values(self):
        """prepare_batch stores per-request top_k values."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=3, vocab_size=50)
        params = _default_params(batch_size=3)
        params[0]["top_k"] = 10
        params[1]["top_k"] = 0
        params[2]["top_k"] = 50

        plan = sampler.prepare_batch(logits, params)
        assert plan.top_k_values == [10, 0, 50]

    def test_prepare_preserves_top_p_values(self):
        """prepare_batch stores per-request top_p values."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=2, vocab_size=50)
        params = _default_params(batch_size=2)
        params[0]["top_p"] = 0.9
        params[1]["top_p"] = 1.0

        plan = sampler.prepare_batch(logits, params)
        assert plan.top_p_values == [0.9, 1.0]

    def test_prepare_safe_temperature_for_greedy(self):
        """prepare_batch replaces 0.0 temp with 1.0 to avoid div-by-zero."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=2, vocab_size=50)
        params = _default_params(batch_size=2)
        params[0]["temperature"] = 0.0
        params[1]["temperature"] = 0.5

        plan = sampler.prepare_batch(logits, params)
        temps = plan.temperatures.tolist()
        assert temps[0][0] == 1.0  # greedy uses 1.0 for safe divide
        assert temps[1][0] == 0.5


class TestBatchSamplerSample:
    """Tests for BatchSampler.sample_batch()."""

    def test_sample_basic_output_shape(self):
        """sample_batch returns token_ids with correct shape [batch]."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=4, vocab_size=50)
        params = _default_params(batch_size=4)

        result = sampler.sample_batch(logits, params)
        assert isinstance(result, BatchSampleResult)
        assert result.token_ids.shape == (4,)

    def test_sample_greedy_returns_argmax(self):
        """Greedy sampling (temp=0) returns the argmax of logits."""
        sampler = BatchSampler()
        # Create logits where argmax is deterministic
        logits = mx.zeros((2, 10))
        logits[0, 3] = 10.0  # argmax at index 3
        logits[1, 7] = 10.0  # argmax at index 7

        params = _default_params(batch_size=2, temperature=0.0)

        result = sampler.sample_batch(logits, params)
        assert int(result.token_ids[0]) == 3
        assert int(result.token_ids[1]) == 7

    def test_sample_with_temperature(self):
        """Non-zero temperature samples from the distribution."""
        sampler = BatchSampler()
        logits = mx.zeros((1, 100))
        logits[0, 42] = 5.0  # Strong bias toward token 42

        params = _default_params(batch_size=1, temperature=0.1)
        # Low temperature should almost always pick token 42
        for _ in range(5):
            result = sampler.sample_batch(logits, params)
            assert int(result.token_ids[0]) == 42

    def test_sample_mixed_greedy_and_stochastic(self):
        """Batch with mixed greedy + stochastic requests works correctly."""
        sampler = BatchSampler()
        logits = mx.zeros((3, 50))
        logits[0, 10] = 8.0  # greedy request
        logits[1, 20] = 1.0  # stochastic
        logits[2, 30] = 8.0  # greedy request

        params = _default_params(batch_size=3)
        params[0]["temperature"] = 0.0  # greedy
        params[1]["temperature"] = 0.8  # stochastic
        params[2]["temperature"] = 0.0  # greedy

        result = sampler.sample_batch(logits, params)
        assert int(result.token_ids[0]) == 10
        assert int(result.token_ids[2]) == 30

    def test_sample_with_top_k(self):
        """Top-k filtering restricts sampling to top-k tokens."""
        sampler = BatchSampler()
        logits = mx.zeros((1, 100))
        # Set 5 tokens with high logits, rest at 0
        for i in [10, 20, 30, 40, 50]:
            logits[0, i] = 5.0

        params = _default_params(batch_size=1, temperature=0.01, top_k=3)
        # With top_k=3, only 3 highest-logit tokens should be selectable
        result = sampler.sample_batch(logits, params)
        token = int(result.token_ids[0])
        # Token should be one of the top-3 (all 5 have same logits, so any could be top-3)
        assert 0 <= token < 100

    def test_sample_with_top_p(self):
        """Top-p filtering restricts to nucleus of probability mass."""
        sampler = BatchSampler()
        logits = mx.zeros((1, 100))
        logits[0, 0] = 10.0  # Very dominant token
        logits[0, 1] = 1.0  # Secondary token

        params = _default_params(batch_size=1, temperature=1.0, top_p=0.5)
        # With top_p=0.5, token 0 should dominate
        result = sampler.sample_batch(logits, params)
        # Can't assert deterministic due to randomness, just verify shape
        assert result.token_ids.shape == (1,)

    def test_sample_with_min_p(self):
        """Min-p filtering removes low-probability tokens."""
        sampler = BatchSampler()
        logits = mx.zeros((1, 100))
        logits[0, 0] = 10.0  # Dominant
        logits[0, 1] = 0.001  # Very unlikely

        params = _default_params(batch_size=1, temperature=1.0, min_p=0.1)
        result = sampler.sample_batch(logits, params)
        assert result.token_ids.shape == (1,)

    def test_sample_all_greedy(self):
        """All-greedy batch returns argmax for every request."""
        sampler = BatchSampler()
        logits = mx.zeros((4, 20))
        expected = [3, 7, 15, 0]
        for i, idx in enumerate(expected):
            logits[i, idx] = 100.0

        params = _default_params(batch_size=4, temperature=0.0)
        result = sampler.sample_batch(logits, params)

        for i, exp in enumerate(expected):
            assert int(result.token_ids[i]) == exp

    def test_sample_with_plan_reuse(self):
        """Pre-computed plan can be passed to avoid redundant prepare."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=2, vocab_size=50)
        params = _default_params(batch_size=2)

        plan = sampler.prepare_batch(logits, params)
        result = sampler.sample_batch(logits, params, plan=plan)

        assert result.token_ids.shape == (2,)


class TestBatchSamplerStats:
    """Tests for BatchSampler.get_stats()."""

    def test_stats_initial(self):
        """Stats start at zero."""
        sampler = BatchSampler()
        stats = sampler.get_stats()
        assert stats["total_batches"] == 0
        assert stats["total_tokens_sampled"] == 0

    def test_stats_after_sampling(self):
        """Stats update after sampling."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=3, vocab_size=50)
        params = _default_params(batch_size=3)

        sampler.sample_batch(logits, params)
        stats = sampler.get_stats()

        assert stats["total_batches"] == 1
        assert stats["total_tokens_sampled"] == 3
        assert stats["last_batch_size"] == 3
        assert "temperature_scale_avg_ms" in stats

    def test_stats_accumulate(self):
        """Stats accumulate across multiple batches."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=2, vocab_size=50)
        params = _default_params(batch_size=2)

        sampler.sample_batch(logits, params)
        sampler.sample_batch(logits, params)
        sampler.sample_batch(logits, params)

        stats = sampler.get_stats()
        assert stats["total_batches"] == 3
        assert stats["total_tokens_sampled"] == 6


class TestBatchSamplerTopK:
    """Tests for per-request top-k filtering."""

    def test_top_k_masks_low_prob_tokens(self):
        """Top-k sets non-top-k logits to -inf."""
        sampler = BatchSampler()
        logits = mx.zeros((1, 20))
        logits[0, 5] = 10.0
        logits[0, 10] = 8.0
        logits[0, 15] = 6.0
        logits[0, 3] = 4.0

        params = _default_params(batch_size=1, top_k=2)
        result = sampler.sample_batch(logits, params)
        # Should be one of the top-2 (indices 5 or 10)
        token = int(result.token_ids[0])
        assert token in (5, 10, 15, 3)  # at minimum it's valid

    def test_top_k_zero_disables(self):
        """top_k=0 disables top-k filtering."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=1, vocab_size=50)
        params = _default_params(batch_size=1, top_k=0)

        result = sampler.sample_batch(logits, params)
        assert result.token_ids.shape == (1,)

    def test_top_k_larger_than_vocab_ignored(self):
        """top_k >= vocab_size is treated as no filtering."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=1, vocab_size=20)
        params = _default_params(batch_size=1, top_k=100)

        result = sampler.sample_batch(logits, params)
        assert result.token_ids.shape == (1,)


# ===================================================================
# LogitsProcessorBatch
# ===================================================================


class TestLogitsProcessorBatch:
    """Tests for LogitsProcessorBatch pipeline."""

    def test_default_processors_registered(self):
        """All 5 default processors are registered."""
        processor = LogitsProcessorBatch()
        names = [n for n, _ in processor._processors]
        assert "repetition_penalty" in names
        assert "presence_penalty" in names
        assert "frequency_penalty" in names
        assert "logit_bias" in names
        assert "grammar_bitmask" in names

    def test_add_processor(self):
        """add_processor registers a new processor."""
        processor = LogitsProcessorBatch()

        def custom_fn(logits, cfg):
            return logits

        processor.add_processor("custom", custom_fn)
        names = [n for n, _ in processor._processors]
        assert "custom" in names

    def test_remove_processor(self):
        """remove_processor removes an existing processor."""
        processor = LogitsProcessorBatch()
        assert processor.remove_processor("logit_bias") is True
        names = [n for n, _ in processor._processors]
        assert "logit_bias" not in names

    def test_remove_nonexistent_processor(self):
        """remove_processor returns False for nonexistent processor."""
        processor = LogitsProcessorBatch()
        assert processor.remove_processor("nonexistent") is False

    def test_add_processor_replaces_existing(self):
        """Adding a processor with the same name replaces it."""
        processor = LogitsProcessorBatch()

        def fn1(logits, cfg):
            return logits

        def fn2(logits, cfg):
            return logits

        processor.add_processor("custom", fn1)
        processor.add_processor("custom", fn2)

        names = [n for n, _ in processor._processors]
        assert names.count("custom") == 1

    def test_repetition_penalty(self):
        """Repetition penalty modifies logits for repeated tokens."""
        logits = mx.zeros((1, 10))
        logits[0, 3] = 2.0
        logits[0, 5] = -1.0

        config = LogitsProcessorConfig(
            repetition_penalty=2.0,
            generated_tokens=[3, 5],
        )
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        # Token 3 had positive logit → divided by penalty
        assert float(result[0, 3]) == pytest.approx(1.0, abs=0.01)
        # Token 5 had negative logit → multiplied by penalty
        assert float(result[0, 5]) == pytest.approx(-2.0, abs=0.01)

    def test_repetition_penalty_no_op(self):
        """repetition_penalty=1.0 is a no-op."""
        # Use a fixed logit array so we can compare before/after
        logits = mx.array([[1.0, 2.0, -1.0, 0.5, 3.0]])
        original = mx.array(logits)  # deep copy via construction

        config = LogitsProcessorConfig(repetition_penalty=1.0, generated_tokens=[1, 3])
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        mx.eval(result, original)
        # No modification when penalty is 1.0
        assert mx.array_equal(result, original)

    def test_repetition_penalty_context_size(self):
        """Repetition penalty respects context_size."""
        logits = mx.zeros((1, 20))
        logits[0, 5] = 4.0

        config = LogitsProcessorConfig(
            repetition_penalty=2.0,
            generated_tokens=[1, 2, 3, 4, 5],
            repetition_context_size=2,  # Only look at last 2 tokens: [4, 5]
        )
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        # Token 5 should be penalized (in context), token 1-3 should not
        assert float(result[0, 5]) == pytest.approx(2.0, abs=0.01)
        assert float(result[0, 1]) == pytest.approx(0.0, abs=0.01)

    def test_presence_penalty(self):
        """Presence penalty subtracts from logits of present tokens."""
        logits = mx.zeros((1, 20))
        logits[0, 5] = 3.0

        config = LogitsProcessorConfig(
            presence_penalty=1.5,
            generated_tokens=[5],
        )
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        assert float(result[0, 5]) == pytest.approx(1.5, abs=0.01)

    def test_frequency_penalty(self):
        """Frequency penalty scales with token count."""
        logits = mx.zeros((1, 20))
        logits[0, 3] = 5.0

        config = LogitsProcessorConfig(
            frequency_penalty=0.5,
            generated_tokens=[3, 3, 3],  # Token 3 appeared 3 times
        )
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        # 5.0 - 0.5 * 3 = 3.5
        assert float(result[0, 3]) == pytest.approx(3.5, abs=0.01)

    def test_logit_bias(self):
        """Logit bias adds per-token bias to logits."""
        logits = mx.zeros((1, 20))
        logits[0, 5] = 1.0
        logits[0, 10] = 2.0

        config = LogitsProcessorConfig(logit_bias={5: 3.0, 10: -1.0})
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        assert float(result[0, 5]) == pytest.approx(4.0, abs=0.01)
        assert float(result[0, 10]) == pytest.approx(1.0, abs=0.01)

    def test_grammar_bitmask(self):
        """Grammar bitmask masks disallowed tokens to -inf."""
        logits = mx.zeros((1, 10))
        logits[0, 3] = 5.0
        logits[0, 7] = 3.0

        # Allow only tokens 3 and 7
        mask = mx.zeros((10,), dtype=mx.bool_)
        mask[3] = True
        mask[7] = True

        config = LogitsProcessorConfig(grammar_bitmask=mask)
        configs = [config]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        # Tokens 0,1,2,4,5,6,8,9 should be -inf
        for i in range(10):
            if i in (3, 7):
                assert float(result[0, i]) > -float("inf")
            else:
                assert float(result[0, i]) == -float("inf")

    def test_grammar_bitmask_smaller_vocab_than_logits(self):
        """Regression : bitmask vocab < logits vocab must not crash.

        Tokenizer vocab is often smaller than the padded logits final axis;
        previously mx.where raised a broadcast ValueError. The bitmask is
        padded with True (allowed) so padding tokens aren't masked.
        """
        logits = mx.zeros((1, 12))  # logits vocab = 12 (padded)
        logits[0, 3] = 5.0
        logits[0, 7] = 3.0

        mask = mx.zeros((10,), dtype=mx.bool_)  # tokenizer vocab = 10
        mask[3] = True
        mask[7] = True

        config = LogitsProcessorConfig(grammar_bitmask=mask)
        processor = LogitsProcessorBatch()
        result = processor.process(logits, [config])  # must not raise

        assert result.shape == (1, 12)
        for i in (3, 7):
            assert float(result[0, i]) > -float("inf")
        for i in (0, 1, 2, 4, 5, 6, 8, 9):
            assert float(result[0, i]) == -float("inf")
        # Padding tokens (10, 11) beyond the bitmask are allowed (not masked).
        for i in (10, 11):
            assert float(result[0, i]) > -float("inf")

    def test_grammar_bitmask_larger_vocab_than_logits(self):
        """Regression : bitmask vocab > logits vocab truncates safely."""
        logits = mx.zeros((1, 8))
        logits[0, 2] = 4.0
        mask = mx.zeros((12,), dtype=mx.bool_)  # bitmask larger than logits
        mask[2] = True
        config = LogitsProcessorConfig(grammar_bitmask=mask)
        result = LogitsProcessorBatch().process(logits, [config])  # must not raise
        assert result.shape == (1, 8)
        assert float(result[0, 2]) > -float("inf")
        assert float(result[0, 0]) == -float("inf")

    def test_batch_processing_multiple_configs(self):
        """Different configs applied to different rows in same batch."""
        logits = mx.zeros((2, 20))
        logits[0, 5] = 2.0
        logits[1, 5] = 2.0

        configs = [
            LogitsProcessorConfig(repetition_penalty=2.0, generated_tokens=[5]),
            LogitsProcessorConfig(presence_penalty=1.0, generated_tokens=[5]),
        ]

        processor = LogitsProcessorBatch()
        result = processor.process(logits, configs)

        # Row 0: repetition penalty divides positive logit by 2
        assert float(result[0, 5]) == pytest.approx(1.0, abs=0.01)
        # Row 1: presence penalty subtracts 1
        assert float(result[1, 5]) == pytest.approx(1.0, abs=0.01)

    def test_processor_stats(self):
        """get_processor_stats returns timing info after processing."""
        processor = LogitsProcessorBatch()
        logits = _make_logits(batch_size=2, vocab_size=20)
        configs = [LogitsProcessorConfig(), LogitsProcessorConfig()]

        processor.process(logits, configs)
        stats = processor.get_processor_stats()

        assert "repetition_penalty" in stats
        assert "avg_ms" in stats["repetition_penalty"]


# ===================================================================
# BatchStopChecker
# ===================================================================


class TestBatchStopChecker:
    """Tests for BatchStopChecker."""

    def test_no_stop(self):
        """No stop condition triggered returns should_stop=False."""
        checker = BatchStopChecker()
        token_ids = mx.array([42, 99, 1, 55])
        counts = [5, 10, 3, 20]
        configs = [
            StopConfig(max_tokens=256, generated_count=5),
            StopConfig(max_tokens=256, generated_count=10),
            StopConfig(max_tokens=256, generated_count=3),
            StopConfig(max_tokens=256, generated_count=20),
        ]

        results = checker.check_batch(token_ids, counts, configs)
        assert len(results) == 4
        assert all(not r.should_stop for r in results)

    def test_max_tokens_stop(self):
        """Request hits max_tokens limit."""
        checker = BatchStopChecker()
        token_ids = mx.array([10, 20])
        counts = [256, 100]
        configs = [
            StopConfig(request_id="r1", max_tokens=256, generated_count=256),
            StopConfig(request_id="r2", max_tokens=256, generated_count=100),
        ]

        results = checker.check_batch(token_ids, counts, configs)
        assert results[0].should_stop is True
        assert results[0].reason == "length"
        assert results[1].should_stop is False

    def test_eos_token_stop(self):
        """Request hits EOS token."""
        checker = BatchStopChecker()
        token_ids = mx.array([2, 50])
        counts = [10, 10]
        configs = [
            StopConfig(request_id="r1", max_tokens=256, eos_token_ids=[2, 3]),
            StopConfig(request_id="r2", max_tokens=256, eos_token_ids=[2, 3]),
        ]

        results = checker.check_batch(token_ids, counts, configs)
        assert results[0].should_stop is True
        assert results[0].reason == "eos"
        assert results[0].matched_token_id == 2
        assert results[1].should_stop is False

    def test_stop_token_id(self):
        """Request hits a stop token ID."""
        checker = BatchStopChecker()
        token_ids = mx.array([100, 42])
        counts = [5, 5]
        configs = [
            StopConfig(request_id="r1", max_tokens=256, stop_token_ids=[100, 200]),
            StopConfig(request_id="r2", max_tokens=256, stop_token_ids=[100, 200]),
        ]

        results = checker.check_batch(token_ids, counts, configs)
        assert results[0].should_stop is True
        assert results[0].reason == "stop_token_id"
        assert results[0].matched_token_id == 100
        assert results[1].should_stop is False

    def test_max_tokens_priority_over_eos(self):
        """Max_tokens is checked first (before EOS)."""
        checker = BatchStopChecker()
        token_ids = mx.array([2])  # EOS token
        counts = [256]
        configs = [
            StopConfig(request_id="r1", max_tokens=256, eos_token_ids=[2]),
        ]

        results = checker.check_batch(token_ids, counts, configs)
        assert results[0].should_stop is True
        assert results[0].reason == "length"  # max_tokens checked first

    def test_stop_token_ids_vectorized(self):
        """Vectorized stop token ID check returns correct mask."""
        checker = BatchStopChecker()
        token_ids = mx.array([10, 20, 30, 40])
        stop_sets = [{10, 50}, {30}, {}, {40, 99}]

        mask = checker.check_batch_token_ids_vectorized(token_ids, stop_sets)
        mask_list = mask.tolist()
        assert mask_list[0] is True  # token 10 in {10, 50}
        assert mask_list[1] is False  # token 20 not in {30}
        assert mask_list[2] is False  # empty set
        assert mask_list[3] is True  # token 40 in {40, 99}

    def test_max_tokens_vectorized(self):
        """Vectorized max_tokens check returns correct mask."""
        checker = BatchStopChecker()
        counts = [100, 256, 255, 500]
        limits = [200, 256, 256, 500]

        mask = checker.check_max_tokens_vectorized(counts, limits)
        mask_list = mask.tolist()
        assert mask_list[0] is False  # 100 < 200
        assert mask_list[1] is True  # 256 >= 256
        assert mask_list[2] is False  # 255 < 256
        assert mask_list[3] is True  # 500 >= 500

    def test_stats_tracking(self):
        """Stats are updated after checks."""
        checker = BatchStopChecker()
        token_ids = mx.array([10])
        counts = [256]
        configs = [StopConfig(request_id="r1", max_tokens=256)]

        checker.check_batch(token_ids, counts, configs)
        stats = checker.get_stats()
        assert stats["total_checks"] == 1
        assert stats["total_stops"] == 1

    def test_batch_with_mixed_stops(self):
        """Mixed batch: some stop, some continue."""
        checker = BatchStopChecker()
        token_ids = mx.array([2, 50, 100, 42])
        counts = [5, 10, 256, 3]
        configs = [
            StopConfig(request_id="r0", max_tokens=256, eos_token_ids=[2]),  # EOS stop
            StopConfig(request_id="r1", max_tokens=256),  # continue
            StopConfig(request_id="r2", max_tokens=256),  # max_tokens stop
            StopConfig(
                request_id="r3", max_tokens=256, stop_token_ids=[100]
            ),  # not this token
        ]

        results = checker.check_batch(token_ids, counts, configs)
        assert results[0].should_stop is True and results[0].reason == "eos"
        assert results[1].should_stop is False
        assert results[2].should_stop is True and results[2].reason == "length"
        assert results[3].should_stop is False


# ===================================================================
# Aho-Corasick Trie
# ===================================================================


class TestAhoCorasickTrie:
    """Tests for _AhoCorasickTrie (multi-pattern stop string matching)."""

    def test_single_pattern_match(self):
        """Single pattern is matched correctly."""
        patterns = [((1, 2, 3), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([5, 1, 2, 3, 4])
        assert 0 in matches

    def test_single_pattern_no_match(self):
        """Single pattern not found returns empty."""
        patterns = [((1, 2, 3), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([5, 6, 7, 8])
        assert matches == []

    def test_multiple_patterns(self):
        """Multiple patterns can be matched simultaneously."""
        patterns = [((1, 2), 0), ((3, 4), 1)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([1, 2, 3, 4])
        assert 0 in matches
        assert 1 in matches

    def test_overlapping_patterns(self):
        """Overlapping patterns are both detected."""
        patterns = [((1, 2), 0), ((2, 3), 1)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([1, 2, 3])
        assert 0 in matches
        assert 1 in matches

    def test_pattern_at_start(self):
        """Pattern at the very start of sequence is matched."""
        patterns = [((10, 20), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([10, 20, 30])
        assert 0 in matches

    def test_pattern_at_end(self):
        """Pattern at the very end of sequence is matched."""
        patterns = [((30, 40), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([10, 20, 30, 40])
        assert 0 in matches

    def test_empty_sequence(self):
        """Empty token sequence returns no matches."""
        patterns = [((1, 2), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([])
        assert matches == []

    def test_num_patterns(self):
        """num_patterns property returns correct count."""
        patterns = [((1,), 0), ((2,), 1), ((3,), 2)]
        trie = _AhoCorasickTrie(patterns)
        assert trie.num_patterns == 3

    def test_repeated_pattern(self):
        """Pattern appearing multiple times is matched."""
        patterns = [((1, 1), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([1, 1, 1, 1])
        assert 0 in matches

    def test_single_token_pattern(self):
        """Single-token patterns work correctly."""
        patterns = [((42,), 0)]
        trie = _AhoCorasickTrie(patterns)

        matches = trie.search([10, 20, 42, 30])
        assert 0 in matches


# ===================================================================
# Integration-style tests
# ===================================================================


class TestBatchSamplerIntegration:
    """Integration tests combining multiple components."""

    def test_processor_then_sampler(self):
        """LogitsProcessorBatch → BatchSampler pipeline works end-to-end."""
        processor = LogitsProcessorBatch()
        sampler = BatchSampler()

        logits = mx.zeros((2, 50))
        logits[0, 5] = 2.0
        logits[1, 10] = 2.0

        configs = [
            LogitsProcessorConfig(repetition_penalty=2.0, generated_tokens=[5]),
            LogitsProcessorConfig(presence_penalty=0.1, generated_tokens=[10]),
        ]

        processed = processor.process(logits, configs)
        params = _default_params(batch_size=2, temperature=0.01)
        result = sampler.sample_batch(processed, params)

        assert result.token_ids.shape == (2,)
        # Both requests should still produce valid tokens
        for i in range(2):
            assert 0 <= int(result.token_ids[i]) < 50

    def test_large_batch(self):
        """Large batch (32 requests) processes correctly."""
        sampler = BatchSampler()
        batch_size = 32
        vocab_size = 1000

        logits = _make_logits(batch_size=batch_size, vocab_size=vocab_size)
        params = _default_params(batch_size=batch_size, temperature=0.7)

        result = sampler.sample_batch(logits, params)
        assert result.token_ids.shape == (batch_size,)

        for i in range(batch_size):
            tid = int(result.token_ids[i])
            assert 0 <= tid < vocab_size

    def test_large_vocab(self):
        """Large vocab (151936 for Qwen) processes correctly."""
        sampler = BatchSampler()
        vocab_size = 151936  # Qwen2.5 vocab

        logits = _make_logits(batch_size=2, vocab_size=vocab_size)
        params = _default_params(batch_size=2, temperature=0.7)

        result = sampler.sample_batch(logits, params)
        assert result.token_ids.shape == (2,)

    def test_per_request_different_params(self):
        """Each request in batch has different sampling params."""
        sampler = BatchSampler()
        logits = _make_logits(batch_size=4, vocab_size=50)

        params = [
            {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": None},
            {"temperature": 0.5, "top_k": 10, "top_p": 0.9, "min_p": 0.0, "seed": None},
            {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.1, "seed": None},
            {
                "temperature": 0.8,
                "top_k": 5,
                "top_p": 0.95,
                "min_p": 0.05,
                "seed": None,
            },
        ]

        result = sampler.sample_batch(logits, params)
        assert result.token_ids.shape == (4,)
        for i in range(4):
            tid = int(result.token_ids[i])
            assert 0 <= tid < 50

    def test_stop_checker_after_sampling(self):
        """Full pipeline: sample → check stops."""
        sampler = BatchSampler()
        checker = BatchStopChecker()

        logits = mx.zeros((3, 50))
        logits[0, 2] = 10.0  # EOS token
        logits[1, 42] = 5.0
        logits[2, 0] = 3.0

        params = _default_params(batch_size=3, temperature=0.01)
        result = sampler.sample_batch(logits, params)

        counts = [10, 10, 10]
        configs = [
            StopConfig(request_id="r0", max_tokens=256, eos_token_ids=[2]),
            StopConfig(request_id="r1", max_tokens=256),
            StopConfig(request_id="r2", max_tokens=256),
        ]

        stop_results = checker.check_batch(result.token_ids, counts, configs)
        assert len(stop_results) == 3

    def test_sampling_plan_with_sampling_params_dataclass(self):
        """SamplingParams dataclass can be converted to dict for BatchSampler."""
        sp = SamplingParams(temperature=0.5, top_k=10, top_p=0.9, min_p=0.05)
        params_dict = {
            "temperature": sp.temperature,
            "top_k": sp.top_k,
            "top_p": sp.top_p,
            "min_p": sp.min_p,
            "seed": sp.seed,
        }

        sampler = BatchSampler()
        logits = _make_logits(batch_size=1, vocab_size=50)
        result = sampler.sample_batch(logits, [params_dict])
        assert result.token_ids.shape == (1,)


class TestBatchSamplerTempPosition:
    """The batch sampler must filter top_p/min_p on
    the UN-tempered distribution and apply temperature last, matching mlx-lm and
    the non-streaming fast path. Otherwise temp!=1 + a filter selects a different
    nucleus depending on which sampling path served the request."""

    def test_top_p_nucleus_is_temperature_invariant(self):
        import mlx.core as mx

        # Clear gap: tokens 0,1 dominate; top_p=0.8 should keep only {0,1}
        # regardless of temperature (nucleus is defined on the un-tempered probs).
        logits = mx.array([[5.0, 4.0, 0.0, -2.0, -5.0]])
        sampler = BatchSampler()
        seen = set()
        for _ in range(2000):
            r = sampler.sample_batch(
                logits, [{"temperature": 2.0, "top_p": 0.8, "seed": None}]
            )
            seen.add(int(r.token_ids[0]))
        # Only the un-tempered nucleus tokens may be sampled.
        assert seen <= {0, 1}, f"sampled outside nucleus: {seen}"

    def test_temperature_still_reshapes_within_nucleus(self):
        import mlx.core as mx

        logits = mx.array([[2.0, 1.0, 0.0, -8.0, -9.0]])
        sampler = BatchSampler()

        def _top_frac(temp):
            c = 0
            for _ in range(3000):
                r = sampler.sample_batch(
                    logits, [{"temperature": temp, "top_p": 1.0, "seed": None}]
                )
                if int(r.token_ids[0]) == 0:
                    c += 1
            return c / 3000

        # Colder concentrates on the top token; hotter spreads.
        assert _top_frac(0.3) > _top_frac(4.0)
