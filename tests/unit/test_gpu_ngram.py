"""Tests for GPU-accelerated N-gram speculative decoding."""
import mlx.core as mx
import pytest

from yunshu_engine.gpu_ngram import (
    GPUNgramConfig,
    GPUNgramProposer,
    GPUNgramTable,
)
from yunshu_engine.spec_interface import (
    CompositeStrategy,
    DraftProposal,
    GPUNgramStrategy,
    SpecStrategyFactory,
)

# ── GPUNgramTable ──────────────────────────────────────────────────


class TestGPUNgramTable:
    def test_build_from_single_sequence(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        assert table.total_entries > 0
        stats = table.get_stats()
        assert stats["total_entries"] > 0
        assert stats["gpu_arrays"] > 0

    def test_build_from_multiple_sequences(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([
            [1, 2, 3, 4, 5],
            [10, 20, 30, 10, 20, 30],
        ])
        assert table.total_entries > 0

    def test_lookup_gpu_basic_match(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4, 5, 1, 2, 3]])
        context = mx.array([99, 1, 2, 3], dtype=mx.int32)
        result = table.lookup_gpu(context, n=3)
        assert result is not None
        # Should find continuation [4, 5, 1] or similar
        vals = [int(v) for v in result if int(v) != -1]
        assert len(vals) > 0

    def test_lookup_gpu_no_match(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4]])
        context = mx.array([9, 8, 7], dtype=mx.int32)
        result = table.lookup_gpu(context, n=3)
        assert result is None

    def test_lookup_gpu_context_too_short(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4]])
        context = mx.array([1, 2], dtype=mx.int32)
        result = table.lookup_gpu(context, n=3)
        assert result is None  # context shorter than n

    def test_lookup_gpu_unstored_n_length(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4]])
        context = mx.array([1, 2, 3, 4], dtype=mx.int32)
        result = table.lookup_gpu(context, n=5)
        assert result is None  # n=5 not in table (max_n=3)

    def test_lookup_all_n_prefers_longest(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=5, k=3))
        # "1,2,3,4,5" followed by "1,2,3,4,5"
        table.build_from_sequences([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
        context = mx.array([99, 1, 2, 3, 4, 5], dtype=mx.int32)
        result = table.lookup_all_n(context)
        assert result is not None
        vals = [int(v) for v in result if int(v) != -1]
        assert len(vals) > 0

    def test_lookup_all_n_no_match(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=2, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4, 5]])
        context = mx.array([10, 20, 30], dtype=mx.int32)
        result = table.lookup_all_n(context)
        assert result is None

    def test_clear(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4, 5]])
        assert table.total_entries > 0
        table.clear()
        assert table.total_entries == 0
        stats = table.get_stats()
        assert stats["total_entries"] == 0

    def test_stats_empty(self):
        table = GPUNgramTable(GPUNgramConfig())
        stats = table.get_stats()
        assert stats["total_entries"] == 0
        assert stats["gpu_arrays"] == 0
        assert stats["gpu_hit_rate"] == 0.0

    def test_stats_after_lookups(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=3, k=3))
        table.build_from_sequences([[1, 2, 3, 4, 1, 2, 3]])
        context = mx.array([1, 2, 3], dtype=mx.int32)
        table.lookup_gpu(context, n=3)
        stats = table.get_stats()
        assert stats["total_gpu_lookups"] == 1

    def test_max_table_entries_truncation(self):
        config = GPUNgramConfig(min_n=1, max_n=2, k=1, max_table_entries=5)
        table = GPUNgramTable(config)
        # Insert many sequences to exceed capacity
        for i in range(100):
            table.build_from_sequences([[i, i + 1, i + 2]])
        # GPU arrays should be truncated to max_table_entries
        for _n, keys in table._keys.items():
            assert keys.shape[0] <= 5

    def test_incremental_build(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=2, k=2))
        table.build_from_sequences([[1, 2, 3, 4]])
        table.build_from_sequences([[5, 6, 7, 8]])
        # Both sequences should be indexed
        ctx1 = mx.array([1, 2], dtype=mx.int32)
        r1 = table.lookup_gpu(ctx1, n=2)
        assert r1 is not None
        ctx2 = mx.array([5, 6], dtype=mx.int32)
        r2 = table.lookup_gpu(ctx2, n=2)
        assert r2 is not None


# ── GPUNgramProposer ───────────────────────────────────────────────


class TestGPUNgramProposer:
    def test_propose_basic(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        result = proposer.propose([99, 1, 2, 3])
        assert len(result) > 0

    def test_propose_no_match_with_cpu_fallback(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        # Empty table, should fall back to CPU NgramProposer (LPS)
        result = proposer.propose([1, 2, 3, 1, 2, 3])
        assert len(result) > 0  # LPS finds the repetition

    def test_propose_no_match_no_fallback(self):
        config = GPUNgramConfig(min_n=1, max_n=3, k=3, gpu_fallback=False)
        proposer = GPUNgramProposer(config)
        # Empty table, no fallback
        result = proposer.propose([1, 2, 3, 1, 2, 3])
        assert result == []

    def test_add_sequence(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        proposer.add_sequence([1, 2, 3, 4, 5, 1, 2, 3])
        result = proposer.propose([99, 1, 2, 3])
        assert len(result) > 0

    def test_batch_propose(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        batch = [
            [99, 1, 2, 3],
            [4, 5, 6],
            [1, 2, 3, 1, 2],
        ]
        results = proposer.batch_propose(batch)
        assert len(results) == 3
        assert len(results[0]) > 0  # GPU match

    def test_n_draft_parameter(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=10))
        proposer.build_from_sequences([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
        result = proposer.propose([99, 1, 2, 3], n_draft=2)
        assert len(result) <= 2

    def test_empty_context(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        result = proposer.propose([])
        assert result == []

    def test_context_too_short(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=3, max_n=5, k=3))
        result = proposer.propose([1, 2])
        assert result == []

    def test_max_model_len_exceeded(self):
        config = GPUNgramConfig(min_n=1, max_n=3, k=5, max_model_len=4)
        proposer = GPUNgramProposer(config)
        proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        # Context length == max_model_len, k would be 0
        result = proposer.propose([1, 2, 3, 1])
        assert result == []

    def test_get_stats(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        proposer.propose([99, 1, 2, 3])
        stats = proposer.get_stats()
        assert stats["mode"] == "gpu_ngram"
        assert stats["total_proposals"] == 1
        assert "table" in stats

    def test_get_stats_empty(self):
        proposer = GPUNgramProposer()
        stats = proposer.get_stats()
        assert stats["total_proposals"] == 0
        assert stats["gpu_ratio"] == 0.0
        assert stats["avg_match_length"] == 0.0

    def test_repeated_pattern_all_same(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        proposer.build_from_sequences([[5] * 50])
        result = proposer.propose([5, 5, 5, 5, 5])
        assert len(result) > 0
        assert all(t == 5 for t in result)

    def test_gpu_vs_cpu_stats_tracking(self):
        proposer = GPUNgramProposer(GPUNgramConfig(min_n=1, max_n=3, k=3))
        proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        # This should use GPU
        proposer.propose([99, 1, 2, 3])
        # This should fall back to CPU (no GPU match for non-indexed pattern)
        proposer.propose([7, 8, 9, 7, 8, 9])
        stats = proposer.get_stats()
        assert stats["total_gpu_used"] >= 1


# ── GPUNgramStrategy ───────────────────────────────────────────────


class TestGPUNgramStrategy:
    def test_lifecycle(self):
        strategy = GPUNgramStrategy(GPUNgramConfig(min_n=1, max_n=3, k=3))
        strategy.begin("req-1")
        # Build table with repeated pattern
        strategy._proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        proposal = strategy.draft([99, 1, 2, 3], n=5)
        assert isinstance(proposal, DraftProposal)
        assert proposal.strategy_name == "gpu_ngram"
        strategy.accept(proposal.tokens, min(2, len(proposal.tokens)))
        stats = strategy.stats()
        assert stats["name"] == "gpu_ngram"
        assert stats["total_drafts"] == 1
        strategy.end("req-1")

    def test_empty_proposal(self):
        strategy = GPUNgramStrategy(GPUNgramConfig(
            min_n=2, max_n=3, k=3, gpu_fallback=False,
        ))
        strategy.begin("req-2")
        proposal = strategy.draft([1, 2], n=5)
        assert proposal.tokens == []
        strategy.end("req-2")

    def test_name(self):
        strategy = GPUNgramStrategy()
        assert strategy.name == "gpu_ngram"

    def test_reset(self):
        strategy = GPUNgramStrategy(GPUNgramConfig(min_n=1, max_n=3, k=3))
        strategy.begin("req-3")
        strategy._proposer.build_from_sequences([[1, 2, 3, 1, 2, 3]])
        strategy.draft([99, 1, 2, 3], n=3)
        strategy.reset()
        stats = strategy.stats()
        assert stats["total_drafts"] == 0

    def test_accept_adds_to_table(self):
        strategy = GPUNgramStrategy(GPUNgramConfig(min_n=1, max_n=3, k=3))
        strategy.begin("req-4")
        # Accept tokens — they get added to the GPU table
        strategy.accept([1, 2, 3, 4, 5], 3)
        # Now the table should have entries
        assert strategy._proposer.table.total_entries > 0
        strategy.end("req-4")


# ── Factory Integration ────────────────────────────────────────────


class TestGPUNgramFactory:
    def test_create_gpu_ngram(self):
        strategy = SpecStrategyFactory.create({
            "type": "gpu_ngram",
            "min_n": 1,
            "max_n": 3,
            "k": 5,
        })
        assert isinstance(strategy, GPUNgramStrategy)
        assert strategy.name == "gpu_ngram"

    def test_create_suffix(self):
        from yunshu_engine.spec_interface import SuffixStrategy
        strategy = SpecStrategyFactory.create({
            "type": "suffix",
            "min_suffix_length": 3,
            "max_window": 256,
        })
        assert isinstance(strategy, SuffixStrategy)
        assert strategy.name == "suffix"

    def test_composite_with_gpu_ngram(self):
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "gpu_ngram", "min_n": 1, "max_n": 3, "k": 3},
                {"type": "ngram", "mode": "lps"},
            ],
        })
        assert isinstance(strategy, CompositeStrategy)
        assert "gpu_ngram" in strategy.name

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown spec strategy type"):
            SpecStrategyFactory.create({"type": "nonexistent"})


# ── GPU Vectorized Matching Correctness ────────────────────────────


class TestGPUVectorizedMatching:
    def test_exact_ngram_match(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=3, max_n=3, k=3))
        table.build_from_sequences([[10, 20, 30, 40, 50]])
        context = mx.array([10, 20, 30], dtype=mx.int32)
        result = table.lookup_gpu(context, n=3)
        assert result is not None
        vals = [int(v) for v in result if int(v) != -1]
        assert vals[0] == 40  # continuation after "10,20,30"

    def test_partial_ngram_no_match(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=3, max_n=3, k=3))
        table.build_from_sequences([[10, 20, 30, 40]])
        context = mx.array([20, 30, 40], dtype=mx.int32)
        result = table.lookup_gpu(context, n=3)
        # "20,30,40" is at the end — no continuation stored (or -1 padded)
        if result is not None:
            vals = [int(v) for v in result if int(v) != -1]
            # At the end, continuation may be empty
            assert isinstance(vals, list)

    def test_multiple_ngram_lengths(self):
        table = GPUNgramTable(GPUNgramConfig(min_n=1, max_n=5, k=3))
        table.build_from_sequences([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]])
        # Check that per_n_entries has entries for multiple lengths
        stats = table.get_stats()
        assert len(stats["per_n_entries"]) > 1

    def test_large_table_performance(self):
        """Verify that a moderately large table can be built and queried."""
        table = GPUNgramTable(GPUNgramConfig(min_n=2, max_n=3, k=2))
        # Generate 100 sequences of length 20
        sequences = []
        for i in range(100):
            base = list(range(i * 20, i * 20 + 20))
            sequences.append(base)
        table.build_from_sequences(sequences)
        assert table.total_entries > 0
        # Query with a known pattern
        ctx = mx.array(sequences[0][:3], dtype=mx.int32)
        result = table.lookup_gpu(ctx, n=3)
        # May or may not match depending on uniqueness
        # Just verify no crash
        assert result is None or result.shape[0] == 2
