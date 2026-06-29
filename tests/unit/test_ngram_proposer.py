"""Tests for N-gram speculative decoding proposer."""

from yunshu_engine.ngram_proposer import (
    NgramConfig,
    NgramHashPool,
    NgramProposer,
    _find_longest_ngram_and_propose,
)


class TestFindLongestNgram:
    def test_repeated_sequence(self):
        tokens = [1, 2, 3, 1, 2, 3]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=5, max_model_len=100, k=3
        )
        assert len(result) > 0
        # After matching "1,2,3" suffix, should propose tokens following
        # the first occurrence: tokens[3:6] = [1, 2, 3]
        assert result == [1, 2, 3]

    def test_no_repetition(self):
        tokens = [1, 2, 3, 4, 5]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=2, max_n=5, max_model_len=100, k=3
        )
        assert result == []

    def test_single_token_repeat(self):
        tokens = [5, 1, 2, 5]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=5, max_model_len=100, k=3
        )
        assert len(result) > 0
        # Matches "5" at suffix, proposes "1, 2, 5"
        assert result[0] == 1

    def test_too_short(self):
        tokens = [1, 2]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=3, max_n=5, max_model_len=100, k=3
        )
        assert result == []

    def test_k_limited_by_remaining(self):
        tokens = [1, 2, 1, 2, 1]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=5, max_model_len=100, k=10
        )
        # Should not propose more tokens than available
        assert len(result) <= len(tokens)

    def test_k_limited_by_max_model_len(self):
        tokens = [1, 2, 1, 2, 1]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=5, max_model_len=5, k=5
        )
        # k = min(5, 5-5) = 0
        assert result == []

    def test_exact_ngram_match(self):
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=3, max_n=5, max_model_len=100, k=5
        )
        assert len(result) == 5
        assert result == [1, 2, 3, 4, 5]

    def test_max_n_limits_search(self):
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        # max_n=2 means only match 2-grams
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=2, max_model_len=100, k=3
        )
        assert len(result) > 0

    def test_empty_tokens(self):
        result = _find_longest_ngram_and_propose(
            [], min_n=1, max_n=5, max_model_len=100, k=3
        )
        assert result == []


class TestNgramProposer:
    def test_propose_basic(self):
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=3))
        tokens = [1, 2, 3, 1, 2, 3]
        result = proposer.propose(tokens)
        assert len(result) > 0

    def test_batch_propose(self):
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=3))
        batch = [
            [1, 2, 3, 1, 2, 3],
            [4, 5, 6],
            [7, 8, 7, 8, 7],
        ]
        results = proposer.batch_propose(batch)
        assert len(results) == 3
        assert len(results[0]) > 0  # Has repetition
        assert results[1] == []  # No repetition with min_n=1+max_n=5
        assert len(results[2]) > 0  # Has "7,8" repetition

    def test_config_defaults(self):
        config = NgramConfig()
        assert config.min_n == 1
        assert config.max_n == 5
        assert config.k == 5

    def test_no_model_loading_needed(self):
        proposer = NgramProposer(NgramConfig())
        # Should work without any model loading
        result = proposer.propose([1, 2, 1, 2, 1])
        assert len(result) > 0

    def test_stats_lps_mode(self):
        proposer = NgramProposer(NgramConfig(mode="lps"))
        stats = proposer.get_stats()
        assert stats["mode"] == "lps"


class TestNgramEdgeCases:
    def test_all_same_tokens(self):
        tokens = [5] * 100
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=5, max_model_len=200, k=5
        )
        assert len(result) == 5
        assert all(t == 5 for t in result)

    def test_palindromic_sequence(self):
        tokens = [1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=3, max_n=5, max_model_len=200, k=5
        )
        assert len(result) == 5

    def test_overlapping_matches(self):
        # "ababa" pattern
        tokens = [1, 2, 1, 2, 1, 1, 2, 1, 2, 1]
        result = _find_longest_ngram_and_propose(
            tokens, min_n=1, max_n=5, max_model_len=200, k=5
        )
        assert len(result) > 0


class TestNgramHashPool:
    """Tests for the O(1) hash pool mode (llama.cpp ngram-mod pattern)."""

    def test_basic_update_and_propose(self):
        pool = NgramHashPool(NgramConfig(min_n=1, max_n=3, k=3))
        # Index a sequence with repetition
        pool.update([1, 2, 3, 1, 2, 3])
        # Suffix "1,2,3" should match and propose continuation
        result = pool.propose([10, 20, 1, 2, 3])
        assert result == [1, 2, 3]

    def test_longer_ngram_preferred(self):
        pool = NgramHashPool(NgramConfig(min_n=1, max_n=5, k=3))
        # "a,b,c" maps to [4], "a,b,c,4" maps to [5]
        pool.update([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        # Suffix "1,2,3,4,5" → should find 5-gram match
        result = pool.propose([99, 1, 2, 3, 4, 5])
        assert len(result) > 0

    def test_empty_proposal_no_match(self):
        pool = NgramHashPool(NgramConfig(min_n=2, max_n=3, k=3))
        pool.update([1, 2, 3])
        # No ngram of length 2+ appears twice
        result = pool.propose([4, 5, 6])
        assert result == []

    def test_capacity_eviction(self):
        pool = NgramHashPool(NgramConfig(min_n=1, max_n=2, k=1, hashpool_capacity=5))
        # Insert many ngrams to trigger eviction
        for i in range(100):
            pool.update([i, i + 1, i + 2])
        assert len(pool._pool) <= 5
        stats = pool.get_stats()
        assert stats["total_evictions"] > 0

    def test_clear(self):
        pool = NgramHashPool(NgramConfig(min_n=1, max_n=3, k=3))
        pool.update([1, 2, 3, 1, 2, 3])
        assert len(pool._pool) > 0
        pool.clear()
        assert len(pool._pool) == 0

    def test_stats(self):
        pool = NgramHashPool(NgramConfig(min_n=1, max_n=2, k=2))
        pool.update([1, 2, 3])
        stats = pool.get_stats()
        assert stats["pool_size"] > 0
        assert stats["total_inserts"] > 0
        assert stats["total_evictions"] == 0

    def test_incremental_update(self):
        pool = NgramHashPool(NgramConfig(min_n=1, max_n=3, k=3))
        # First update
        pool.update([1, 2, 3, 4])
        # Second update — adds more context
        pool.update([1, 2, 3, 4, 5])
        # Should now find "4,5" → ?
        pool.propose([1, 2, 3, 4, 5])
        # The 5-gram "1,2,3,4,5" was indexed, but we're at the end
        # so there's no continuation — should return empty or fall to shorter
        # Actually "4,5" doesn't have a continuation indexed since it's at the end
        # But "1,2,3" → [4] was indexed from the first update
        result2 = pool.propose([99, 99, 1, 2, 3])
        assert result2[0] == 4

    def test_proposer_hashpool_mode(self):
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=3, mode="hashpool"))
        tokens = [1, 2, 3, 1, 2, 3]
        result = proposer.propose(tokens)
        assert len(result) > 0
        stats = proposer.get_stats()
        assert stats["mode"] == "hashpool"

    def test_hashpool_fallback_to_lps(self):
        """HashPool falls back to LPS when no match found in pool."""
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=3, mode="hashpool"))
        # No prior context — hashpool empty, LPS finds the repetition
        tokens = [1, 2, 3, 1, 2, 3]
        result = proposer.propose(tokens)
        # Should still find a proposal via LPS fallback
        assert len(result) > 0

    def test_hashpool_matches_lps_for_repeated(self):
        """Both modes should find proposals for clearly repeated patterns."""
        tokens = [1, 2, 3, 4, 5] * 3
        lps = NgramProposer(NgramConfig(min_n=1, max_n=5, k=5, mode="lps"))
        hp = NgramProposer(NgramConfig(min_n=1, max_n=5, k=5, mode="hashpool"))
        lps_result = lps.propose(tokens)
        hp_result = hp.propose(tokens)
        assert len(lps_result) > 0
        assert len(hp_result) > 0
