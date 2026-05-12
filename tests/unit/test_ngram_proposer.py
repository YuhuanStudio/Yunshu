"""Tests for N-gram speculative decoding proposer."""
import pytest

from yunshu_engine.ngram_proposer import (
    NgramConfig,
    NgramProposer,
    _find_longest_ngram_and_propose,
)


class TestFindLongestNgram:
    def test_repeated_sequence(self):
        tokens = [1, 2, 3, 1, 2, 3]
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=5, max_model_len=100, k=3)
        assert len(result) > 0
        # After matching "1,2,3" suffix, should propose tokens following
        # the first occurrence: tokens[3:6] = [1, 2, 3]
        assert result == [1, 2, 3]

    def test_no_repetition(self):
        tokens = [1, 2, 3, 4, 5]
        result = _find_longest_ngram_and_propose(tokens, min_n=2, max_n=5, max_model_len=100, k=3)
        assert result == []

    def test_single_token_repeat(self):
        tokens = [5, 1, 2, 5]
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=5, max_model_len=100, k=3)
        assert len(result) > 0
        # Matches "5" at suffix, proposes "1, 2, 5"
        assert result[0] == 1

    def test_too_short(self):
        tokens = [1, 2]
        result = _find_longest_ngram_and_propose(tokens, min_n=3, max_n=5, max_model_len=100, k=3)
        assert result == []

    def test_k_limited_by_remaining(self):
        tokens = [1, 2, 1, 2, 1]
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=5, max_model_len=100, k=10)
        # Should not propose more tokens than available
        assert len(result) <= len(tokens)

    def test_k_limited_by_max_model_len(self):
        tokens = [1, 2, 1, 2, 1]
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=5, max_model_len=5, k=5)
        # k = min(5, 5-5) = 0
        assert result == []

    def test_exact_ngram_match(self):
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        result = _find_longest_ngram_and_propose(tokens, min_n=3, max_n=5, max_model_len=100, k=5)
        assert len(result) == 5
        assert result == [1, 2, 3, 4, 5]

    def test_max_n_limits_search(self):
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        # max_n=2 means only match 2-grams
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=2, max_model_len=100, k=3)
        assert len(result) > 0

    def test_empty_tokens(self):
        result = _find_longest_ngram_and_propose([], min_n=1, max_n=5, max_model_len=100, k=3)
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
        assert results[1] == []     # No repetition with min_n=1+max_n=5
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


class TestNgramEdgeCases:
    def test_all_same_tokens(self):
        tokens = [5] * 100
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=5, max_model_len=200, k=5)
        assert len(result) == 5
        assert all(t == 5 for t in result)

    def test_palindromic_sequence(self):
        tokens = [1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        result = _find_longest_ngram_and_propose(tokens, min_n=3, max_n=5, max_model_len=200, k=5)
        assert len(result) == 5

    def test_overlapping_matches(self):
        # "ababa" pattern
        tokens = [1, 2, 1, 2, 1, 1, 2, 1, 2, 1]
        result = _find_longest_ngram_and_propose(tokens, min_n=1, max_n=5, max_model_len=200, k=5)
        assert len(result) > 0
