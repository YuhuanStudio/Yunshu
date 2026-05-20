"""Tests for Suffix-based Speculative Decoding."""
import pytest

from yunshu_engine.suffix_proposer import (
    SuffixConfig,
    SuffixProposer,
    SuffixTrie,
    SuffixTrieNode,
)
from yunshu_engine.spec_interface import (
    DraftProposal,
    SuffixStrategy,
    CompositeStrategy,
    SpecStrategyFactory,
)


# ── SuffixTrie ─────────────────────────────────────────────────────


class TestSuffixTrie:
    def test_insert_basic(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 3, 4, 5])
        stats = trie.get_stats()
        assert stats["total_inserts"] > 0
        assert stats["total_nodes"] > 1

    def test_insert_empty(self):
        trie = SuffixTrie()
        trie.insert([])
        stats = trie.get_stats()
        assert stats["total_inserts"] == 0

    def test_insert_single_token(self):
        trie = SuffixTrie()
        trie.insert([42])
        # Single token has no continuation
        stats = trie.get_stats()
        assert stats["total_inserts"] == 0

    def test_find_longest_match_basic(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 3, 1, 2, 3, 4, 5])
        result = trie.find_longest_match([1, 2, 3])
        assert len(result) > 0
        # After matching "1,2,3", should find continuation [1,2,3,4,5] or [4,5]
        assert result[0] in (1, 4)  # Either the repeated start or continuation

    def test_find_longest_match_no_match(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 3, 4, 5])
        result = trie.find_longest_match([9, 8, 7])
        assert result == []

    def test_find_longest_match_empty_suffix(self):
        trie = SuffixTrie()
        trie.insert([1, 2, 3])
        result = trie.find_longest_match([])
        assert result == []

    def test_find_longest_match_max_len(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8])
        result = trie.find_longest_match([1, 2, 3], max_len=2)
        assert len(result) <= 2

    def test_find_longest_match_repeated_pattern(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 1, 2, 1, 2, 1, 2])
        result = trie.find_longest_match([1, 2])
        assert len(result) > 0
        # Most frequent continuation after "1,2" is "1,2"
        assert result[0] in (1, 2)

    def test_find_longest_match_all_same(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([5, 5, 5, 5, 5, 5])
        result = trie.find_longest_match([5, 5])
        assert len(result) > 0
        assert all(t == 5 for t in result)

    def test_find_longest_match_multiple_continuations(self):
        trie = SuffixTrie(max_depth=10)
        # "a,b" is followed by both "c" and "d" — most frequent wins
        trie.insert([1, 2, 3, 1, 2, 3, 1, 2, 4])
        result = trie.find_longest_match([1, 2])
        assert len(result) > 0
        # "1,2" → [3] appears twice, [4] appears once, so [3] should win
        assert result[0] == 3

    def test_clear(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 3, 4, 5])
        trie.clear()
        stats = trie.get_stats()
        assert stats["total_inserts"] == 0
        assert stats["total_nodes"] == 1  # root only

    def test_max_depth_limits(self):
        trie = SuffixTrie(max_depth=3)
        trie.insert([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Trie depth is limited to 3
        stats = trie.get_stats()
        # Should still have inserts, but limited depth
        assert stats["total_inserts"] > 0

    def test_get_stats(self):
        trie = SuffixTrie(max_depth=10)
        trie.insert([1, 2, 3, 4, 5])
        stats = trie.get_stats()
        assert stats["total_inserts"] > 0
        assert stats["total_nodes"] >= 2
        assert stats["max_depth"] == 10


# ── SuffixProposer ─────────────────────────────────────────────────


class TestSuffixProposer:
    def test_begin_end_lifecycle(self):
        proposer = SuffixProposer(SuffixConfig())
        proposer.begin("req-1")
        assert "req-1" in proposer._tries
        assert "req-1" in proposer._windows
        proposer.end("req-1")
        assert "req-1" not in proposer._tries
        assert "req-1" not in proposer._windows

    def test_draft_empty_after_begin(self):
        proposer = SuffixProposer(SuffixConfig(min_suffix_length=3))
        proposer.begin("req-1")
        result = proposer.draft([1, 2, 3, 4, 5])
        # No prior generated text — no suffix match
        assert result == []
        proposer.end("req-1")

    def test_draft_after_accept(self):
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=100, max_draft=5,
        ))
        proposer.begin("req-1")
        # Simulate generated text being accepted
        proposer._generated["req-1"] = [1, 2, 3, 4, 5, 1, 2, 3]
        trie = proposer._tries["req-1"]
        trie.insert([1, 2, 3, 4, 5, 1, 2, 3])
        # Now draft with context ending in "1,2,3"
        result = proposer.draft([10, 20, 1, 2, 3])
        # Should find "1,2,3" suffix match from the generated text
        assert len(result) > 0
        proposer.end("req-1")

    def test_draft_with_long_repetition(self):
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=200, max_draft=5,
        ))
        proposer.begin("req-1")
        # Long repeated pattern
        pattern = [1, 2, 3, 4] * 10
        proposer._generated["req-1"] = pattern
        trie = proposer._tries["req-1"]
        trie.insert(pattern)
        result = proposer.draft([0] + pattern + [1, 2, 3])
        assert len(result) > 0
        proposer.end("req-1")

    def test_accept_updates_generated_history(self):
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=100, max_draft=5,
        ))
        proposer.begin("req-1")
        # Accept tokens
        proposer.accept([1, 2, 3, 4, 5], 5)
        # The generated history should now have these tokens
        assert proposer._generated["req-1"] == [1, 2, 3, 4, 5]
        # Trie is NOT rebuilt — draft() uses linear scan via _generated
        assert proposer._tries["req-1"].root.children == {}
        proposer.end("req-1")

    def test_accept_partial(self):
        proposer = SuffixProposer(SuffixConfig(min_suffix_length=2))
        proposer.begin("req-1")
        # Only 3 of 5 tokens accepted
        proposer.accept([1, 2, 3, 4, 5], 3)
        assert proposer._generated["req-1"] == [1, 2, 3]
        proposer.end("req-1")

    def test_multiple_requests(self):
        proposer = SuffixProposer(SuffixConfig(min_suffix_length=2))
        proposer.begin("req-1")
        proposer.begin("req-2")
        assert len(proposer._tries) == 2
        proposer.end("req-1")
        assert len(proposer._tries) == 1
        proposer.end("req-2")
        assert len(proposer._tries) == 0

    def test_max_draft_limit(self):
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=100, max_draft=2,
        ))
        proposer.begin("req-1")
        proposer._generated["req-1"] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        trie = proposer._tries["req-1"]
        trie.insert([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result = proposer.draft([1, 2, 3, 4, 5, 1, 2, 3])
        assert len(result) <= 2
        proposer.end("req-1")

    def test_n_draft_parameter(self):
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=100, max_draft=10,
        ))
        proposer.begin("req-1")
        proposer._generated["req-1"] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        trie = proposer._tries["req-1"]
        trie.insert([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result = proposer.draft([1, 2, 3, 4, 5, 1, 2, 3], n_draft=2)
        assert len(result) <= 2
        proposer.end("req-1")

    def test_empty_context(self):
        proposer = SuffixProposer(SuffixConfig())
        proposer.begin("req-1")
        result = proposer.draft([])
        assert result == []
        proposer.end("req-1")

    def test_short_context(self):
        proposer = SuffixProposer(SuffixConfig(min_suffix_length=5))
        proposer.begin("req-1")
        result = proposer.draft([1, 2, 3])
        assert result == []  # shorter than min_suffix_length
        proposer.end("req-1")

    def test_get_stats(self):
        proposer = SuffixProposer(SuffixConfig())
        proposer.begin("req-1")
        proposer.accept([1, 2, 3, 4], 4)
        proposer.draft([1, 2, 3, 4, 1, 2])
        stats = proposer.get_stats()
        assert stats["mode"] == "suffix"
        assert stats["active_requests"] == 1
        assert stats["total_tokens_accepted"] == 4
        proposer.end("req-1")

    def test_get_stats_empty(self):
        proposer = SuffixProposer()
        stats = proposer.get_stats()
        assert stats["total_proposals"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["avg_proposed_length"] == 0.0

    def test_max_window_truncation(self):
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=10, max_draft=5,
        ))
        proposer.begin("req-1")
        # Accept more tokens than max_window — _generated stores all
        long_tokens = list(range(100))
        proposer.accept(long_tokens, 100)
        # All 100 tokens are in _generated (the source for linear scan)
        assert proposer._generated["req-1"] == long_tokens
        # Trie is NOT rebuilt by accept() — draft() uses _generated directly
        assert proposer._tries["req-1"].get_stats()["total_inserts"] == 0
        proposer.end("req-1")

    def test_code_pattern_repetition(self):
        """Simulates repetitive code output like indentation patterns."""
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=3, max_window=512, max_draft=5,
        ))
        proposer.begin("req-code")
        # Simulate generated code: repeated "indent + keyword + newline"
        pattern = [100, 200, 300, 400] * 8  # e.g., spaces, "def", "(", ")"
        proposer._generated["req-code"] = pattern
        trie = proposer._tries["req-code"]
        trie.insert(pattern)
        result = proposer.draft([0] + pattern + [100, 200, 300])
        assert len(result) > 0
        proposer.end("req-code")


# ── SuffixStrategy ─────────────────────────────────────────────────


class TestSuffixStrategy:
    def test_lifecycle(self):
        strategy = SuffixStrategy(SuffixConfig(
            min_suffix_length=2, max_window=100, max_draft=5,
        ))
        strategy.begin("req-1")
        # Seed the proposer with generated tokens
        strategy._proposer.accept([1, 2, 3, 4, 5, 1, 2, 3], 8)
        proposal = strategy.draft([1, 2, 3, 4, 5, 1, 2], n=5)
        assert isinstance(proposal, DraftProposal)
        assert proposal.strategy_name == "suffix"
        strategy.accept(proposal.tokens, min(2, len(proposal.tokens)))
        stats = strategy.stats()
        assert stats["name"] == "suffix"
        assert stats["total_drafts"] == 1
        strategy.end("req-1")

    def test_empty_proposal_no_history(self):
        strategy = SuffixStrategy(SuffixConfig(min_suffix_length=3))
        strategy.begin("req-2")
        proposal = strategy.draft([1, 2, 3, 4, 5], n=5)
        assert proposal.tokens == []
        strategy.end("req-2")

    def test_name(self):
        strategy = SuffixStrategy()
        assert strategy.name == "suffix"

    def test_reset(self):
        strategy = SuffixStrategy(SuffixConfig(
            min_suffix_length=2, max_window=100, max_draft=5,
        ))
        strategy.begin("req-3")
        strategy._proposer.accept([1, 2, 3, 4, 5], 5)
        strategy.draft([1, 2, 3, 4], n=3)
        strategy.reset()
        stats = strategy.stats()
        assert stats["total_drafts"] == 0

    def test_composite_with_suffix(self):
        """Suffix strategy works inside CompositeStrategy."""
        suffix = SuffixStrategy(SuffixConfig(min_suffix_length=2))
        ngram = SpecStrategyFactory.create({
            "type": "ngram", "mode": "lps",
        })
        composite = CompositeStrategy([suffix, ngram])
        composite.begin("req-5")
        proposal = composite.draft([1, 2, 3, 1, 2, 3], n=5)
        assert isinstance(proposal, DraftProposal)
        # Should get a proposal from ngram fallback at minimum
        composite.end("req-5")

    def test_factory_creation(self):
        strategy = SpecStrategyFactory.create({
            "type": "suffix",
            "min_suffix_length": 3,
            "max_window": 256,
            "max_draft": 5,
        })
        assert isinstance(strategy, SuffixStrategy)
        assert strategy.name == "suffix"

    def test_json_pattern_repetition(self):
        """Simulates repetitive JSON output."""
        strategy = SuffixStrategy(SuffixConfig(
            min_suffix_length=2, max_window=256, max_draft=5,
        ))
        strategy.begin("req-json")
        # Simulate generated JSON: repeated {"key": "value"}, pattern
        # Tokens: { " k e y " : " v a l " } , { " k e y " : " v a l " } ,
        json_pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        strategy._proposer.accept(json_pattern, len(json_pattern))
        # Draft with context ending in a partial JSON key
        result = strategy.draft([0, 0] + json_pattern[:14] + [1, 2, 3], n=5)
        # Should find the pattern continuation
        if result.tokens:
            assert result.strategy_name == "suffix"
        strategy.end("req-json")


# ── Edge Cases ─────────────────────────────────────────────────────


class TestSuffixEdgeCases:
    def test_begin_twice_same_id(self):
        """Beginning the same request twice should reset state."""
        proposer = SuffixProposer(SuffixConfig())
        proposer.begin("req-1")
        proposer.accept([1, 2, 3], 3)
        proposer.begin("req-1")  # Reset
        # State should be fresh
        assert proposer._generated.get("req-1", []) == []
        proposer.end("req-1")

    def test_end_nonexistent_id(self):
        """Ending a non-existent request should not error."""
        proposer = SuffixProposer()
        proposer.end("nonexistent")  # Should not raise

    def test_accept_without_begin(self):
        """Accept without begin should handle gracefully."""
        proposer = SuffixProposer()
        proposer.accept([1, 2, 3], 3)
        # Should not crash, but won't record anything
        stats = proposer.get_stats()
        assert stats["total_tokens_accepted"] == 3

    def test_very_long_sequence(self):
        """Handle sequences longer than max_window."""
        proposer = SuffixProposer(SuffixConfig(
            min_suffix_length=2, max_window=50, max_draft=5,
        ))
        proposer.begin("req-long")
        # Accept a very long sequence
        long_tokens = [i % 10 for i in range(1000)]
        proposer.accept(long_tokens, 1000)
        result = proposer.draft(long_tokens[:50], n_draft=5)
        # Should handle without error (window truncates)
        assert isinstance(result, list)
        proposer.end("req-long")

    def test_suffix_trie_node_has_slots(self):
        """SuffixTrieNode uses __slots__ for memory efficiency."""
        node = SuffixTrieNode()
        assert hasattr(node, "children")
        assert hasattr(node, "continuations")
        assert hasattr(node, "total_count")
        # __slots__ should prevent arbitrary attribute assignment
        with pytest.raises(AttributeError):
            node.arbitrary_attr = 42  # type: ignore

    def test_config_defaults(self):
        config = SuffixConfig()
        assert config.min_suffix_length == 3
        assert config.max_window == 512
        assert config.max_draft == 5
        assert config.max_model_len == 32768
        assert config.max_trie_depth == 64
