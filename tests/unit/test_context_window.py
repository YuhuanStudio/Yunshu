"""Tests for ContextWindowManager — context window truncation strategies."""

import pytest

from yunshu_engine.context_window import (
    ContextWindowManager,
    ContextWindowStats,
    TruncationResult,
    TruncationStrategy,
)


# Helper: create messages with known token counts.
# Default counter = len(text) // 4, so "abcd" = 1 token, "abcd" * 100 = 100 tokens.
def _msg(role: str, content: str, token_estimate: int | None = None) -> dict:
    """Create a message dict. If token_estimate given, pad content to match."""
    if token_estimate is not None:
        # 4 chars per token (default counter)
        target_chars = token_estimate * 4
        if len(content) < target_chars:
            content = content + "x" * (target_chars - len(content))
    return {"role": role, "content": content}


def _system_msg(token_estimate: int = 50) -> dict:
    return _msg("system", "You are a helpful assistant.", token_estimate)


def _user_msg(text: str = "Hello", token_estimate: int = 10) -> dict:
    return _msg("user", text, token_estimate)


def _assistant_msg(text: str = "Hi there", token_estimate: int = 10) -> dict:
    return _msg("assistant", text, token_estimate)


def _conversation(turns: int = 5, tokens_per_msg: int = 50) -> list[dict]:
    """Create a conversation with system + N user/assistant turns."""
    msgs = [_system_msg(tokens_per_msg)]
    for i in range(turns):
        msgs.append(_user_msg(f"Question {i}", tokens_per_msg))
        msgs.append(_assistant_msg(f"Answer {i}", tokens_per_msg))
    return msgs


# ── TruncationStrategy ──


class TestTruncationStrategy:
    def test_enum_values(self):
        assert TruncationStrategy.TRUNCATE_OLDEST.value == "truncate_oldest"
        assert TruncationStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert TruncationStrategy.IMPORTANCE_AWARE.value == "importance_aware"
        assert TruncationStrategy.SUMMARY_COMPRESSION.value == "summary_compression"

    def test_from_string(self):
        assert TruncationStrategy("truncate_oldest") == TruncationStrategy.TRUNCATE_OLDEST
        assert TruncationStrategy("importance_aware") == TruncationStrategy.IMPORTANCE_AWARE

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            TruncationStrategy("nonexistent")


# ── TruncationResult ──


class TestTruncationResult:
    def test_fields(self):
        result = TruncationResult(
            messages=[{"role": "user", "content": "hi"}],
            original_token_count=100,
            truncated_token_count=50,
            tokens_saved=50,
            strategy="truncate_oldest",
        )
        assert result.original_token_count == 100
        assert result.tokens_saved == 50
        assert result.strategy == "truncate_oldest"
        assert result.messages_removed == 0


# ── ContextWindowManager — basic ──


class TestContextWindowManagerBasic:
    def test_no_truncation_needed(self):
        mgr = ContextWindowManager()
        msgs = [_user_msg("Hi", token_estimate=10)]
        result = mgr.compute_truncation(msgs, max_tokens=1000)
        assert result.tokens_saved == 0
        assert len(result.messages) == len(msgs)

    def test_empty_messages(self):
        mgr = ContextWindowManager()
        result = mgr.compute_truncation([], max_tokens=100)
        assert result.tokens_saved == 0
        assert result.messages == []

    def test_default_strategy_is_importance_aware(self):
        mgr = ContextWindowManager()
        assert mgr._default_strategy == "importance_aware"

    def test_custom_token_counter(self):
        counter = lambda text: len(text.split())  # word-based
        mgr = ContextWindowManager(token_counter=counter)
        assert mgr.count_tokens("hello world") == 2

    def test_unknown_strategy_falls_back(self):
        mgr = ContextWindowManager(default_strategy="importance_aware")
        result = mgr.compute_truncation(
            [_user_msg("hi", 10)],
            max_tokens=1000,
            strategy="nonexistent",
        )
        # Should fall back to default without crashing
        assert result.strategy == "importance_aware"

    def test_fits_in_window(self):
        mgr = ContextWindowManager()
        msgs = [_user_msg("Short message", token_estimate=5)]
        assert mgr.fits_in_window(msgs, max_tokens=100) is True

    def test_does_not_fit_in_window(self):
        mgr = ContextWindowManager()
        msgs = _conversation(turns=10, tokens_per_msg=100)
        assert mgr.fits_in_window(msgs, max_tokens=10) is False


# ── truncate_oldest ──


class TestTruncateOldest:
    def test_removes_oldest_non_system(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = [_system_msg(50), _user_msg("Q1", 100), _user_msg("Q2", 100), _user_msg("Q3", 100)]
        # Budget = system(50+4) + one user(100+4) = 158
        result = mgr.compute_truncation(msgs, max_tokens=160)
        # Should keep system + latest user, drop Q1 and Q2
        assert result.tokens_saved > 0
        assert result.messages[0]["role"] == "system"

    def test_preserves_system_prompt(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = [_system_msg(100)]
        result = mgr.compute_truncation(msgs, max_tokens=200)
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "system"

    def test_all_messages_exceed_budget(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = [_user_msg("Huge", 500)]
        result = mgr.compute_truncation(msgs, max_tokens=10)
        # Should still return something (at minimum, the system messages)
        assert isinstance(result.messages, list)


# ── sliding_window ──


class TestSlidingWindow:
    def test_keeps_recent_messages(self):
        mgr = ContextWindowManager(default_strategy="sliding_window")
        msgs = [
            _system_msg(20),
            _user_msg("Old Q1", 100),
            _assistant_msg("Old A1", 100),
            _user_msg("New Q", 10),
            _assistant_msg("New A", 10),
        ]
        # Budget allows system + last 2 messages
        result = mgr.compute_truncation(msgs, max_tokens=100)
        assert result.tokens_saved > 0
        roles = [m["role"] for m in result.messages]
        assert "system" in roles
        # Newest messages should be present
        assert result.messages[-1]["role"] == "assistant"

    def test_preserves_system_at_start(self):
        mgr = ContextWindowManager(default_strategy="sliding_window")
        msgs = [_system_msg(20), _user_msg("Q", 10)]
        result = mgr.compute_truncation(msgs, max_tokens=100)
        assert result.messages[0]["role"] == "system"

    def test_empty_non_system(self):
        mgr = ContextWindowManager(default_strategy="sliding_window")
        msgs = [_system_msg(20)]
        result = mgr.compute_truncation(msgs, max_tokens=100)
        assert len(result.messages) == 1


# ── importance_aware ──


class TestImportanceAware:
    def test_keeps_system_and_recent(self):
        mgr = ContextWindowManager(
            default_strategy="importance_aware",
            min_recent_turns=1,
        )
        msgs = [
            _system_msg(20),
            _user_msg("Q1", 80),
            _assistant_msg("A1", 80),
            _user_msg("Q2", 10),
            _assistant_msg("A2", 10),
        ]
        result = mgr.compute_truncation(msgs, max_tokens=100)
        roles = [m["role"] for m in result.messages]
        assert "system" in roles
        # Recent turn should be preserved
        assert result.messages[-1]["role"] == "assistant"

    def test_drops_long_middle_messages(self):
        mgr = ContextWindowManager(
            default_strategy="importance_aware",
            min_recent_turns=1,
            importance_threshold=0.8,
        )
        msgs = [
            _system_msg(10),
            _user_msg("short", 5),
            _assistant_msg("long answer", 200),
            _user_msg("short2", 5),
            _assistant_msg("short answer", 5),
        ]
        result = mgr.compute_truncation(msgs, max_tokens=80)
        # Long middle message should likely be dropped due to budget
        assert result.tokens_saved > 0

    def test_no_truncation_if_fits(self):
        mgr = ContextWindowManager(default_strategy="importance_aware")
        msgs = [_system_msg(10), _user_msg("hi", 5)]
        result = mgr.compute_truncation(msgs, max_tokens=1000)
        assert result.tokens_saved == 0
        assert len(result.messages) == 2


# ── summary_compression ──


class TestSummaryCompression:
    def test_replaces_old_with_summary(self):
        mgr = ContextWindowManager(default_strategy="summary_compression")
        # Use small messages so the summary fits within budget
        msgs = [
            _system_msg(5),          # ~9 tokens
            _user_msg("Q1", 20),     # ~24 tokens
            _assistant_msg("A1", 20), # ~24 tokens
            _user_msg("Q2", 20),     # ~24 tokens
            _assistant_msg("A2", 20), # ~24 tokens
            _user_msg("Q3", 3),      # ~7 tokens
            _assistant_msg("A3", 3),  # ~7 tokens
        ]
        # Budget forces summary of old messages: ~9 + summary + 7 + 7 <= 80
        result = mgr.compute_truncation(msgs, max_tokens=80)
        assert result.tokens_saved > 0
        # Should have a summary message
        summary_msgs = [m for m in result.messages if "summary" in m.get("content", "")]
        assert len(summary_msgs) >= 1

    def test_keeps_recent_turns_intact(self):
        mgr = ContextWindowManager(default_strategy="summary_compression")
        msgs = [
            _system_msg(10),
            _user_msg("Q1", 50),
            _assistant_msg("A1", 50),
            _user_msg("Q2", 5),
            _assistant_msg("A2", 5),
        ]
        result = mgr.compute_truncation(msgs, max_tokens=200)
        # Last messages should be intact
        assert result.messages[-1]["role"] == "assistant"

    def test_no_summary_if_all_recent(self):
        mgr = ContextWindowManager(default_strategy="summary_compression")
        msgs = [_system_msg(10), _user_msg("Q", 5)]
        result = mgr.compute_truncation(msgs, max_tokens=1000)
        # No summary needed if everything fits
        assert result.tokens_saved == 0


# ── get_strategy_for_length ──


class TestGetStrategyForLength:
    def test_small_overflow_truncate_oldest(self):
        mgr = ContextWindowManager()
        strat = mgr.get_strategy_for_length(
            message_count=5, estimated_tokens=1200, max_tokens=1000,
        )
        assert strat == "truncate_oldest"

    def test_medium_overflow_importance_aware(self):
        mgr = ContextWindowManager()
        strat = mgr.get_strategy_for_length(
            message_count=10, estimated_tokens=2000, max_tokens=1000,
        )
        assert strat == "importance_aware"

    def test_large_overflow_summary_compression(self):
        mgr = ContextWindowManager()
        strat = mgr.get_strategy_for_length(
            message_count=50, estimated_tokens=5000, max_tokens=1000,
        )
        assert strat == "summary_compression"


# ── stats ──


class TestContextWindowStats:
    def test_initial_stats(self):
        mgr = ContextWindowManager()
        stats = mgr.get_stats()
        assert stats["truncations_applied"] == 0
        assert stats["total_tokens_saved"] == 0
        for strat in TruncationStrategy:
            assert stats["strategy_usage"][strat.value] == 0

    def test_stats_after_truncation(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = _conversation(turns=10, tokens_per_msg=100)
        mgr.compute_truncation(msgs, max_tokens=200)
        stats = mgr.get_stats()
        assert stats["truncations_applied"] == 1
        assert stats["total_tokens_saved"] > 0
        assert stats["strategy_usage"]["truncate_oldest"] == 1

    def test_stats_multiple_strategies(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = _conversation(turns=10, tokens_per_msg=100)
        mgr.compute_truncation(msgs, max_tokens=200, strategy="truncate_oldest")
        mgr.compute_truncation(msgs, max_tokens=200, strategy="importance_aware")
        mgr.compute_truncation(msgs, max_tokens=200, strategy="sliding_window")
        stats = mgr.get_stats()
        assert stats["truncations_applied"] == 3
        assert stats["strategy_usage"]["truncate_oldest"] == 1
        assert stats["strategy_usage"]["importance_aware"] == 1
        assert stats["strategy_usage"]["sliding_window"] == 1

    def test_stats_not_counted_when_no_truncation(self):
        mgr = ContextWindowManager()
        msgs = [_user_msg("short", 5)]
        mgr.compute_truncation(msgs, max_tokens=10000)
        stats = mgr.get_stats()
        assert stats["truncations_applied"] == 0
        assert stats["total_tokens_saved"] == 0


# ── message handling edge cases ──


class TestEdgeCases:
    def test_multimodal_content(self):
        mgr = ContextWindowManager()
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ]}
        ]
        result = mgr.compute_truncation(msgs, max_tokens=1000)
        assert result.tokens_saved == 0  # fits

    def test_empty_content(self):
        mgr = ContextWindowManager()
        msgs = [{"role": "user", "content": ""}]
        result = mgr.compute_truncation(msgs, max_tokens=1000)
        assert result.tokens_saved == 0

    def test_messages_not_mutated(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = [_system_msg(50), _user_msg("Q1", 100), _user_msg("Q2", 100)]
        original_len = len(msgs)
        mgr.compute_truncation(msgs, max_tokens=100)
        assert len(msgs) == original_len

    def test_count_messages_tokens(self):
        mgr = ContextWindowManager()
        msgs = [_user_msg("Hello world", token_estimate=10)]
        count = mgr.count_messages_tokens(msgs)
        assert count > 0  # includes role overhead

    def test_only_system_messages(self):
        mgr = ContextWindowManager(default_strategy="truncate_oldest")
        msgs = [_system_msg(10)]
        result = mgr.compute_truncation(msgs, max_tokens=50)
        assert len(result.messages) == 1
