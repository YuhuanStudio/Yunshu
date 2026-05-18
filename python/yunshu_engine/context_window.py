from __future__ import annotations
"""Yunshu ContextWindowManager — context window truncation strategies.

When conversations exceed the model's context window limit, applies
truncation strategies to fit within the token budget:

  1. truncate_oldest: Drop oldest messages until under budget.
  2. sliding_window: Keep only the last N tokens (rolling buffer).
  3. importance_aware: Keep system prompt + recent turns + important middle turns.
  4. summary_compression: Replace old turns with a summary placeholder.

Integration:
  EngineCore.add_request()
    → ContextWindowManager.compute_truncation(messages, max_tokens, strategy)
    → returns truncated messages that fit within the budget
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


def _truncate_first_message_group(messages: list[dict]) -> None:
    """Remove the first message (and related tool messages) from a message list.

    Modifies the list in-place. When the first message is an assistant message
    with tool_calls, also removes the following tool role messages to keep the
    sequence valid. When the first message is an orphaned tool result (no
    preceding assistant tool_calls), removes it too.

    This ensures truncation never splits a tool call/response pair.
    """
    if not messages:
        return

    idx = 0
    first_role = messages[idx].get("role")

    # If the first message is an orphaned tool result, remove it
    if first_role == "tool":
        messages.pop(idx)
        return

    # If the first message is an assistant with tool_calls, remove it + all following tool results
    if first_role == "assistant" and messages[idx].get("tool_calls"):
        messages.pop(idx)
        # Remove consecutive tool messages that are responses to this assistant's tool_calls
        while messages and messages[0].get("role") == "tool":
            messages.pop(0)
        return

    # Default: remove just the first message
    messages.pop(0)


class TruncationStrategy(str, Enum):
    """Available context window truncation strategies."""

    TRUNCATE_OLDEST = "truncate_oldest"
    SLIDING_WINDOW = "sliding_window"
    IMPORTANCE_AWARE = "importance_aware"
    SUMMARY_COMPRESSION = "summary_compression"


@dataclass
class TruncationResult:
    """Result of a truncation operation."""

    messages: list[dict]
    original_token_count: int
    truncated_token_count: int
    tokens_saved: int
    strategy: str
    messages_removed: int = 0
    messages_summarized: int = 0


@dataclass
class ContextWindowStats:
    """Accumulated statistics for ContextWindowManager."""

    truncations_applied: int = 0
    total_tokens_saved: int = 0
    strategy_usage: dict[str, int] = field(default_factory=lambda: {
        s.value: 0 for s in TruncationStrategy
    })


class ContextWindowManager:
    """Manages context window truncation for conversations exceeding model limits.

    Usage:
        mgr = ContextWindowManager(token_counter=my_token_counter)
        result = mgr.compute_truncation(
            messages=[{"role": "user", "content": "..."}],
            max_tokens=4096,
            strategy="importance_aware",
        )
        truncated = result.messages
    """

    # Role priorities for importance_aware strategy
    _ROLE_PRIORITY = {
        "system": 100,
        "developer": 95,
        "assistant": 50,
        "user": 40,
        "tool": 30,
        "function": 30,
    }

    def __init__(
        self,
        token_counter: Callable[[str], int] | None = None,
        default_strategy: str = "importance_aware",
        min_recent_turns: int = 2,
        importance_threshold: float = 0.5,
    ) -> None:
        self._token_counter = token_counter or self._default_token_counter
        self._default_strategy = default_strategy
        self._min_recent_turns = min_recent_turns
        self._importance_threshold = importance_threshold
        self._stats = ContextWindowStats()

    # ── Public API ──

    def compute_truncation(
        self,
        messages: list[dict],
        max_tokens: int,
        strategy: str | None = None,
    ) -> TruncationResult:
        """Truncate messages to fit within max_tokens.

        Args:
            messages: Chat messages [{"role": ..., "content": ...}, ...].
            max_tokens: Maximum token budget.
            strategy: Truncation strategy name. Uses default if None.

        Returns:
            TruncationResult with the truncated messages and stats.
        """
        strat_name = strategy or self._default_strategy
        try:
            strat = TruncationStrategy(strat_name)
        except ValueError:
            logger.warning(
                f"Unknown truncation strategy '{strat_name}', "
                f"falling back to {self._default_strategy}"
            )
            strat = TruncationStrategy(self._default_strategy)
            strat_name = strat.value

        original_count = self._count_messages_tokens(messages)

        # No truncation needed
        if original_count <= max_tokens:
            return TruncationResult(
                messages=deepcopy(messages),
                original_token_count=original_count,
                truncated_token_count=original_count,
                tokens_saved=0,
                strategy=strat_name,
            )

        # Apply strategy
        if strat == TruncationStrategy.TRUNCATE_OLDEST:
            result_messages = self._truncate_oldest(messages, max_tokens)
        elif strat == TruncationStrategy.SLIDING_WINDOW:
            result_messages = self._sliding_window(messages, max_tokens)
        elif strat == TruncationStrategy.IMPORTANCE_AWARE:
            result_messages = self._importance_aware(messages, max_tokens)
        elif strat == TruncationStrategy.SUMMARY_COMPRESSION:
            result_messages = self._summary_compression(messages, max_tokens)
        else:
            result_messages = deepcopy(messages)

        truncated_count = self._count_messages_tokens(result_messages)
        tokens_saved = original_count - truncated_count

        # Update stats
        self._stats.truncations_applied += 1
        self._stats.total_tokens_saved += tokens_saved
        self._stats.strategy_usage[strat_name] = (
            self._stats.strategy_usage.get(strat_name, 0) + 1
        )

        return TruncationResult(
            messages=result_messages,
            original_token_count=original_count,
            truncated_token_count=truncated_count,
            tokens_saved=tokens_saved,
            strategy=strat_name,
            messages_removed=len(messages) - len(result_messages),
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens for a text string using the configured counter."""
        return self._token_counter(text)

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens across all messages."""
        return self._count_messages_tokens(messages)

    def fits_in_window(self, messages: list[dict], max_tokens: int) -> bool:
        """Check if messages fit within the token budget."""
        return self._count_messages_tokens(messages) <= max_tokens

    def get_strategy_for_length(
        self,
        message_count: int,
        estimated_tokens: int,
        max_tokens: int,
    ) -> str:
        """Select the best truncation strategy based on conversation characteristics.

        Heuristics:
        - Short conversations: truncate_oldest (simple, fast)
        - Medium conversations: importance_aware (preserves key context)
        - Very long conversations: summary_compression (max compression)
        """
        overflow_ratio = estimated_tokens / max(1, max_tokens)
        if overflow_ratio < 1.5:
            return TruncationStrategy.TRUNCATE_OLDEST.value
        elif overflow_ratio < 3.0 or message_count < 20:
            return TruncationStrategy.IMPORTANCE_AWARE.value
        else:
            return TruncationStrategy.SUMMARY_COMPRESSION.value

    def get_stats(self) -> dict:
        """Return manager statistics."""
        return {
            "truncations_applied": self._stats.truncations_applied,
            "total_tokens_saved": self._stats.total_tokens_saved,
            "strategy_usage": dict(self._stats.strategy_usage),
        }

    # ── Strategy implementations ──

    def _truncate_oldest(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Drop oldest messages until under budget. Always keeps system prompt.

        Preserves tool call/response pairs: if an assistant message with
        tool_calls is at the truncation boundary, also removes the following
        tool role messages to keep the message sequence valid for chat templates.

        When even a single non-system message exceeds the budget, returns just
        the system messages (if any) rather than looping infinitely.
        """
        if not messages:
            return []

        result = deepcopy(messages)
        # Identify and protect system messages
        system_msgs = [m for m in result if m.get("role") == "system"]
        non_system = [m for m in result if m.get("role") != "system"]

        # Remove oldest non-system messages first
        prev_len = -1
        while non_system and self._count_messages_tokens(system_msgs + non_system) > max_tokens:
            # Guard against infinite loop: if the last iteration didn't remove
            # anything, the remaining message(s) are simply too large for the
            # budget.  Return just the system messages in that case.
            if len(non_system) == prev_len:
                return deepcopy(system_msgs) if system_msgs else []
            prev_len = len(non_system)
            _truncate_first_message_group(non_system)

        return system_msgs + non_system

    def _sliding_window(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Keep only the most recent messages that fit in the window.

        Always preserves system messages at the start.
        Ensures tool call/response pairs are kept together.
        """
        if not messages:
            return []

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Start from the most recent and work backwards
        window = []
        for msg in reversed(non_system):
            candidate = system_msgs + [msg] + window
            if self._count_messages_tokens(candidate) > max_tokens:
                break
            window.insert(0, msg)

        # Ensure the window doesn't start with orphaned tool results
        while window and window[0].get("role") == "tool":
            window.pop(0)

        return system_msgs + window

    def _importance_aware(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Keep system prompt + recent turns + important middle turns.

        Importance scoring based on:
        - Role (system > user > assistant)
        - Content length (longer = more context)
        - Recency (recent = more relevant)

        Tool call/response pairs are always kept together.
        """
        if not messages:
            return []

        # Always keep system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return deepcopy(system_msgs)

        # Always keep the most recent N turns (expand to include any
        # trailing tool messages that belong to the last tool call)
        recent_count = self._min_recent_turns * 2
        recent = non_system[-recent_count:]  # user+assistant pairs
        # Extend recent to include any tool call groups at the boundary
        while recent and recent[0].get("role") == "tool":
            # This tool message at the start of 'recent' is orphaned without
            # its assistant tool_calls message. Include one more message.
            recent_count += 1
            if recent_count > len(non_system):
                recent = list(non_system)
                break
            recent = non_system[-recent_count:]

        middle = non_system[:-len(recent)] if len(non_system) > len(recent) else []

        # Safety check: if even system + recent exceed the budget, fall back
        # to truncate_oldest which will trim the recent messages too.
        baseline = deepcopy(system_msgs + recent)
        if self._count_messages_tokens(baseline) > max_tokens:
            return self._truncate_oldest(messages, max_tokens)

        # Score middle messages by importance, treating tool call groups as units
        scored = []
        skip_next = 0
        for i, msg in enumerate(middle):
            if skip_next > 0:
                skip_next -= 1
                continue
            # Compute importance for this message
            score = self._compute_importance(msg, i, len(middle))
            # If this is an assistant with tool_calls, group it with following tool messages
            group_size = 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                j = i + 1
                while j < len(middle) and middle[j].get("role") == "tool":
                    j += 1
                group_size = j - i
                skip_next = group_size - 1
            scored.append((score, i, msg, group_size))

        # Sort by score descending, keep highest-scoring middle messages
        scored.sort(key=lambda x: -x[0])

        # Greedily add middle messages by importance until budget exceeded
        kept_middle_indices: set[int] = set()
        current = deepcopy(system_msgs + recent)
        budget_remaining = max_tokens - self._count_messages_tokens(current)

        for score, idx, msg, group_size in scored:
            if idx in kept_middle_indices:
                continue
            # Calculate tokens for the entire group (assistant + tool responses)
            group = middle[idx:idx + group_size]
            group_tokens = self._count_messages_tokens(group)
            if group_tokens <= budget_remaining and score >= self._importance_threshold:
                for gi in range(group_size):
                    kept_middle_indices.add(idx + gi)
                budget_remaining -= group_tokens

        # Reconstruct in original order
        kept_middle = [middle[i] for i in range(len(middle)) if i in kept_middle_indices]
        return deepcopy(system_msgs) + deepcopy(kept_middle) + deepcopy(recent)

    def _summary_compression(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Replace old messages with a summary placeholder.

        Keeps system messages and recent turns intact; replaces older
        conversation turns with a single summary message.
        """
        if not messages:
            return []

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return deepcopy(system_msgs)

        # Try different split points: keep more recent turns
        for recent_count in range(len(non_system), 0, -2):
            recent = non_system[-recent_count:]
            old = non_system[:-recent_count]

            if not old:
                result = deepcopy(system_msgs + recent)
                if self._count_messages_tokens(result) <= max_tokens:
                    return result
                continue

            # Budget for summary = max_tokens - system - recent
            system_tokens = self._count_messages_tokens(system_msgs)
            recent_tokens = self._count_messages_tokens(recent)
            summary_token_budget = max_tokens - system_tokens - recent_tokens - 4  # -4 for role overhead
            if summary_token_budget <= 0:
                continue

            # Convert token budget to character budget (~4 chars/token)
            summary_char_budget = max(40, summary_token_budget * 4 - 40)  # -40 for header text

            # Build summary message
            summary_text = self._build_summary(old, max_chars=summary_char_budget)
            summary_msg = {
                "role": "system",
                "content": f"[Conversation summary]\n{summary_text}",
            }

            result = deepcopy(system_msgs) + [summary_msg] + deepcopy(recent)
            if self._count_messages_tokens(result) <= max_tokens:
                return result

        # Last resort: only system + last message
        if system_msgs:
            return deepcopy(system_msgs[:1]) + deepcopy(non_system[-1:])
        return deepcopy(non_system[-1:])

    # ── Helpers ──

    def _count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens in a list of messages.

        Includes tool_calls in assistant messages and tool_call_id/name
        in tool role messages for accurate multi-turn token estimation.
        """
        import json as _json

        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._token_counter(content)
            elif isinstance(content, list):
                # Multimodal content blocks
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        total += self._token_counter(text)
                    elif isinstance(block, str):
                        total += self._token_counter(block)
            # Account for tool_calls in assistant messages
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    total += 4  # tool call overhead
                    func = tc.get("function", {}) if isinstance(tc, dict) else {}
                    total += self._token_counter(func.get("name", ""))
                    args = func.get("arguments", "")
                    if isinstance(args, dict):
                        args = _json.dumps(args)
                    total += self._token_counter(str(args))
            # Account for tool_call_id and name in tool role messages
            if msg.get("tool_call_id"):
                total += 4
            if msg.get("name"):
                total += self._token_counter(msg["name"])
            # Role overhead (~4 tokens per message for role markers)
            total += 4
        return total

    def _compute_importance(
        self, message: dict, index: int, total: int
    ) -> float:
        """Compute an importance score for a message.

        Factors:
        - Role priority (system > developer > user > assistant > tool)
        - Content length (longer messages carry more context)
        - Position (first/last messages often more important)
        """
        role = message.get("role", "user")
        content = message.get("content", "")
        content_len = len(content) if isinstance(content, str) else 0

        # Role component (0-1)
        role_score = self._ROLE_PRIORITY.get(role, 40) / 100.0

        # Length component (0-1): normalized by typical max content length
        length_score = min(content_len / 2000.0, 1.0)

        # Position component: first and last messages score higher
        if total <= 1:
            position_score = 1.0
        elif index == 0:
            position_score = 1.0
        elif index == total - 1:
            position_score = 0.9
        else:
            # Middle messages: slightly higher in the first half
            relative = index / total
            position_score = 0.5 + 0.2 * (1.0 - abs(relative - 0.3))

        # Weighted combination
        return 0.4 * role_score + 0.3 * length_score + 0.3 * position_score

    def _build_summary(self, messages: list[dict], max_chars: int = 400) -> str:
        """Build a text summary of old messages for summary_compression.

        Produces a compact summary by truncating each message to a short
        preview. The total length is capped at max_chars to keep the
        summary from dominating the context window.
        """
        # Target per-message preview length (characters)
        per_msg_chars = max(20, max_chars // max(len(messages), 1))
        parts = []
        running_len = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                display = content[:per_msg_chars]
                if len(content) > per_msg_chars:
                    display = display.rstrip() + "..."
            elif isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"][:per_msg_chars // 2])
                    elif isinstance(block, str):
                        texts.append(block[:per_msg_chars // 2])
                combined = " ".join(texts)
                display = combined[:per_msg_chars]
                if len(combined) > per_msg_chars:
                    display = display.rstrip() + "..."
            else:
                display = "..."

            line = f"{role.capitalize()}: {display}"
            if running_len + len(line) > max_chars:
                break
            parts.append(line)
            running_len += len(line)

        summary = "\n".join(parts)
        return summary[:max_chars]

    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Default token counter: approximate 1 token per 4 characters."""
        return max(1, len(text) // 4)
