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

    # Roles protected from truncation (never dropped by any strategy)
    _PROTECTED_ROLES = {"system", "developer"}

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

        Preserves original message order. Protected roles (system/developer)
        are never removed, but retain their original positions.

        When even a single non-system message exceeds the budget, returns just
        the system messages (if any) rather than looping infinitely.
        """
        if not messages:
            return []

        # If ALL messages are system/developer and they exceed the budget,
        # there are no removable messages.  Return them as-is (better to
        # send overlength system prompts than an empty conversation) and
        # log a warning so the operator can adjust.
        if not any(m.get("role") not in self._PROTECTED_ROLES for m in messages):
            total = self._count_messages_tokens(messages)
            if total > max_tokens:
                logger.warning(
                    "All messages are system/developer (%d tokens) but "
                    "max_tokens=%d — returning without truncation",
                    total, max_tokens,
                )
            return deepcopy(messages)

        result = deepcopy(messages)
        # Indices of removable (non-protected) messages
        removable_indices = [
            i for i, m in enumerate(result)
            if m.get("role") not in self._PROTECTED_ROLES
        ]

        # Remove oldest removable messages first (by original index order)
        prev_count = -1
        while removable_indices and self._count_messages_tokens(result) > max_tokens:
            if len(removable_indices) == prev_count:
                # Remaining messages still over budget — strip all non-protected
                return [m for m in deepcopy(messages) if m.get("role") in self._PROTECTED_ROLES] or []
            prev_count = len(removable_indices)
            # Find contiguous group at start of removable_indices
            group = [removable_indices[0]]
            for j in range(1, len(removable_indices)):
                if removable_indices[j] == removable_indices[j - 1] + 1:
                    group.append(removable_indices[j])
                else:
                    break
            # Also include trailing tool messages after assistant tool_calls
            last_idx = group[-1]
            msg = result[last_idx]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                k = last_idx + 1
                while k < len(result) and result[k].get("role") == "tool":
                    if k in removable_indices:
                        group.append(k)
                    k += 1
            for idx in sorted(group, reverse=True):
                result.pop(idx)
            removable_indices = [
                i for i, m in enumerate(result)
                if m.get("role") not in self._PROTECTED_ROLES
            ]

        return result

    def _sliding_window(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Keep only the most recent messages that fit in the window.

        Always preserves system messages at the start.
        Ensures tool call/response pairs are kept together.
        """
        if not messages:
            return []

        system_msgs = [m for m in messages if m.get("role") in self._PROTECTED_ROLES]
        non_system = [m for m in messages if m.get("role") not in self._PROTECTED_ROLES]

        # Pre-compute system token cost to avoid recounting every iteration.
        system_token_cost = self._count_messages_tokens(system_msgs)

        # If system messages alone exceed the budget, we cannot fit anything.
        # Return only the system messages (truncating them would lose the prompt).
        if system_token_cost > max_tokens:
            logger.warning(
                "System messages (%d tokens) alone exceed max_tokens (%d); "
                "returning system messages without truncation",
                system_token_cost, max_tokens,
            )
            return deepcopy(system_msgs)

        remaining_budget = max_tokens - system_token_cost

        # Build window from most recent backwards, appending to a list (O(1))
        # and reversing at the end instead of insert(0, ...) which is O(n^2).
        window_rev: list[dict] = []
        current_cost = 0
        for msg in reversed(non_system):
            msg_cost = self._count_messages_tokens([msg])
            if current_cost + msg_cost > remaining_budget:
                break
            window_rev.append(msg)
            current_cost += msg_cost
        window = list(reversed(window_rev))

        # Ensure the window doesn't start with orphaned tool results
        while window and window[0].get("role") == "tool":
            window.pop(0)

        # Ensure the window doesn't end with an assistant message whose
        # tool_calls have no matching tool responses (they were truncated).
        # Such dangling tool_calls would cause chat template errors.
        #
        # We must be careful to only remove tool messages that are genuinely
        # orphaned (no preceding assistant with tool_calls that they could
        # belong to). Simply stripping all trailing tool messages would
        # discard valid tool responses belonging to an earlier assistant.
        while window and window[-1].get("role") == "assistant" and window[-1].get("tool_calls"):
            window.pop(-1)
            # After removing the trailing assistant, remove trailing tool
            # messages ONLY if they are orphaned (no preceding assistant
            # with tool_calls exists in the window to claim them).
            while window and window[-1].get("role") == "tool":
                # Walk backwards to find if there's an assistant(tool_calls)
                # that could own this tool message. If found, the tool is
                # valid and we must stop removing.
                has_owner = False
                for j in range(len(window) - 2, -1, -1):
                    prev_role = window[j].get("role")
                    if prev_role == "assistant" and window[j].get("tool_calls"):
                        has_owner = True
                        break
                    # Stop searching at any non-tool, non-assistant boundary
                    if prev_role not in ("tool", "assistant"):
                        break
                if not has_owner:
                    window.pop(-1)
                else:
                    break

        return deepcopy(system_msgs) + deepcopy(window)

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

        # Always keep system + developer messages
        system_msgs = [m for m in messages if m.get("role") in self._PROTECTED_ROLES]
        non_system = [m for m in messages if m.get("role") not in self._PROTECTED_ROLES]

        if not non_system:
            return deepcopy(system_msgs)

        # Always keep the most recent N turns (expand to include any
        # trailing tool messages that belong to the last tool call)
        recent_count = self._min_recent_turns * 2
        recent = non_system[-recent_count:]  # user+assistant pairs
        # Extend recent to include any tool call groups at the boundary.
        # We must handle two cases:
        #   (a) Leading orphaned tool messages (no preceding assistant tool_calls).
        #   (b) An assistant(tool_calls) at the boundary whose tool responses
        #       would be split — we need to extend until ALL its tool responses
        #       are included.
        while True:
            if not recent:
                break
            first_role = recent[0].get("role")
            # Case (a): orphaned tool result at the boundary
            if first_role == "tool":
                recent_count += 1
                if recent_count > len(non_system):
                    recent = list(non_system)
                    break
                recent = non_system[-recent_count:]
                continue
            # Case (b): assistant(tool_calls) whose tool responses may be split.
            # Check that all following tool messages in the original list are
            # included in recent.
            if first_role == "assistant" and recent[0].get("tool_calls"):
                # Find the index of recent[0] in non_system
                boundary_idx = len(non_system) - len(recent)
                # Count tool messages following this assistant in non_system
                expected_tools = 0
                k = boundary_idx + 1
                while k < len(non_system) and non_system[k].get("role") == "tool":
                    expected_tools += 1
                    k += 1
                # Count tool messages following the assistant in recent
                actual_tools = 0
                for m in recent[1:]:
                    if m.get("role") == "tool":
                        actual_tools += 1
                    else:
                        break
                if actual_tools < expected_tools:
                    # Tool group is split — extend to include all tool responses
                    recent_count += (expected_tools - actual_tools)
                    if recent_count > len(non_system):
                        recent = list(non_system)
                        break
                    recent = non_system[-recent_count:]
                    continue
            break

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

        system_msgs = [m for m in messages if m.get("role") in self._PROTECTED_ROLES]
        non_system = [m for m in messages if m.get("role") not in self._PROTECTED_ROLES]

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

    # Estimated token cost for non-text content blocks (images, video frames).
    # Real models use 256–1024 tokens per image; we use a conservative 128.
    _IMAGE_TOKEN_ESTIMATE = 128

    def _count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens in a list of messages.

        Includes tool_calls in assistant messages and tool_call_id/name
        in tool role messages for accurate multi-turn token estimation.
        Handles multimodal content (image/video blocks) and None content.
        """
        import json as _json

        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if content is None:
                # Message with null content — still has role overhead
                total += 0
            elif isinstance(content, str):
                total += self._token_counter(content)
            elif isinstance(content, list):
                # Multimodal content blocks
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type in ("image_url", "image", "video", "video_url"):
                            # Image/video blocks cost hundreds of tokens in practice
                            total += self._IMAGE_TOKEN_ESTIMATE
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
        if isinstance(content, str):
            content_len = len(content)
        elif isinstance(content, list):
            # Multimodal: sum text lengths + estimate for media blocks
            content_len = 0
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type in ("image_url", "image", "video", "video_url"):
                        content_len += self._IMAGE_TOKEN_ESTIMATE * 4  # convert back to chars
                    content_len += len(block.get("text", ""))
                elif isinstance(block, str):
                    content_len += len(block)
        elif content is None:
            content_len = 0
        else:
            content_len = 0

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
