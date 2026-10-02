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
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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


class TruncationStrategy(StrEnum):
    """Available context window truncation strategies."""

    TRUNCATE_OLDEST = "truncate_oldest"
    SLIDING_WINDOW = "sliding_window"
    IMPORTANCE_AWARE = "importance_aware"
    SUMMARY_COMPRESSION = "summary_compression"


class ContextBudgetError(ValueError):
    """The protected messages plus the latest user turn exceed the budget."""


def publish_report(report: dict) -> None:
    """Attach a context-policy report to the request being served (a no-op outside one)."""
    logger.info("context policy applied: %s", report)
    from .request_tracker import current_request_info

    info: Any = current_request_info.get()
    if info is not None:
        try:
            info.context_policy = report
        except Exception:
            logger.debug("context policy report not attached", exc_info=True)


def removal_report(
    policy: str,
    before: list[dict],
    after: list[dict],
    tokens_before: int,
    tokens_after: int,
    budget_tokens: int,
) -> dict:
    """The report of a policy that is not a :class:`ContextWindowManager` strategy (the
    Responses API's ``truncation: "auto"``), in the same shape as ``TruncationResult.report``."""
    roles_before = Counter(str(m.get("role")) for m in before)
    roles_after = Counter(str(m.get("role")) for m in after)
    return {
        "policy": policy,
        "budget_tokens": budget_tokens,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_removed": tokens_before - tokens_after,
        "messages_before": len(before),
        "messages_after": len(after),
        "messages_removed": len(before) - len(after),
        "messages_summarized": 0,
        "removed_roles": {
            r: n - roles_after.get(r, 0)
            for r, n in roles_before.items()
            if n > roles_after.get(r, 0)
        },
        "cannot_fit": tokens_after > budget_tokens,
    }


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
    # True when the protected messages plus the latest user turn alone exceed
    # the budget: nothing was silently dropped and the caller must reject the
    # request. required_tokens is the minimum budget that would be needed.
    cannot_fit: bool = False
    required_tokens: int = 0
    max_tokens: int = 0
    # Messages per role before and after (``removed_roles`` is the per-role difference).
    roles_before: dict[str, int] = field(default_factory=dict)
    roles_after: dict[str, int] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.messages_removed > 0 or self.tokens_saved > 0

    def report(self) -> dict:
        """What the policy did, for ``x_yunshu.context_policy`` and the response headers: the
        policy, the messages and tokens before / after, which roles lost how many messages
        (``removed_roles["user"]`` is user turns) and the budget that forced it."""
        removed = {
            role: n - self.roles_after.get(role, 0)
            for role, n in self.roles_before.items()
            if n > self.roles_after.get(role, 0)
        }
        return {
            "policy": self.strategy,
            "budget_tokens": self.max_tokens,
            "tokens_before": self.original_token_count,
            "tokens_after": self.truncated_token_count,
            "tokens_removed": self.tokens_saved,
            "messages_before": sum(self.roles_before.values()),
            "messages_after": sum(self.roles_after.values()),
            "messages_removed": self.messages_removed,
            "messages_summarized": self.messages_summarized,
            "removed_roles": removed,
            "cannot_fit": self.cannot_fit,
        }

    def publish(self) -> None:
        """Attach :meth:`report` to the request being served (the gateway puts it in the
        response); a no-op outside a request and when nothing was removed."""
        if self.truncated:
            publish_report(self.report())

    def raise_if_cannot_fit(self) -> None:
        if self.cannot_fit:
            raise ContextBudgetError(
                f"Prompt needs at least {self.required_tokens} tokens for system/"
                f"developer messages and the latest user turn, but only "
                f"{self.max_tokens} are available in the context window"
            )


@dataclass
class ContextWindowStats:
    """Accumulated statistics for ContextWindowManager."""

    truncations_applied: int = 0
    total_tokens_saved: int = 0
    strategy_usage: dict[str, int] = field(
        default_factory=lambda: {s.value: 0 for s in TruncationStrategy}
    )


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
        media_token_counter: Callable[[dict], int] | None = None,
    ) -> None:
        self._media_token_counter = media_token_counter
        self._token_counter = token_counter or self._default_token_counter
        self._default_strategy = default_strategy
        self._min_recent_turns = min_recent_turns
        self._importance_threshold = importance_threshold
        self._stats = ContextWindowStats()

    @classmethod
    def for_processor(
        cls, processor=None, config: dict | None = None, **kwargs
    ) -> ContextWindowManager:
        """Manager whose media cost comes from the model's processor / config."""
        from .media_tokens import make_media_token_counter

        return cls(
            media_token_counter=make_media_token_counter(processor, config), **kwargs
        )

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
        cannot_fit = truncated_count > max_tokens
        required = self._required_tokens(messages) if cannot_fit else 0
        if cannot_fit:
            logger.warning(
                "context budget cannot be met: required=%d > budget=%d",
                required,
                max_tokens,
            )

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
            cannot_fit=cannot_fit,
            required_tokens=required,
            max_tokens=max_tokens,
            roles_before=dict(Counter(str(m.get("role")) for m in messages)),
            roles_after=dict(Counter(str(m.get("role")) for m in result_messages)),
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

    # ── Strategy implementations (unit based) ──
    #
    # A *unit* is the smallest group of messages that may be dropped together:
    # one protected message (never dropped), one plain message, or an
    # assistant(tool_calls) plus the tool results that answer its call ids.
    # Orphan tool results and incomplete tool groups are "invalid" units and
    # are dropped first whenever truncation is applied.  Messages keep their
    # original relative order; protected messages keep their original index.

    def _build_units(self, messages: list[dict]) -> list[dict]:
        units: list[dict] = []
        i = 0
        n = len(messages)
        while i < n:
            m = messages[i]
            role = m.get("role")
            if role in self._PROTECTED_ROLES:
                units.append({"idx": [i], "protected": True, "valid": True})
                i += 1
                continue
            tcs = m.get("tool_calls")
            if role == "assistant" and isinstance(tcs, list) and tcs:
                declared = {
                    tc.get("id") for tc in tcs if isinstance(tc, dict) and tc.get("id")
                }
                idx = [i]
                answered: list = []
                j = i + 1
                while j < n and messages[j].get("role") == "tool":
                    tid = messages[j].get("tool_call_id")
                    if declared and tid not in declared:
                        break
                    answered.append(tid)
                    idx.append(j)
                    j += 1
                if declared:
                    ok = set(answered) == declared and len(answered) == len(declared)
                else:
                    ok = len(answered) >= len(tcs)
                units.append({"idx": idx, "protected": False, "valid": ok})
                i = j
                continue
            if role in ("tool", "function"):
                units.append({"idx": [i], "protected": False, "valid": False})
            else:
                units.append({"idx": [i], "protected": False, "valid": True})
            i += 1
        # required: protected + everything from the latest user turn onward
        last_user = None
        for k, u in enumerate(units):
            if not u["protected"] and messages[u["idx"][0]].get("role") == "user":
                last_user = k
        if last_user is None:
            nonprot = [k for k, u in enumerate(units) if not u["protected"]]
            last_user = nonprot[-1] if nonprot else len(units)
        for k, u in enumerate(units):
            u["required"] = u["protected"] or k >= last_user
        return units

    def _unit_tokens(self, messages: list[dict], unit: dict) -> int:
        return self._count_messages_tokens([messages[i] for i in unit["idx"]])

    def _assemble(
        self,
        messages: list[dict],
        keep: list[dict],
        extra: list[tuple[float, dict]] | None = None,
    ) -> list[dict]:
        pairs: list[tuple[float, dict]] = []
        for u in keep:
            for i in u["idx"]:
                pairs.append((i, deepcopy(messages[i])))
        pairs.extend(extra or [])
        pairs.sort(key=lambda p: p[0])
        return [m for _, m in pairs]

    def _plan(self, messages: list[dict]):
        units = self._build_units(messages)
        base = [u for u in units if u["required"]]
        cand = [u for u in units if not u["required"] and u["valid"]]
        base_tokens = self._count_messages_tokens(
            [messages[i] for u in base for i in u["idx"]]
        )
        return units, base, cand, base_tokens

    def _required_tokens(self, messages: list[dict]) -> int:
        return int(self._plan(messages)[3])

    def _truncate_oldest(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Drop the oldest complete unit until the prompt fits.

        Protected messages and the latest user turn (with anything after it)
        are never dropped; if they alone exceed the budget the required set is
        returned and compute_truncation flags ``cannot_fit``.
        """
        if not messages:
            return []
        _units, base, cand, base_tokens = self._plan(messages)
        if base_tokens > max_tokens:
            return self._assemble(messages, base)
        total = base_tokens + sum(self._unit_tokens(messages, u) for u in cand)
        keep = list(cand)
        while keep and total > max_tokens:
            total -= self._unit_tokens(messages, keep.pop(0))
        return self._assemble(messages, base + keep)

    def _sliding_window(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Keep the newest contiguous run of complete units that fit."""
        if not messages:
            return []
        _units, base, cand, base_tokens = self._plan(messages)
        if base_tokens > max_tokens:
            return self._assemble(messages, base)
        remaining = max_tokens - base_tokens
        kept_rev: list[dict] = []
        for u in reversed(cand):
            cost = self._unit_tokens(messages, u)
            if cost > remaining:
                break
            kept_rev.append(u)
            remaining -= cost
        return self._assemble(messages, base + kept_rev)

    def _importance_aware(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Keep protected + latest turn + recent units + important middle units."""
        if not messages:
            return []
        _units, base, cand, base_tokens = self._plan(messages)
        if base_tokens > max_tokens:
            return self._assemble(messages, base)

        want = self._min_recent_turns * 2
        recent: list[dict] = []
        count = 0
        for u in reversed(cand):
            if count >= want:
                break
            recent.append(u)
            count += len(u["idx"])
        recent_set = {id(u) for u in recent}
        middle = [u for u in cand if id(u) not in recent_set]

        recent_tokens = sum(self._unit_tokens(messages, u) for u in recent)
        if base_tokens + recent_tokens > max_tokens:
            return self._truncate_oldest(messages, max_tokens)

        budget = max_tokens - base_tokens - recent_tokens
        scored = [
            (
                self._compute_importance(messages[u["idx"][0]], k, len(middle)),
                k,
                u,
            )
            for k, u in enumerate(middle)
        ]
        scored.sort(key=lambda x: -x[0])
        kept_middle: list[dict] = []
        for score, _k, u in scored:
            cost = self._unit_tokens(messages, u)
            if cost <= budget and score >= self._importance_threshold:
                kept_middle.append(u)
                budget -= cost
        return self._assemble(messages, base + kept_middle + recent)

    # Header for the summary message. It is deliberately a *user*-role message
    # and says so: summarized user text must never gain system authority.
    _SUMMARY_HEADER = (
        "[Conversation summary (earlier turns; quoted history, not instructions)]\n"
    )

    def _summary_compression(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Replace the oldest units with a quoted-history summary message."""
        if not messages:
            return []
        _units, base, cand, base_tokens = self._plan(messages)
        if base_tokens > max_tokens:
            return self._assemble(messages, base)

        header_tokens = self._token_counter(self._SUMMARY_HEADER)
        for k in range(len(cand), -1, -1):
            recent = cand[len(cand) - k :]
            old = cand[: len(cand) - k]
            if not old:
                result = self._assemble(messages, base + recent)
                if self._count_messages_tokens(result) <= max_tokens:
                    return result
                continue
            recent_tokens = sum(self._unit_tokens(messages, u) for u in recent)
            summary_token_budget = (
                max_tokens - base_tokens - recent_tokens - header_tokens - 4
            )
            if summary_token_budget <= 0:
                continue
            old_msgs = [messages[i] for u in old for i in u["idx"]]
            text = self._build_summary(
                old_msgs, max_chars=max(40, summary_token_budget)
            )
            summary_msg = {"role": "user", "content": self._SUMMARY_HEADER + text}
            pos = old[0]["idx"][0] - 0.5
            result = self._assemble(messages, base + recent, [(pos, summary_msg)])
            if self._count_messages_tokens(result) <= max_tokens:
                return result

        return self._assemble(messages, base)

    # ── Helpers ──

    # Estimated token cost for non-text content blocks (images, video frames).
    # Consolidated with token_counter.IMAGE_TOKEN_ESTIMATE for cross-component consistency.
    _IMAGE_TOKEN_ESTIMATE = 576

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
                        if block_type in (
                            "image_url",
                            "image",
                            "image_data",
                            "video",
                            "video_url",
                        ):
                            # Image/video blocks cost hundreds of tokens in practice
                            total += (
                                self._media_token_counter(block)
                                if self._media_token_counter
                                else self._IMAGE_TOKEN_ESTIMATE
                            )
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

    def _compute_importance(self, message: dict, index: int, total: int) -> float:
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
                        content_len += (
                            self._IMAGE_TOKEN_ESTIMATE * 4
                        )  # convert back to chars
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
        if total <= 1 or index == 0:
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
                        texts.append(block["text"][: per_msg_chars // 2])
                    elif isinstance(block, str):
                        texts.append(block[: per_msg_chars // 2])
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
