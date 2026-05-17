from __future__ import annotations
"""Reasoning model thinking budget processor.

Controls thinking token consumption for models that use <think/> tags
(DeepSeek-R1, Qwen3, etc.). Limits the number of tokens spent on
reasoning to prevent runaway thinking that wastes compute and latency.

oMLX pattern: thinking budget is enforced by injecting a stop sequence
when the thinking token count exceeds the budget, forcing the model to
exit the reasoning state.

Also provides:
- Think-tag auto-detection: determines if a prompt enables reasoning mode
- Close pattern resolution: handles model-specific whitespace around </think/>
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

THINK_START = "<think/>"
THINK_END = "</think/>"


@dataclass
class ThinkingBudgetConfig:
    """Configuration for thinking budget enforcement."""
    max_thinking_tokens: int = 8192
    min_thinking_tokens: int = 0
    enabled: bool = True


class ThinkingBudgetProcessor:
    """Enforces thinking token budgets during generation.

    Tracks tokens generated inside <think/>...</think/> blocks and
    forces exit when the budget is exceeded.

    Integration with mlx-lm's SequenceStateMachine:
    - The state machine already tracks 'normal' vs 'reasoning' states
    - We observe the current_state from each response
    - When in 'reasoning' state, we count tokens toward the budget
    - When budget exceeded, we inject the think-end tokens to force
      transition back to 'normal' state
    """

    def __init__(self, config: ThinkingBudgetConfig | None = None) -> None:
        self.config = config or ThinkingBudgetConfig()
        self._thinking_token_count: int = 0
        self._segment_start_count: int = 0
        self._in_thinking: bool = False
        self._budget_exceeded: bool = False

    @property
    def thinking_tokens_used(self) -> int:
        return self._thinking_token_count

    @property
    def budget_remaining(self) -> int:
        return max(0, self.config.max_thinking_tokens - self._thinking_token_count)

    @property
    def is_budget_exceeded(self) -> bool:
        return self._budget_exceeded

    def process_token(self, current_state: str) -> dict:
        """Process one token and check budget.

        Args:
            current_state: From SequenceStateMachine ('normal' or 'reasoning').

        Returns:
            Dict with:
            - 'force_stop': bool — should we stop thinking?
            - 'budget_exceeded': bool
            - 'thinking_tokens_used': int
            - 'budget_remaining': int
        """
        if not self.config.enabled:
            return {
                'force_stop': False,
                'budget_exceeded': False,
                'thinking_tokens_used': 0,
                'budget_remaining': self.config.max_thinking_tokens,
            }

        was_thinking = self._in_thinking
        self._in_thinking = current_state == 'reasoning'

        if self._in_thinking:
            # Track per-segment count for multi-segment reasoning awareness
            if not was_thinking:
                self._segment_start_count = self._thinking_token_count
            self._thinking_token_count += 1

            if self._thinking_token_count > self.config.max_thinking_tokens:
                self._budget_exceeded = True
                logger.debug(
                    f"Thinking budget exceeded: {self._thinking_token_count} > "
                    f"{self.config.max_thinking_tokens}"
                )
                return {
                    'force_stop': True,
                    'budget_exceeded': True,
                    'thinking_tokens_used': self._thinking_token_count,
                    'budget_remaining': 0,
                }

        return {
            'force_stop': False,
            'budget_exceeded': self._budget_exceeded,
            'thinking_tokens_used': self._thinking_token_count,
            'budget_remaining': self.budget_remaining,
        }

    def get_think_end_tokens(self, tokenizer) -> list[int] | None:
        """Get token IDs for the think-end tag to force exit reasoning.

        When budget is exceeded, inject these tokens so the model's
        SequenceStateMachine transitions from 'reasoning' to 'normal'.
        """
        if not self._budget_exceeded or tokenizer is None:
            return None

        try:
            return tokenizer.encode(THINK_END, add_special_tokens=False)
        except Exception:
            logger.debug("tokenizer encode for think-end tag failed", exc_info=True)
            return None

    def reset(self) -> None:
        """Reset for a new request."""
        self._thinking_token_count = 0
        self._segment_start_count = 0
        self._in_thinking = False
        self._budget_exceeded = False

    def get_stats(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "max_thinking_tokens": self.config.max_thinking_tokens,
            "thinking_tokens_used": self._thinking_token_count,
            "budget_remaining": self.budget_remaining,
            "budget_exceeded": self._budget_exceeded,
            "in_thinking": self._in_thinking,
        }


def parse_thinking_budget(params: dict) -> ThinkingBudgetConfig | None:
    """Parse thinking budget from OpenAI/Anthropic request params.

    OpenAI reasoning_effort: "low" | "medium" | "high"
    Maps to thinking token budgets:
    - low: 2048 tokens
    - medium: 8192 tokens (default)
    - high: 32768 tokens
    """
    budget_tokens = params.get("thinking_budget")
    reasoning_effort = params.get("reasoning_effort")

    if budget_tokens is not None:
        return ThinkingBudgetConfig(
            max_thinking_tokens=int(budget_tokens),
            enabled=True,
        )

    if reasoning_effort is not None:
        effort_map = {
            "low": 2048,
            "medium": 8192,
            "high": 32768,
        }
        tokens = effort_map.get(reasoning_effort, 8192)
        return ThinkingBudgetConfig(max_thinking_tokens=tokens, enabled=True)

    return None


# ── Think-Tag Auto-Detection (oMLX pattern) ──


def detect_needs_think_prefix(
    prompt_token_ids: list[int],
    tokenizer,
) -> bool:
    """Detect if prompt ends with an open <think/> tag (thinking enabled).

    Returns False for disabled-thinking patterns like <think/>\n</think/>
    where </think/> immediately follows <think/> in the prompt tail.

    oMLX pattern: checks last few tokens for the think_start token,
    then verifies the think_end token doesn't follow immediately.
    """
    think_start_id = _get_think_token_id(tokenizer, 'think_start_id')
    if think_start_id is None:
        try:
            think_start_id = tokenizer.convert_tokens_to_ids("<think/>")
            unk = getattr(tokenizer, 'unk_token_id', None)
            if think_start_id == unk:
                return False
        except (AttributeError, KeyError, TypeError):
            return False

    if not think_start_id or not prompt_token_ids:
        return False

    last_tokens = list(prompt_token_ids[-3:])
    if think_start_id not in last_tokens:
        return False

    # <think/> found. Check if </think/> follows it (disabled thinking).
    last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
    after_start = last_tokens[last_idx + 1:]

    if after_start:
        think_end_ids = _resolve_think_end_token_ids(tokenizer)
        if think_end_ids and think_end_ids[0] in after_start:
            return False

    return True


def resolve_think_close_pattern(
    tokenizer,
) -> tuple[list[int] | None, list[int] | None]:
    """Detect leading/trailing tokens around </think/> from the chat template.

    Different models use different patterns:
    - Qwen3/3.5, MiniMax: ``\\n</think/>\\n\\n``
    - DeepSeek V3.2, GLM-5: ``</think/>`` (no newlines)
    - GLM-4.6V: ``</think/>\\n``

    Returns (leading_token_ids, trailing_token_ids) or (None, None).

    oMLX pattern: extracts whitespace patterns from the chat template
    surrounding the think_end tag.
    """
    think_end_str = getattr(tokenizer, 'think_end', '</think/>')

    template_text = _get_chat_template_text(tokenizer)
    if not template_text:
        return None, None

    escaped = re.escape(think_end_str)
    match = re.search(
        r'(\\n|\\r|[\n\r])*' + escaped + r'((?:\\n|\\r|[\n\r])*)',
        template_text,
    )
    if not match:
        return None, None

    raw_leading = (match.group(0).split(think_end_str)[0]
                   .replace('\\n', '\n').replace('\\r', '\r'))
    raw_trailing = (match.group(0).split(think_end_str)[1]
                    .replace('\\n', '\n').replace('\\r', '\r'))

    leading_ids = None
    trailing_ids = None
    if raw_leading:
        try:
            ids = tokenizer.encode(raw_leading, add_special_tokens=False)
            if ids:
                leading_ids = list(ids)
        except Exception:
            logger.debug("tokenizer encode for leading tags failed", exc_info=True)
    if raw_trailing:
        try:
            ids = tokenizer.encode(raw_trailing, add_special_tokens=False)
            if ids:
                trailing_ids = list(ids)
        except Exception:
            logger.debug("tokenizer encode for trailing tags failed", exc_info=True)

    return leading_ids, trailing_ids


def _get_think_token_id(tokenizer, attr_name: str) -> int | None:
    """Get a think-related token ID from tokenizer attributes or encode."""
    # Try common attribute names
    for name in (attr_name, 'think_start_id', 'think_start_token_id'):
        val = getattr(tokenizer, name, None)
        if val is not None and isinstance(val, int):
            return val

    # Try special tokens map
    special = getattr(tokenizer, 'special_tokens_map', {}) or {}
    for key in ('think_start', 'think'):
        token = special.get(key)
        if token:
            try:
                return tokenizer.convert_tokens_to_ids(token)
            except Exception:
                logger.debug("convert_tokens_to_ids failed", exc_info=True)

    return None


def _resolve_think_end_token_ids(tokenizer) -> list[int] | None:
    """Get token IDs for the think-end tag."""
    for name in ('think_end_id', 'think_end_token_id'):
        val = getattr(tokenizer, name, None)
        if isinstance(val, int):
            return [val]
        elif isinstance(val, (list, tuple)):
            return list(val)

    try:
        return tokenizer.encode("</think/>", add_special_tokens=False)
    except Exception:
        logger.debug("tokenizer encode for </think/> failed", exc_info=True)
        return None


def _get_chat_template_text(tokenizer) -> str | None:
    """Extract the chat template text from tokenizer."""
    # Try jinja template attribute
    for attr in ('chat_template', 'default_chat_template'):
        tpl = getattr(tokenizer, attr, None)
        if tpl and isinstance(tpl, str):
            return tpl

    # Try tokenizer config
    if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'chat_template'):
        ct = tokenizer._tokenizer.chat_template
        if isinstance(ct, dict):
            return ct.get('chat_template')
        return ct

    return None
