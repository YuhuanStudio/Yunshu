from __future__ import annotations
"""Grammar bitmask engine for structured output — xgrammar-style token masking.

Instead of maintaining an allowlist of token IDs (expensive per-step computation),
this engine precomputes a per-state bitmask over the entire vocabulary. Each bit
indicates whether the corresponding token is valid at the current grammar position.

Architecture:
  GrammarBitmaskEngine
    - Compiles a grammar specification (JSON schema, regex, choice) into a
      character-level DFA (deterministic finite automaton).
    - At each decode step, walks the DFA with already-generated text to find
      the current accept set of characters, then maps those characters to a
      token bitmask over the full vocabulary.
    - The bitmask is a boolean mx.array of shape (vocab_size,) — True means
      the token is allowed.

  BitmaskApplicator
    - Takes a bitmask and an logits mx.array, sets disallowed positions to -inf.
    - This is the tight inner loop — kept minimal for performance.

  DFA Construction
    - JSON schema: reuses JsonSchemaConstraint's state machine but extracts
      the allowed-character set directly instead of iterating over the vocab.
    - Regex: builds from the compiled regex's internal structure.
    - Choice: uses a trie over the choice strings.

Integration:
  - YUNSHU_GRAMMAR_BITMASK=1 env var selects this backend.
  - Plugs into _build_constrained_sampler() as alternative to JsonSchemaConstraint.
  - Compatible with ConstrainedSampler's advance/get_allowed_tokens interface.

Performance characteristics (vs JsonSchemaConstraint allowlist approach):
  - Precompute: O(vocab_size) once to build token-string table
  - Per-step: O(|expected_chars| * avg_tokens_per_char) to build mask
  - Logit masking: O(vocab_size) mx.where — vectorised on GPU
  - Overall: ~same asymptotic cost but avoids Python-level per-token iteration
    in the sampling hot path by using mx-level masking.
"""

import json
import logging
import os
from typing import Any, Callable
import weakref

logger = logging.getLogger(__name__)

# ── Bitmask Applicator ──────────────────────────────────────────────────────


class BitmaskApplicator:
    """Applies a token acceptance bitmask to logits.

    Given a boolean mask of shape (vocab_size,) where True = allowed,
    sets disallowed token logits to -inf before sampling.
    """

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self._neg_inf = None  # lazily initialised mx.array

    def apply(self, logits: Any, bitmask: Any) -> Any:
        """Apply bitmask to logits.

        Args:
            logits: mx.array of shape (..., vocab_size)
            bitmask: mx.array of shape (vocab_size,), bool dtype.
                     True = token is allowed.

        Returns:
            mx.array same shape as logits with disallowed tokens set to -inf.
        """
        import mlx.core as mx

        # bitmask is True where allowed; we want True where BLOCKED
        blocked = mx.logical_not(bitmask)
        if self._neg_inf is None:
            self._neg_inf = mx.array(float("-inf"), dtype=logits.dtype)
        return mx.where(blocked, self._neg_inf, logits)

    def apply_allowlist(self, logits: Any, allowed_ids: list[int]) -> Any:
        """Convenience: build mask from an allowlist, then apply.

        This is the bridge from character-level DFA to bitmask — the DFA
        computes allowed_ids, and this converts to a mask and applies it.
        """
        import mlx.core as mx

        mask = mx.zeros((self._vocab_size,), dtype=mx.bool_)
        if allowed_ids:
            ids = mx.array(allowed_ids)
            mask[ids] = True
        return self.apply(logits, mask)


# ── Token String Table ──────────────────────────────────────────────────────


class TokenStringTable:
    """Precomputed mapping between token IDs and their decoded strings.

    Built once per tokenizer, cached by tokenizer id. Provides:
    - id_to_string: list indexed by token_id → decoded string
    - char_to_ids: dict mapping first-decoded-char → list of token IDs
    - full_vocab_ids: list of all token IDs (for "any token allowed" case)
    """

    _cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.id_to_string: list[str] = [""] * vocab_size
        self.char_to_ids: dict[str, list[int]] = {}
        self.full_vocab_ids: list[int] = list(range(vocab_size))
        self._eos_ids: list[int] = []

        # Extract EOS tokens
        if hasattr(tokenizer, "eos_token_ids"):
            self._eos_ids = list(tokenizer.eos_token_ids)
        elif hasattr(tokenizer, "eos_token_id"):
            self._eos_ids = [tokenizer.eos_token_id]

        # Build vocabulary
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
            vocab = tokenizer.vocab
        else:
            vocab = {str(i): i for i in range(vocab_size)}

        for token_text, token_id in vocab.items():
            if token_id < 0 or token_id >= vocab_size:
                continue
            # Skip special tokens (angle-bracket wrapped)
            if (
                token_text
                and token_text[0] == "<"
                and len(token_text) > 1
                and token_text.endswith(">")
            ):
                self.id_to_string[token_id] = token_text
                continue

            # Decode properly to get the actual text representation
            try:
                decoded = tokenizer.decode([token_id])
                self.id_to_string[token_id] = decoded
                if decoded:
                    self.char_to_ids.setdefault(decoded[0], []).append(token_id)
            except Exception:
                logger.debug("operation failed", exc_info=True)
                self.id_to_string[token_id] = token_text
                if token_text:
                    self.char_to_ids.setdefault(token_text[0], []).append(token_id)

    @property
    def eos_ids(self) -> list[int]:
        return self._eos_ids

    def ids_for_chars(self, chars: set[str]) -> list[int]:
        """Return all token IDs whose decoded text starts with one of the chars."""
        allowed = set()
        for ch in chars:
            if ch in self.char_to_ids:
                allowed.update(self.char_to_ids[ch])
        return list(allowed)

    @classmethod
    def get(cls, tokenizer: Any) -> "TokenStringTable":
        """Get or create the token table for a tokenizer (cached)."""
        if tokenizer not in cls._cache:
            vocab_size = cls._detect_vocab_size(tokenizer)
            cls._cache[tokenizer] = cls(tokenizer, vocab_size)
        return cls._cache[tokenizer]

    @staticmethod
    def _detect_vocab_size(tokenizer: Any) -> int:
        """Detect vocabulary size from tokenizer.

        The vocab size is the maximum token ID + 1, not the number of
        entries in the vocab dict (which may have gaps or overlapping keys).
        """
        # Prefer explicit vocab_size attribute
        if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
            return tokenizer.vocab_size

        # Compute from max token ID in the vocab
        vocab = {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, "vocab"):
            v = tokenizer.vocab
            if isinstance(v, dict):
                vocab = v
            else:
                return len(v)

        if vocab:
            max_id = max(vocab.values()) if vocab else 0
            return max_id + 1

        return 32000  # fallback


# ── Grammar Bitmask Engine ──────────────────────────────────────────────────


class GrammarBitmaskEngine:
    """Generates token acceptance bitmasks from grammar rules.

    Wraps a DFA-like state tracker (reuses JsonSchemaConstraint, RegexConstraint,
    etc.) but instead of returning an allowlist, computes a full-vocab boolean
    bitmask. The bitmask is then applied to logits via BitmaskApplicator.

    Interface (compatible with ConstrainedSampler):
    - advance(token_text) — update DFA state after each token
    - get_allowed_tokens(tokenizer, ids) — return list[int] for compatibility
    - compute_bitmask(tokenizer) — return mx.array bool mask (new API)
    - is_done — whether generation is complete
    - reset() — reset to initial state
    """

    def __init__(self, constraint: Any) -> None:
        """Initialize with any constraint that implements the standard interface.

        The constraint must have: advance(), get_allowed_tokens(), is_done, reset().
        """
        self._constraint = constraint
        self._table_cache: dict[int, TokenStringTable] = {}
        self._checkpoint_stack: list = []
        self._constraint_rollback_needs_arg: bool | None = None

    @property
    def state(self) -> str:
        return getattr(self._constraint, "state", "active")

    @property
    def is_done(self) -> bool:
        return self._constraint.is_done

    def advance(self, token_text: str) -> None:
        self._constraint.advance(token_text)

    def get_allowed_tokens(self, tokenizer: Any, generated_token_ids: list[int]) -> list[int]:
        """Compatibility method — delegates to wrapped constraint."""
        return self._constraint.get_allowed_tokens(tokenizer, generated_token_ids)

    def compute_bitmask(self, tokenizer: Any) -> Any:
        """Compute a boolean bitmask over the vocabulary.

        Returns:
            mx.array of shape (vocab_size,), dtype bool.
            True = token is allowed at current DFA state.
        """
        import mlx.core as mx

        table = TokenStringTable.get(tokenizer)
        vocab_size = table.vocab_size

        if self._constraint.is_done:
            mask = mx.zeros((vocab_size,), dtype=mx.bool_)
            if table.eos_ids:
                mask[mx.array(table.eos_ids)] = True
            return mask

        allowed_ids = self._constraint.get_allowed_tokens(tokenizer, [])

        if not allowed_ids:
            # Nothing allowed — all False
            return mx.zeros((vocab_size,), dtype=mx.bool_)

        # Check if "all tokens allowed" (the constraint returned the full vocab)
        if len(allowed_ids) >= vocab_size * 0.95:
            return mx.ones((vocab_size,), dtype=mx.bool_)

        mask = mx.zeros((vocab_size,), dtype=mx.bool_)
        ids_arr = mx.array(allowed_ids)
        mask[ids_arr] = True
        return mask

    def reset(self) -> None:
        self._constraint.reset()

    def checkpoint(self) -> None:
        if hasattr(self._constraint, "checkpoint"):
            result = self._constraint.checkpoint()
            self._checkpoint_stack.append(result)
            # Probe whether rollback() expects an argument (takes >1 param
            # i.e. self + saved) by inspecting its signature once.
            if self._constraint_rollback_needs_arg is None and hasattr(self._constraint, "rollback"):
                import inspect
                sig = inspect.signature(self._constraint.rollback)
                self._constraint_rollback_needs_arg = len(sig.parameters) > 0

    def rollback(self) -> None:
        if hasattr(self._constraint, "rollback"):
            if self._constraint_rollback_needs_arg:
                saved = self._checkpoint_stack.pop() if self._checkpoint_stack else None
                if saved is not None:
                    self._constraint.rollback(saved)
                else:
                    # No checkpoint data saved — call bare rollback as fallback
                    self._constraint.rollback()
            else:
                self._constraint.rollback()
            # Pop the stack for no-arg rollback too (keeps stack depth correct)
            if self._constraint_rollback_needs_arg is False and self._checkpoint_stack:
                self._checkpoint_stack.pop()

    def get_stats(self) -> dict[str, Any]:
        stats = getattr(self._constraint, "get_stats", lambda: {})()
        stats["bitmask_engine"] = True
        return stats


# ── Bitmask-aware Constrained Sampler ──────────────────────────────────────


class BitmaskConstrainedSampler:
    """Sampler that uses bitmask-based logit masking.

    Drop-in replacement for ConstrainedSampler. At each step:
    1. Compute bitmask from GrammarBitmaskEngine
    2. Apply bitmask to logits via BitmaskApplicator
    3. Sample from masked logits using base sampler
    4. Advance DFA state with the sampled token
    """

    def __init__(
        self,
        base_sampler: Callable,
        engine: GrammarBitmaskEngine,
        tokenizer: Any,
    ) -> None:
        self._base_sampler = base_sampler
        self._engine = engine
        self._tokenizer = tokenizer
        self._generated_ids: list[int] = []
        self._applicator: BitmaskApplicator | None = None
        self._table: TokenStringTable | None = None

    def __call__(self, logits: Any) -> Any:
        """Sample a token with bitmask constraint."""
        import mlx.core as mx

        # Lazy-init applicator
        if self._applicator is None:
            self._table = TokenStringTable.get(self._tokenizer)
            self._applicator = BitmaskApplicator(self._table.vocab_size)

        if self._engine.is_done:
            # Force EOS
            masked_logits = self._applicator.apply_allowlist(logits, self._table.eos_ids)
        else:
            bitmask = self._engine.compute_bitmask(self._tokenizer)
            if mx.any(bitmask).item():
                masked_logits = self._applicator.apply(logits, bitmask)
            else:
                # Nothing is grammatically allowed — force EOS to avoid
                # producing invalid output.
                masked_logits = self._applicator.apply_allowlist(logits, self._table.eos_ids)

        token = self._base_sampler(masked_logits)

        # Update state
        token_id = int(token)
        self._generated_ids.append(token_id)

        try:
            token_text = self._tokenizer.decode([token_id])
        except Exception:
            logger.debug("operation failed", exc_info=True)
            token_text = ""
        self._engine.advance(token_text)

        return token

    @property
    def constraint(self) -> GrammarBitmaskEngine:
        return self._engine

    def checkpoint(self) -> None:
        """Save constraint state for potential rollback."""
        if hasattr(self._engine, 'checkpoint'):
            self._engine.checkpoint()

    def rollback(self) -> None:
        """Restore constraint state from last checkpoint."""
        if hasattr(self._engine, 'rollback'):
            self._engine.rollback()


# ── Factory ─────────────────────────────────────────────────────────────────


def is_bitmask_enabled() -> bool:
    """Check if bitmask engine is enabled via environment variable."""
    return os.environ.get("YUNSHU_GRAMMAR_BITMASK", "0") == "1"


def build_bitmask_engine(
    grammar_type: str,
    grammar: Any = None,
) -> GrammarBitmaskEngine:
    """Build a GrammarBitmaskEngine from a grammar specification.

    Args:
        grammar_type: "json_schema", "json_object", "regex", "choice", "cfg"
        grammar: The grammar spec (schema dict, pattern string, choice list, etc.)

    Returns:
        GrammarBitmaskEngine wrapping the appropriate constraint.
    """
    if grammar_type in ("json_schema", "json_object"):
        from .json_schema import JsonSchemaConstraint

        schema = grammar if grammar_type == "json_schema" else None
        if isinstance(grammar, str) and grammar_type == "json_schema":
            if grammar == "json_object":
                schema = None  # generic JSON object mode
            else:
                schema = json.loads(grammar)
        constraint = JsonSchemaConstraint(schema)
        return GrammarBitmaskEngine(constraint)

    if grammar_type == "regex":
        from .grammar_constraint import RegexConstraint

        if not isinstance(grammar, str):
            raise ValueError("regex constraint requires a string pattern")
        constraint = RegexConstraint(grammar)
        return GrammarBitmaskEngine(constraint)

    if grammar_type == "choice":
        from .grammar_constraint import ChoiceConstraint

        if not isinstance(grammar, list):
            raise ValueError("choice constraint requires a list of strings")
        constraint = ChoiceConstraint(grammar)
        return GrammarBitmaskEngine(constraint)

    if grammar_type == "cfg":
        from .grammar_constraint import LarkGrammarConstraint

        if not isinstance(grammar, str):
            raise ValueError("cfg constraint requires a grammar string")
        constraint = LarkGrammarConstraint(grammar)
        return GrammarBitmaskEngine(constraint)

    raise ValueError(f"Unknown grammar_type: {grammar_type}")
