"""Speculative draft token verifier for single-request fast-path generation.

This module implements the core verification algorithm for speculative decoding
that actually accelerates token production. Unlike the scheduler's statistical
shadow (which tracks acceptance rates but produces only 1 token per step),
this verifier:

1. Feeds draft tokens through the target model forward pass (populates KV cache)
2. Compares model logits against draft tokens to find the acceptance boundary
3. Returns verified tokens to emit + a bonus token from the first rejection point
4. Trims KV cache to remove entries beyond the acceptance boundary

The algorithm follows vLLM/SGLang's verification:
  - Draft proposer generates K candidate tokens
  - Feed those K tokens + context through model forward -> K+1 sets of logits
  - Compare: if model's top-1 at position i matches draft[i], accept it
  - Return: all accepted tokens + 1 bonus token from the rejection point

Two verification modes:
  - verify(): Simpler mode for when the last token is already in cache.
    Compares logits[i] with draft[i] — note this is a shifted comparison
    (logits[0] predicts position AFTER d0, so it verifies d1).
    Best for n-gram proposals where d0 is typically from the pattern match.
  - verify_with_last_token(): Correct mlx-lm algorithm that re-includes the
    last accepted token in the forward pass, giving proper K comparisons + bonus.
    Uses cache rollback to avoid duplication.

Integration contract:
  - Does NOT modify mlx-lm source code
  - Uses the model's forward function directly to get logits
  - Trims KV cache in-place when partially accepted (requires trimmable cache)
  - Works with both greedy (argmax) and sampler-based verification
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Result from spec draft verification.

    Attributes:
        accepted_tokens: Token IDs that passed verification (prefix of drafts).
        accepted_count: Number of accepted tokens (len(accepted_tokens)).
        bonus_token: Token ID from the model at the first rejection point.
                     When all drafts are accepted, this comes from the bonus
                     position (one past last draft). None only when draft_ids
                     is empty.
        rejected_count: Number of draft tokens rejected.
        rejection_position: Index of first rejected draft, or None if all accepted.
        cache_trimmed: Number of KV cache entries trimmed.
        latency_us: Verification latency in microseconds.
        all_accepted: True if every draft token matched the model.
    """

    accepted_tokens: list[int]
    accepted_count: int
    bonus_token: Optional[int]
    rejected_count: int
    rejection_position: Optional[int]
    cache_trimmed: int
    latency_us: float
    all_accepted: bool


class SpecDraftVerifier:
    """Verifies speculative draft tokens against the target model's logits.

    Usage (verify_with_last_token — correct alignment):
        verifier = SpecDraftVerifier()
        result = verifier.verify_with_last_token(
            model=model,
            last_token_id=last_tok,
            draft_ids=[42, 17, 99],
            prompt_cache=kv_cache,
        )
        # result.accepted_tokens -> verified tokens to emit
        # result.bonus_token -> model's token from rejection point
        # KV cache is trimmed to match accepted count

    Usage (verify — simpler shifted comparison):
        result = verifier.verify(
            model=model,
            draft_ids=[42, 17, 99],
            prompt_cache=kv_cache,
        )
    """

    def __init__(self, track_stats: bool = True) -> None:
        """Initialize the verifier.

        Args:
            track_stats: If True, accumulate running statistics across calls.
        """
        self._track_stats = track_stats
        self._total_proposals = 0
        self._total_accepted = 0
        self._total_bonus = 0
        self._total_verifications = 0

    # ── Main API ──

    def verify(
        self,
        model,
        draft_ids: list[int],
        prompt_cache: list,
        sampler=None,
        logits_processors: list | None = None,
        generation_stream=None,
        token_history: list[int] | None = None,
    ) -> VerifyResult:
        """Verify K draft tokens against the target model.

        Feeds all K draft tokens through the model in one forward pass,
        then compares the model's preferred tokens against the drafts.
        Trims the KV cache to remove entries for rejected tokens.

        Alignment note: When the last accepted token is already in the KV cache
        and we feed [d0, d1, ..., dK-1]:
          - logits[0] predicts what comes after d0 -> compare with d1
          - logits[i] predicts what comes after d0..di -> compare with d[i+1]
          - logits[K-1] predicts what comes after all -> bonus
        This is a shifted comparison: d0 is never directly verified. This is
        acceptable for n-gram proposals where d0 comes from pattern matching.

        For proper K-way verification, use verify_with_last_token() instead.

        Args:
            model: The target MLX model (nn.Module).
            draft_ids: K draft token IDs to verify.
            prompt_cache: The KV cache (list of cache objects). Must be trimmable.
            sampler: Optional sampler function. If None, uses greedy argmax.
            logits_processors: Optional list of logits processor functions.
            generation_stream: Optional MLX stream for GPU work.
            token_history: Optional token IDs for logits processor context.

        Returns:
            VerifyResult with accepted tokens, bonus token, and cache state.
        """
        t0 = time.perf_counter()
        K = len(draft_ids)

        if K == 0:
            return VerifyResult(
                accepted_tokens=[],
                accepted_count=0,
                bonus_token=None,
                rejected_count=0,
                rejection_position=None,
                cache_trimmed=0,
                latency_us=(time.perf_counter() - t0) * 1e6,
                all_accepted=True,
            )

        # Step 1: Feed all K draft tokens through model (one forward pass).
        # KV cache gets K new entries.
        draft_arr = mx.array(draft_ids, dtype=mx.uint32).reshape(1, -1)

        with _maybe_stream(generation_stream):
            batch_logits = model(draft_arr, cache=prompt_cache)
            if hasattr(batch_logits, "logits"):
                batch_logits = batch_logits.logits

        # Normalize to [K, vocab_size]
        if batch_logits.ndim == 3:
            batch_logits = batch_logits.reshape(
                batch_logits.shape[-2], batch_logits.shape[-1]
            )

        # Step 2: Apply logits processors if provided
        if logits_processors and token_history is not None:
            for i in range(K):
                context_ids = token_history + draft_ids[:i]
                for processor in logits_processors:
                    batch_logits[i] = processor(
                        mx.array(context_ids), batch_logits[i : i + 1]
                    ).squeeze(0)

        # Step 3: Get model predictions (greedy or sampled)
        if sampler is not None:
            logprobs = batch_logits - mx.logsumexp(
                batch_logits, axis=-1, keepdims=True
            )
            model_picks = sampler(logprobs).tolist()
        else:
            model_picks = mx.argmax(batch_logits, axis=-1).tolist()

        # Step 4: Find acceptance boundary (consecutive prefix match)
        accepted_tokens, rejection_position = _find_acceptance_boundary(
            model_picks, draft_ids, K
        )

        accepted_count = len(accepted_tokens)
        all_accepted = accepted_count == K
        rejected_count = K - accepted_count

        # Step 5: Bonus token from rejection point or last position
        bonus_token = _compute_bonus_token(model_picks, rejection_position, K)

        # Step 6: Trim KV cache to remove rejected entries
        cache_trimmed = 0
        if rejected_count > 0 and prompt_cache is not None:
            cache_trimmed = _trim_cache(prompt_cache, rejected_count)

        elapsed = (time.perf_counter() - t0) * 1e6
        self._update_stats(K, accepted_count, bonus_token)

        return VerifyResult(
            accepted_tokens=accepted_tokens,
            accepted_count=accepted_count,
            bonus_token=bonus_token,
            rejected_count=rejected_count,
            rejection_position=rejection_position,
            cache_trimmed=cache_trimmed,
            latency_us=elapsed,
            all_accepted=all_accepted,
        )

    def verify_with_last_token(
        self,
        model,
        last_token_id: int,
        draft_ids: list[int],
        prompt_cache: list,
        sampler=None,
        logits_processors: list | None = None,
        generation_stream=None,
        token_history: list[int] | None = None,
    ) -> VerifyResult:
        """Verify K draft tokens using the correct mlx-lm verification algorithm.

        This matches mlx-lm's speculative_generate_step exactly:
        1. Roll back cache by 1 position (remove last_token's entry)
        2. Feed [last_token, d0, d1, ..., dK-1] to model
        3. logits[0] = prediction after last_token -> compare with d0
        4. logits[i] = prediction after last_token,d0..d[i-1] -> compare with d[i]
        5. logits[K] = prediction after all -> bonus
        6. Trim cache for rejected entries

        This gives exactly K correct comparisons + 1 bonus = K+1 positions.

        Args:
            model: The target MLX model.
            last_token_id: The last accepted/produced token (in cache).
            draft_ids: K draft token IDs to verify.
            prompt_cache: KV cache (must be trimmable for rollback).
            sampler: Optional sampler function.
            logits_processors: Optional logits processors.
            generation_stream: Optional MLX stream.
            token_history: Optional token history for logits processors.

        Returns:
            VerifyResult with correctly aligned verification.
        """
        t0 = time.perf_counter()
        K = len(draft_ids)

        if K == 0:
            return VerifyResult(
                accepted_tokens=[],
                accepted_count=0,
                bonus_token=None,
                rejected_count=0,
                rejection_position=None,
                cache_trimmed=0,
                latency_us=(time.perf_counter() - t0) * 1e6,
                all_accepted=True,
            )

        # Step 1: Roll back cache by 1 to re-include last_token in input.
        # After this, cache is as if last_token was never processed.
        _trim_cache(prompt_cache, 1)

        # Step 2: Feed [last_token, d0, ..., dK-1] to model.
        # This produces K+1 logits positions for K comparisons + bonus.
        all_input = [last_token_id] + draft_ids
        input_arr = mx.array(all_input, dtype=mx.uint32).reshape(1, -1)

        with _maybe_stream(generation_stream):
            batch_logits = model(input_arr, cache=prompt_cache)
            if hasattr(batch_logits, "logits"):
                batch_logits = batch_logits.logits

        # Normalize to [K+1, vocab_size]
        if batch_logits.ndim == 3:
            batch_logits = batch_logits.reshape(
                batch_logits.shape[-2], batch_logits.shape[-1]
            )

        # Step 3: Apply logits processors
        if logits_processors and token_history is not None:
            for i in range(K + 1):
                context_ids = token_history + all_input[:i]
                for processor in logits_processors:
                    batch_logits[i] = processor(
                        mx.array(context_ids), batch_logits[i : i + 1]
                    ).squeeze(0)

        # Step 4: Get model predictions
        if sampler is not None:
            logprobs = batch_logits - mx.logsumexp(
                batch_logits, axis=-1, keepdims=True
            )
            model_picks = sampler(logprobs).tolist()
        else:
            model_picks = mx.argmax(batch_logits, axis=-1).tolist()

        # Step 5: Verify K draft tokens against model picks[0..K-1]
        # logits[0] predicts after last_token -> compare with d0
        # logits[i] predicts after last_token,d0..d[i-1] -> compare with d[i]
        accepted_tokens, rejection_position = _find_acceptance_boundary(
            model_picks[:K], draft_ids, K
        )

        accepted_count = len(accepted_tokens)
        all_accepted = accepted_count == K
        rejected_count = K - accepted_count

        # Bonus token:
        # - If rejected at position i: bonus = model_picks[i] (model's pick at rejection)
        # - If all accepted: bonus = model_picks[K] (prediction after all tokens)
        if rejection_position is not None:
            bonus_token = model_picks[rejection_position]
        else:
            bonus_token = model_picks[K]

        # Step 6: Trim KV cache.
        # Cache now has K+1 new entries (last_token + K drafts).
        # Keep: last_token (always) + accepted drafts.
        # Trim: rejected_count drafts.
        cache_trimmed = 0
        if rejected_count > 0 and prompt_cache is not None:
            cache_trimmed = _trim_cache(prompt_cache, rejected_count)
        # Add 1 for the initial rollback trim
        cache_trimmed += 1

        elapsed = (time.perf_counter() - t0) * 1e6
        self._update_stats(K, accepted_count, bonus_token)

        return VerifyResult(
            accepted_tokens=accepted_tokens,
            accepted_count=accepted_count,
            bonus_token=bonus_token,
            rejected_count=rejected_count,
            rejection_position=rejection_position,
            cache_trimmed=cache_trimmed,
            latency_us=elapsed,
            all_accepted=all_accepted,
        )

    # ── Stats ──

    @property
    def stats(self) -> dict:
        """Running verification statistics."""
        return {
            "total_proposals": self._total_proposals,
            "total_accepted": self._total_accepted,
            "total_bonus": self._total_bonus,
            "total_verifications": self._total_verifications,
            "acceptance_rate": (
                self._total_accepted / self._total_proposals
                if self._total_proposals > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self._total_proposals = 0
        self._total_accepted = 0
        self._total_bonus = 0
        self._total_verifications = 0

    def _update_stats(self, K: int, accepted: int, bonus: int | None) -> None:
        if not self._track_stats:
            return
        self._total_proposals += K
        self._total_accepted += accepted
        self._total_bonus += 1 if bonus is not None else 0
        self._total_verifications += 1


# ── Module-level helpers ──


def _find_acceptance_boundary(
    model_picks: list[int], draft_ids: list[int], K: int
) -> tuple[list[int], int | None]:
    """Find consecutive prefix match between model picks and draft tokens.

    Returns (accepted_tokens, rejection_position).
    rejection_position is None if all tokens match.
    """
    accepted_tokens = []
    rejection_position = None
    for i in range(K):
        if model_picks[i] == draft_ids[i]:
            accepted_tokens.append(draft_ids[i])
        else:
            rejection_position = i
            break
    return accepted_tokens, rejection_position


def _compute_bonus_token(
    model_picks: list[int],
    rejection_position: int | None,
    K: int,
) -> int | None:
    """Compute the bonus token from the rejection point or last position.

    At the rejection point, the model's pick is a "free" correct token.
    When all drafts accepted, the last position's prediction is the bonus.
    """
    if K == 0:
        return None
    if rejection_position is not None:
        return model_picks[rejection_position]
    # All accepted: bonus is from the last model pick
    return model_picks[-1]


def _trim_cache(prompt_cache: list, num_tokens: int) -> int:
    """Trim KV cache entries.

    Uses mlx-lm's trim_prompt_cache if available, otherwise best-effort.

    Args:
        prompt_cache: KV cache (list of cache objects).
        num_tokens: Number of positions to trim.

    Returns:
        Number of positions actually trimmed.
    """
    if num_tokens <= 0 or not prompt_cache:
        return 0

    try:
        from mlx_lm.models.cache import trim_prompt_cache

        return trim_prompt_cache(prompt_cache, num_tokens)
    except ImportError:
        total_trimmed = 0
        for c in prompt_cache:
            if hasattr(c, "trim"):
                total_trimmed = c.trim(num_tokens)
        return total_trimmed


class _maybe_stream:
    """Context manager that uses a stream if provided, otherwise does nothing."""

    def __init__(self, stream):
        self._stream = stream
        self._ctx = None

    def __enter__(self):
        if self._stream is not None:
            self._ctx = mx.stream(self._stream)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._ctx is not None:
            self._ctx.__exit__(*args)
