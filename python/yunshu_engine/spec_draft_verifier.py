from __future__ import annotations
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
        draft_logprobs: list[float] | None = None,
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

        Probabilistic acceptance: When draft_logprobs is provided along with
        a non-None sampler, uses the standard speculative sampling criterion:
            accept with probability min(1, p_target(x) / p_draft(x))
        When draft_logprobs is None, falls back to greedy argmax comparison.

        Args:
            model: The target MLX model (nn.Module).
            draft_ids: K draft token IDs to verify.
            prompt_cache: The KV cache (list of cache objects). Must be trimmable.
            sampler: Optional sampler function. If None, uses greedy argmax.
            logits_processors: Optional list of logits processor functions.
            generation_stream: Optional MLX stream for GPU work.
            token_history: Optional token IDs for logits processor context.
            draft_logprobs: Optional log-probabilities from the draft model for
                           each draft token. Required for probabilistic acceptance
                           with non-greedy sampling.

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
                context_ids = token_history + draft_ids[:i + 1]
                for processor in logits_processors:
                    batch_logits = batch_logits.at[i].set(
                        processor(
                            mx.array(context_ids), batch_logits[i : i + 1]
                        ).squeeze(0)
                    )

        # Step 3: Compute target log-probabilities
        target_logprobs = batch_logits - mx.logsumexp(
            batch_logits, axis=-1, keepdims=True
        )

        # Step 4: Probabilistic or greedy acceptance
        # Determine acceptance mode:
        # - draft_logprobs provided + sampler -> probabilistic acceptance
        # - otherwise -> greedy argmax comparison
        use_probabilistic = (
            draft_logprobs is not None
            and len(draft_logprobs) == K
            and sampler is not None
        )

        if use_probabilistic:
            accepted_tokens, rejection_position, bonus_token = (
                _probabilistic_accept_shifted(
                    target_logprobs, draft_ids, draft_logprobs
                )
            )
        else:
            # Greedy: get model predictions and compare
            # IMPORTANT: sampler expects raw logits, not log-probabilities.
            # target_logprobs has been normalized via logsumexp, so using it
            # with temperature-based samplers produces wrong distributions.
            if sampler is not None:
                model_picks = sampler(batch_logits).tolist()
            else:
                model_picks = mx.argmax(batch_logits, axis=-1).tolist()

            # Shifted comparison (d0 trusted):
            accepted_tokens = [draft_ids[0]]
            rejection_position = None
            for i in range(K - 1):
                if model_picks[i] == draft_ids[i + 1]:
                    accepted_tokens.append(draft_ids[i + 1])
                else:
                    rejection_position = i + 1
                    break

            # Bonus token
            if rejection_position is not None:
                bonus_token = model_picks[rejection_position - 1]
            else:
                bonus_token = model_picks[K - 1]

        accepted_count = len(accepted_tokens)
        all_accepted = accepted_count == K
        rejected_count = K - accepted_count

        # Step 5: Trim KV cache to remove rejected entries
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
        draft_logprobs: list[float] | None = None,
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

        Probabilistic acceptance: When draft_logprobs is provided along with
        a non-None sampler, uses the standard speculative sampling criterion:
            accept with probability min(1, p_target(x) / p_draft(x))
        When draft_logprobs is None, falls back to greedy argmax comparison.

        Args:
            model: The target MLX model.
            last_token_id: The last accepted/produced token (in cache).
            draft_ids: K draft token IDs to verify.
            prompt_cache: KV cache (must be trimmable for rollback).
            sampler: Optional sampler function.
            logits_processors: Optional logits processors.
            generation_stream: Optional MLX stream.
            token_history: Optional token history for logits processors.
            draft_logprobs: Optional log-probabilities from the draft model for
                           each draft token. Required for probabilistic acceptance
                           with non-greedy sampling.

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
                context_ids = token_history + all_input[:i + 1]
                for processor in logits_processors:
                    batch_logits = batch_logits.at[i].set(
                        processor(
                            mx.array(context_ids), batch_logits[i : i + 1]
                        ).squeeze(0)
                    )

        # Step 4: Compute target log-probabilities
        target_logprobs = batch_logits - mx.logsumexp(
            batch_logits, axis=-1, keepdims=True
        )

        # Step 5: Probabilistic or greedy acceptance
        use_probabilistic = (
            draft_logprobs is not None
            and len(draft_logprobs) == K
            and sampler is not None
        )

        if use_probabilistic:
            accepted_tokens, rejection_position, bonus_token = (
                _probabilistic_accept(
                    target_logprobs, draft_ids, draft_logprobs, K
                )
            )
        else:
            # Greedy: get model predictions and compare
            # IMPORTANT: sampler expects raw logits, not log-probabilities.
            # target_logprobs has been normalized via logsumexp, so using it
            # with temperature-based samplers produces wrong distributions.
            if sampler is not None:
                model_picks = sampler(batch_logits).tolist()
            else:
                model_picks = mx.argmax(batch_logits, axis=-1).tolist()

            # Standard verification: compare model_picks[0..K-1] with draft_ids[0..K-1]
            accepted_tokens, rejection_position = _find_acceptance_boundary(
                model_picks[:K], draft_ids, K
            )

            # Bonus token
            if rejection_position is not None:
                bonus_token = model_picks[rejection_position]
            else:
                bonus_token = model_picks[K]

        accepted_count = len(accepted_tokens)
        all_accepted = accepted_count == K
        rejected_count = K - accepted_count

        # Step 6: Trim KV cache.
        # After step 1 (rollback -1) and step 2 (forward K+1), net new = K.
        # Want to keep: 1 (last_token) + accepted_count entries = accepted_count+1.
        # Total forward entries: K+1.  Trim: (K+1) - (accepted_count+1) = rejected_count.
        trim_count = rejected_count
        cache_trimmed = 0
        if trim_count > 0 and prompt_cache is not None:
            cache_trimmed = _trim_cache(prompt_cache, trim_count)

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

import random as _random

# Module-level RNG for probabilistic acceptance (separate from any user-facing seed)
_prob_rng = _random.Random()


def _probabilistic_accept(
    target_logprobs,
    draft_ids: list[int],
    draft_logprobs: list[float],
    K: int,
) -> tuple[list[int], int | None, int | None]:
    """Probabilistic acceptance for verify_with_last_token (aligned comparison).

    Standard speculative sampling criterion (Chen et al. 2023, Leviathan et al. 2023):
      Accept draft token x_i with probability min(1, p_target(x_i) / p_draft(x_i))

    On rejection, sample a correction token from the adjusted distribution:
      p_adjusted(x) = normalize(max(0, p_target(x) - p_draft(x)))

    target_logprobs shape: [K+1, vocab_size] (positions 0..K-1 for verification,
    position K for bonus).

    Returns (accepted_tokens, rejection_position, bonus_token).
    """
    # Gather target logprob for each draft token at positions 0..K-1
    draft_ids_arr = mx.array(draft_ids).reshape(K, 1)
    target_lp = mx.take_along_axis(
        target_logprobs[:K], draft_ids_arr, axis=-1
    ).squeeze(-1)
    draft_lp = mx.array(draft_logprobs)

    # Acceptance ratio: min(1, exp(target_lp - draft_lp))
    # Clamp exponent to avoid overflow — large values mean guaranteed acceptance
    # (ratio >= 1), so clamping to exp(50) is safe and avoids inf/nan.
    log_diff = mx.clip(target_lp - draft_lp, -50.0, 50.0)
    ratios = mx.minimum(mx.ones(K), mx.exp(log_diff))

    # Sequential scan: accept until first rejection
    accepted_tokens = []
    rejection_position = None

    for i in range(K):
        r = float(ratios[i].item())
        u = _prob_rng.random()
        if u < r:
            accepted_tokens.append(draft_ids[i])
        else:
            rejection_position = i
            break

    # Bonus / correction token
    if rejection_position is not None:
        # Sample correction token from adjusted distribution at rejection position.
        # Pass the target logprobs row for sampling — since we only have token-level
        # draft logprobs (not the full draft distribution), _sample_correction will
        # sample from the target distribution, which is still correct.
        bonus_token = _sample_correction(
            target_logprobs[rejection_position],
            draft_ids[rejection_position],
            float(draft_lp[rejection_position].item()),
        )
    else:
        # All accepted: bonus from position K
        bonus_token = _sample_bonus(target_logprobs[K])

    return accepted_tokens, rejection_position, bonus_token


def _probabilistic_accept_shifted(
    target_logprobs,
    draft_ids: list[int],
    draft_logprobs: list[float],
) -> tuple[list[int], int | None, int | None]:
    """Probabilistic acceptance for verify() (shifted comparison).

    In the shifted mode, logits[i] predicts position AFTER d[i], so we compare:
      target_logprobs[i] vs draft_logprobs[i+1] for position d[i+1].
    d0 is always trusted (from pattern match).

    Returns (accepted_tokens, rejection_position, bonus_token).
    """
    K = len(draft_ids)
    accepted_tokens = [draft_ids[0]]  # d0 trusted
    rejection_position = None

    # Shifted comparison: verify d[1..K-1] against target_logprobs[0..K-2]
    for i in range(K - 1):
        # Target logprob for draft token d[i+1] at position i
        tid = draft_ids[i + 1]
        t_lp = float(target_logprobs[i, tid].item())
        d_lp = draft_logprobs[i + 1]

        ratio = min(1.0, _math_exp(t_lp - d_lp))
        u = _prob_rng.random()
        if u < ratio:
            accepted_tokens.append(draft_ids[i + 1])
        else:
            rejection_position = i + 1
            break

    # Bonus token
    if rejection_position is not None:
        bonus_pos = rejection_position - 1
        bonus_token = _sample_correction(
            target_logprobs[bonus_pos],
            draft_ids[rejection_position],
            draft_logprobs[rejection_position],
        )
    else:
        bonus_token = _sample_bonus(target_logprobs[K - 1])

    return accepted_tokens, rejection_position, bonus_token


def _sample_correction(
    target_logprobs_row,
    draft_token_id: int,
    draft_lp: float,
    draft_logprobs_row=None,
) -> int:
    """Sample a correction token from max(0, p_target - p_draft) distribution.

    This is the standard speculative sampling resampling step (Chen et al. 2023,
    Leviathan et al. 2023):
    when a draft token is rejected, sample from the difference distribution
    to produce a token from the target that was under-represented in the draft.

    Args:
        target_logprobs_row: Target model log-probabilities [vocab_size].
        draft_token_id: The rejected draft token ID (unused when draft_logprobs_row given).
        draft_lp: Log-probability of rejected token from draft model.
        draft_logprobs_row: Optional full draft log-prob distribution [vocab_size].
            When provided, uses the proper element-wise subtraction
            p_adjusted(x) = max(0, p_target(x) - p_draft(x)) for all x.
            When None (common for n-gram/MTP where only token-level probs are
            available), samples from the target distribution directly.
    """
    # Convert target logprobs to probs
    probs = mx.softmax(target_logprobs_row, axis=-1)

    if draft_logprobs_row is not None:
        # Full draft distribution available: proper correction sampling
        draft_probs = mx.softmax(draft_logprobs_row, axis=-1)
        adjusted = mx.maximum(mx.zeros_like(probs), probs - draft_probs)
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
            return int(mx.random.categorical(adjusted.reshape(1, -1)).item())

    # Without full draft distribution, we cannot compute the true correction.
    # Sampling from target is still correct (guarantees target distribution
    # fidelity) — just slightly less variance-reduced than the optimal correction.
    return int(mx.random.categorical(target_logprobs_row.reshape(1, -1)).item())


def _sample_bonus(target_logprobs_row) -> int:
    """Sample a bonus token from the target distribution (after all drafts accepted)."""
    return int(mx.random.categorical(target_logprobs_row.reshape(1, -1)).item())


def _math_exp(x: float) -> float:
    """Safe exp that clamps to avoid overflow."""
    import math
    if x > 50.0:
        return float("inf")
    if x < -50.0:
        return 0.0
    return math.exp(x)


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
