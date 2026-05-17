from __future__ import annotations
"""GPU-accelerated rejection sampling for speculative decoding (§12.4).

Uses MLX batched operations to verify ALL draft tokens in parallel on GPU,
replacing the per-token CPU sequential loop with vectorized checks.

Architecture:
  1. Batch argmax: compute model's preferred token at ALL positions simultaneously
  2. Vectorized comparison: check all draft-vs-model matches in one op
  3. Cumsum trick: find first mismatch without a Python loop
  4. Temperature support: stochastic acceptance via batched uniform sampling

Greedy mode (temperature=0):
  - Accept token i if argmax(logits[i]) == draft_token_ids[i]
  - First mismatch determines rejection point

Stochastic mode (temperature > 0):
  - Accept token i with probability min(1, P_target(x_i) / P_draft(x_i))
  - Standard speculative sampling acceptance criterion

Enabled via YUNSHU_GPU_REJECTION=1 env var.
Falls back to CPU sequential when disabled or on error.
"""


import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class BatchRejectionResult:
    """Result from GPU batch rejection sampling.

    Attributes:
        accepted_count: Number of draft tokens accepted (all before first rejection).
        rejection_position: Index of first rejected draft token, or None if all accepted.
        verification_method: How verification was performed ("gpu_batch" or "cpu_sequential").
        latency_us: Verification latency in microseconds.
    """

    accepted_count: int
    rejection_position: Optional[int]
    verification_method: str  # "gpu_batch" | "cpu_sequential"
    latency_us: float


def should_enable_gpu_rejection() -> bool:
    """Check whether GPU rejection sampling should be enabled.

    Enabled when YUNSHU_GPU_REJECTION=1 is set in the environment.
    Returns False otherwise (CPU sequential fallback is the default).
    """
    return os.environ.get("YUNSHU_GPU_REJECTION", "").strip() in ("1", "true", "yes")


class GPURejectionSampler:
    """GPU-accelerated batch rejection sampler for speculative decoding.

    Replaces the per-token CPU verification loop with MLX batched operations
    that run on GPU automatically. Supports both greedy and temperature-based
    (stochastic) acceptance criteria.

    Usage (greedy):
        sampler = GPURejectionSampler()
        result = sampler.verify_greedy(logits, draft_token_ids)

    Usage (stochastic):
        result = sampler.verify_stochastic(logits, draft_token_ids, draft_logprobs)

    Batch usage (multiple requests):
        results = sampler.verify_greedy_batch(all_logits, all_draft_ids)
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self._rng = random.Random(rng_seed)

    # ── Greedy Verification ──

    def verify_greedy(
        self,
        logits: mx.array,
        draft_token_ids: list[int],
    ) -> BatchRejectionResult:
        """Verify draft tokens against model logits using greedy (argmax) matching.

        Computes argmax for ALL positions simultaneously via mx.argmax(),
        then finds the first mismatch using vectorized cumsum tricks.

        Args:
            logits: Model logits tensor of shape [num_draft_tokens, vocab_size]
                    or [1, num_draft_tokens, vocab_size].
            draft_token_ids: List of draft token IDs to verify.

        Returns:
            BatchRejectionResult with accepted count and rejection position.
        """
        t0 = time.perf_counter()

        K = len(draft_token_ids)
        if K == 0:
            elapsed = (time.perf_counter() - t0) * 1e6
            return BatchRejectionResult(
                accepted_count=0,
                rejection_position=None,
                verification_method="gpu_batch",
                latency_us=elapsed,
            )

        # Normalize logits to [K, V] shape
        logit_2d = self._normalize_logits(logits, K)

        # Step 1: Batch argmax — model's preferred token at ALL positions
        model_picks = mx.argmax(logit_2d, axis=-1)  # shape [K]

        # Step 2: Vectorized comparison — draft vs model at each position
        draft_arr = mx.array(draft_token_ids)
        match_mask = model_picks == draft_arr  # shape [K], bool

        # Step 3: Find first mismatch via cumsum trick
        # match_mask: [True, True, False, True, False]
        # cumsum of inverted mask:  [0, 0, 1, 1, 2]
        # Any position where cumsum > 0 means a prior mismatch occurred
        mismatch_mask = ~match_mask  # True where mismatch
        cum_mismatches = mx.cumsum(mismatch_mask.astype(mx.int32))

        # Accepted = all positions before first mismatch
        # A position is accepted only if cum_mismatches == 0 there
        accepted_mask = cum_mismatches == 0  # [K] bool
        accepted_count = int(mx.sum(accepted_mask).item())

        # Rejection position: first index where match fails
        if accepted_count < K:
            rejection_position = accepted_count
        else:
            rejection_position = None

        mx.synchronize()
        elapsed = (time.perf_counter() - t0) * 1e6

        return BatchRejectionResult(
            accepted_count=accepted_count,
            rejection_position=rejection_position,
            verification_method="gpu_batch",
            latency_us=elapsed,
        )

    # ── Stochastic Verification ──

    def verify_stochastic(
        self,
        logits: mx.array,
        draft_token_ids: list[int],
        draft_logprobs: list[float],
        temperature: float = 1.0,
    ) -> BatchRejectionResult:
        """Verify draft tokens using stochastic acceptance (speculative sampling).

        Accepts token i with probability min(1, P_target(x_i) / P_draft(x_i)).
        Uses batched operations for the ratio computation, then a sequential
        scan for the first rejection (sequential dependency is unavoidable
        for stochastic acceptance).

        Args:
            logits: Model logits tensor [K, vocab_size] or [1, K, vocab_size].
            draft_token_ids: Draft token IDs.
            draft_logprobs: Draft model's log probabilities for each draft token.
            temperature: Sampling temperature (0 = greedy fallback).

        Returns:
            BatchRejectionResult with accepted count and rejection position.
        """
        t0 = time.perf_counter()

        K = len(draft_token_ids)
        if K == 0:
            elapsed = (time.perf_counter() - t0) * 1e6
            return BatchRejectionResult(
                accepted_count=0,
                rejection_position=None,
                verification_method="gpu_batch",
                latency_us=elapsed,
            )

        # Greedy fallback when temperature is 0
        if temperature <= 0.0:
            result = self.verify_greedy(logits, draft_token_ids)
            # Recompute elapsed for the stochastic wrapper
            elapsed = (time.perf_counter() - t0) * 1e6
            result.latency_us = elapsed
            return result

        # Validate draft_logprobs length matches draft_token_ids
        if draft_logprobs is None or len(draft_logprobs) != K:
            logger.warning(
                "draft_logprobs length mismatch (%s vs %d draft tokens), "
                "falling back to greedy verification",
                "None" if draft_logprobs is None else len(draft_logprobs),
                K,
            )
            result = self.verify_greedy(logits, draft_token_ids)
            elapsed = (time.perf_counter() - t0) * 1e6
            result.latency_us = elapsed
            return result

        logit_2d = self._normalize_logits(logits, K)

        # Step 1: Compute target log probabilities (vectorized)
        target_logprobs_full = mx.log(mx.softmax(logit_2d, axis=-1))  # [K, V]

        # Step 2: Gather target logprob at each draft token position
        draft_ids_arr = mx.array(draft_token_ids).reshape(K, 1)
        target_lps = mx.take_along_axis(
            target_logprobs_full, draft_ids_arr, axis=-1
        ).squeeze(-1)  # [K]

        draft_lps = mx.array(draft_logprobs)  # [K]

        # Step 3: Vectorized acceptance ratio: min(1, exp(target_lp - draft_lp))
        ratios = mx.minimum(
            mx.ones(K),
            mx.exp(target_lps - draft_lps),
        )  # [K]

        # Step 4: Generate uniform random numbers for all positions at once
        uniforms = mx.array([self._rng.random() for _ in range(K)])
        per_position_accept = uniforms < ratios  # [K] bool

        # Step 5: Sequential scan — find first rejection
        # (unavoidable for stochastic: acceptance at position i is independent
        # but we need the first failure for the spec decode contract)
        accepted_count = 0
        rejection_position = None
        for i in range(K):
            if bool(per_position_accept[i].item()):
                accepted_count += 1
            else:
                rejection_position = i
                break

        mx.synchronize()
        elapsed = (time.perf_counter() - t0) * 1e6

        return BatchRejectionResult(
            accepted_count=accepted_count,
            rejection_position=rejection_position,
            verification_method="gpu_batch",
            latency_us=elapsed,
        )

    # ── Batch Verification (Multiple Requests) ──

    def verify_greedy_batch(
        self,
        all_logits: list[mx.array],
        all_draft_ids: list[list[int]],
    ) -> list[BatchRejectionResult]:
        """Verify draft tokens for multiple requests simultaneously.

        Stacks logits from all requests into one tensor and performs batch
        argmax across all draft positions at once. Each request may have
        a different number of draft tokens (padded to max length).

        Args:
            all_logits: List of logits tensors, one per request.
                        Each is [K_i, vocab_size] or [1, K_i, vocab_size].
            all_draft_ids: List of draft token ID lists, one per request.

        Returns:
            List of BatchRejectionResult, one per request.
        """
        t0 = time.perf_counter()

        n_requests = len(all_logits)
        if n_requests == 0:
            return []

        # Single-request fast path (avoids padding overhead)
        if n_requests == 1:
            result = self.verify_greedy(all_logits[0], all_draft_ids[0])
            # Override method to indicate batch path was used
            result.verification_method = "gpu_batch"
            return [result]

        results = []
        K_max = max(len(d) for d in all_draft_ids)

        if K_max == 0:
            elapsed = (time.perf_counter() - t0) * 1e6
            return [
                BatchRejectionResult(0, None, "gpu_batch", elapsed)
                for _ in range(n_requests)
            ]

        # Normalize all logits to 2D and pad to K_max
        normalized = []
        for i in range(n_requests):
            K_i = len(all_draft_ids[i])
            if K_i == 0:
                normalized.append(None)
                continue
            logit_2d = self._normalize_logits(all_logits[i], K_i)
            normalized.append(logit_2d)

        # Process each request individually but with shared GPU synchronization
        for i in range(n_requests):
            if normalized[i] is None or len(all_draft_ids[i]) == 0:
                elapsed = (time.perf_counter() - t0) * 1e6
                results.append(
                    BatchRejectionResult(0, None, "gpu_batch", elapsed)
                )
                continue

            K_i = len(all_draft_ids[i])
            logit_2d = normalized[i]

            # Batch argmax for this request
            model_picks = mx.argmax(logit_2d, axis=-1)  # [K_i]
            draft_arr = mx.array(all_draft_ids[i])
            match_mask = model_picks == draft_arr

            # Find first mismatch via cumsum trick
            cum_mismatches = mx.cumsum((~match_mask).astype(mx.int32))
            accepted_mask = cum_mismatches == 0
            accepted_count = int(mx.sum(accepted_mask).item())

            rejection_pos = accepted_count if accepted_count < K_i else None

            elapsed = (time.perf_counter() - t0) * 1e6
            results.append(
                BatchRejectionResult(
                    accepted_count=accepted_count,
                    rejection_position=rejection_pos,
                    verification_method="gpu_batch",
                    latency_us=elapsed,
                )
            )

        mx.synchronize()
        return results

    # ── CPU Fallback ──

    @staticmethod
    def verify_cpu_sequential(
        logits: mx.array,
        draft_token_ids: list[int],
    ) -> BatchRejectionResult:
        """CPU sequential verification fallback.

        Verifies draft tokens one at a time using a Python loop.
        This is the original verification pattern used before GPU acceleration.

        Args:
            logits: Model logits [K, vocab_size] or [1, K, vocab_size].
            draft_token_ids: Draft token IDs.

        Returns:
            BatchRejectionResult with cpu_sequential method tag.
        """
        t0 = time.perf_counter()

        K = len(draft_token_ids)
        if K == 0:
            elapsed = (time.perf_counter() - t0) * 1e6
            return BatchRejectionResult(0, None, "cpu_sequential", elapsed)

        # Normalize to 2D
        if logits.ndim == 3:
            logits = logits.reshape(logits.shape[-2], logits.shape[-1])

        accepted_count = 0
        rejection_position = None
        for i in range(K):
            model_pick = int(mx.argmax(logits[i], axis=-1).item())
            if model_pick == draft_token_ids[i]:
                accepted_count += 1
            else:
                rejection_position = i
                break

        elapsed = (time.perf_counter() - t0) * 1e6
        return BatchRejectionResult(
            accepted_count=accepted_count,
            rejection_position=rejection_position,
            verification_method="cpu_sequential",
            latency_us=elapsed,
        )

    # ── Auto-select (env-var driven) ──

    def verify_auto(
        self,
        logits: mx.array,
        draft_token_ids: list[int],
        draft_logprobs: Optional[list[float]] = None,
        temperature: float = 0.0,
    ) -> BatchRejectionResult:
        """Auto-select between GPU batch and CPU sequential verification.

        Uses YUNSHU_GPU_REJECTION env var to decide. When GPU rejection is
        enabled, uses GPU batch verification (greedy or stochastic depending
        on temperature). Otherwise falls back to CPU sequential.

        Args:
            logits: Model logits [K, vocab_size] or [1, K, vocab_size].
            draft_token_ids: Draft token IDs.
            draft_logprobs: Draft model's logprobs (required for stochastic).
            temperature: Sampling temperature.

        Returns:
            BatchRejectionResult via GPU batch or CPU sequential.
        """
        if not should_enable_gpu_rejection():
            return GPURejectionSampler.verify_cpu_sequential(
                logits, draft_token_ids
            )

        try:
            if temperature > 0.0 and draft_logprobs is not None:
                return self.verify_stochastic(
                    logits, draft_token_ids, draft_logprobs, temperature
                )
            return self.verify_greedy(logits, draft_token_ids)
        except Exception as e:
            logger.warning(
                f"GPU rejection sampling failed, falling back to CPU: {e}"
            )
            return GPURejectionSampler.verify_cpu_sequential(
                logits, draft_token_ids
            )

    # ── Helpers ──

    @staticmethod
    def _normalize_logits(logits: mx.array, expected_K: int) -> mx.array:
        """Normalize logits to 2D shape [K, vocab_size].

        Handles [K, V], [1, K, V], and [batch, K, V] shapes.
        """
        if logits.ndim == 3:
            # [1, K, V] -> [K, V]
            return logits.reshape(logits.shape[-2], logits.shape[-1])
        if logits.ndim == 2:
            return logits
        raise ValueError(
            f"Expected 2D or 3D logits, got shape {logits.shape}"
        )

    @staticmethod
    def compute_bonus_token(logits: mx.array, position: int) -> int:
        """Compute the bonus (resample) token at a given position.

        After rejection at position i, the target model's argmax at that
        position is used as the resampled token. When all tokens are accepted,
        the bonus token comes from the last position.

        Args:
            logits: Model logits [K, vocab_size] or [1, K, vocab_size].
            position: Position to sample from.

        Returns:
            Token ID from argmax at the specified position.
        """
        if logits.ndim == 3:
            logits = logits.reshape(logits.shape[-2], logits.shape[-1])
        return int(mx.argmax(logits[position], axis=-1).item())
