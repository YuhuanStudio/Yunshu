from __future__ import annotations
"""Yunshu Speculative Decoding Engine — EAGLE-3 GPU-only path.

Implements speculative decoding where a small draft model proposes K tokens,
then the target model verifies them in a single forward pass. Accepted tokens
are kept; the first rejected token triggers resampling from the target's
distribution at that position.

Architecture (EAGLE-3 / oMLX pattern):
  1. Draft model: smaller/faster model proposes K tokens
  2. Target model: verifies all K tokens in one forward pass
  3. Acceptance: keep tokens up to first mismatch, resample from target
  4. Bonus token: target always generates one extra token after acceptance

For MoE models: also supports "Speculating Experts" where the draft
predicts which experts will be activated, reducing MoE routing overhead.

GPU-only path (Phase 4 baseline):
  - Draft model runs on same GPU as target
  - No ANE integration (that's the risky Δ-2 path)
  - Target: 6.5× baseline speedup with EAGLE-3 (proven in production)

Integration with Scheduler:
  - Scheduler inserts draft+target as paired requests
  - BatchGenerator handles both in same step
  - Output collector merges accepted tokens

Auto-detection (vLLM SpeculativeConfig pattern):
  - MTP: num_nextn_predict_layers / mtp_num_hidden_layers / n_predict / mtp_heads
  - EAGLE: eagle / eagle3 / draft_model_path with eagle in name
  - Medusa: model_type="medusa" / num_draft_tokens / num_speculative_tokens
  - MLPSpeculator: model_type="mlp_speculator"

References:
  - EAGLE-3: Speculative Sampling Requires Rethinking (ICML 2025)
  - Medusa: Simple LLM Inference Acceleration (ICML 2024)
  - vLLM SpeculativeConfig (config/speculative.py)
  - oMLX speculative decoding patterns
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# ── MTP model types that indicate multi-token prediction heads (vLLM pattern)
_MTP_MODEL_TYPES = frozenset({
    "deepseek_mtp",
    "mimo_mtp",
    "mimo_v2_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "exaone4_5_mtp",
    "qwen3_next_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
    "hy_v3_mtp",
})


# ── Spec Head Detection ──


@dataclass
class SpecHeadInfo:
    """Detected speculative decoding head information.

    Returned by detect_spec_heads() after scanning model config.
    """

    # Head type: "eagle", "eagle3", "medusa", "mtp", "mlp_speculator", "none"
    head_type: str = "none"

    # Number of prediction heads (e.g., Medusa has multiple heads)
    num_heads: int = 0

    # Number of draft tokens per step (K)
    draft_length: int = 0

    # Raw config fields used for detection (for debugging/logging)
    head_config: dict = field(default_factory=dict)


def detect_spec_heads(model_config: dict) -> SpecHeadInfo:
    """Detect speculative decoding heads from model config.

    Scans the model's config.json for known speculative decoding patterns.
    Follows vLLM's SpeculativeConfig detection logic but adapted for Yunshu's
    simpler config inspection (no model loading needed).

    Detection priority (first match wins):
      1. MTP: num_nextn_predict_layers, mtp_num_hidden_layers, n_predict,
              mtp_heads, or model_type matching _MTP_MODEL_TYPES
      2. EAGLE-3: eagle3 key or model_type="eagle3"
      3. EAGLE: eagle key or model_type="eagle" or draft_model_path with eagle
      4. Medusa: model_type="medusa" or num_draft_tokens/num_speculative_tokens
      5. MLPSpeculator: model_type="mlp_speculator"
      6. None: no speculative heads detected

    Args:
        model_config: Parsed config.json dict (typically from HuggingFace).

    Returns:
        SpecHeadInfo with detected head type, number of heads, and draft length.
    """
    if not isinstance(model_config, dict):
        return SpecHeadInfo()

    model_type = model_config.get("model_type", "").lower().replace("-", "_")
    architectures = model_config.get("architectures", [])

    # ── 1. MTP (Multi-Token Prediction) ──

    # DeepSeek-V3/R1 pattern: num_nextn_predict_layers
    n_nextn = model_config.get("num_nextn_predict_layers")
    # Qwen3.5 pattern: mtp_num_hidden_layers
    mtp_num_layers = model_config.get("mtp_num_hidden_layers")
    # General MTP pattern: n_predict (used by vLLM after hf_config_override)
    n_predict = model_config.get("n_predict")
    # Alternative: mtp_heads
    mtp_heads = model_config.get("mtp_heads")

    # Check model_type for known MTP types
    if model_type in _MTP_MODEL_TYPES or n_nextn or mtp_num_layers or mtp_heads:
        heads = n_nextn or mtp_num_layers or mtp_heads or 1
        draft_len = n_predict or heads
        return SpecHeadInfo(
            head_type="mtp",
            num_heads=heads,
            draft_length=draft_len,
            head_config={
                "num_nextn_predict_layers": n_nextn,
                "mtp_num_hidden_layers": mtp_num_layers,
                "n_predict": n_predict,
                "mtp_heads": mtp_heads,
                "model_type": model_type,
            },
        )

    # ── 2. EAGLE-3 ──

    eagle3_val = model_config.get("eagle3")
    if eagle3_val is not None or model_type == "eagle3":
        draft_len = 1
        if isinstance(eagle3_val, dict):
            draft_len = eagle3_val.get("num_speculative_tokens", 1)
        elif isinstance(eagle3_val, int):
            draft_len = eagle3_val
        num_lookahead = model_config.get("num_lookahead_tokens", draft_len)
        return SpecHeadInfo(
            head_type="eagle3",
            num_heads=1,
            draft_length=num_lookahead,
            head_config={
                "eagle3": eagle3_val,
                "num_lookahead_tokens": num_lookahead,
                "model_type": model_type,
            },
        )

    # ── 3. EAGLE ──

    eagle_val = model_config.get("eagle")
    draft_model_path = model_config.get("draft_model_path")

    # Check for eagle key, model_type, or draft_model_path containing "eagle"
    is_eagle = (
        eagle_val is not None
        or model_type == "eagle"
        or (draft_model_path is not None
            and "eagle" in str(draft_model_path).lower())
    )

    if is_eagle:
        draft_len = 1
        if isinstance(eagle_val, dict):
            draft_len = eagle_val.get("num_speculative_tokens", 1)
        num_lookahead = model_config.get("num_lookahead_tokens", draft_len)
        return SpecHeadInfo(
            head_type="eagle",
            num_heads=1,
            draft_length=num_lookahead,
            head_config={
                "eagle": eagle_val,
                "draft_model_path": draft_model_path,
                "num_lookahead_tokens": num_lookahead,
                "model_type": model_type,
            },
        )

    # ── 4. MLPSpeculator (check before Medusa — model_type is definitive) ──

    num_draft = model_config.get("num_draft_tokens")
    num_spec = model_config.get("num_speculative_tokens")

    if model_type == "mlp_speculator":
        draft_len = num_spec or model_config.get("num_predict", 5)
        heads = model_config.get("num_heads", 1)
        return SpecHeadInfo(
            head_type="mlp_speculator",
            num_heads=heads,
            draft_length=draft_len,
            head_config={
                "model_type": model_type,
                "num_speculative_tokens": num_spec,
                "num_predict": model_config.get("num_predict"),
            },
        )

    # ── 5. Medusa ──

    medusa_num_heads = model_config.get("medusa_num_heads")

    if model_type == "medusa" or num_draft is not None or num_spec is not None:
        heads = medusa_num_heads or 1
        draft_len = num_draft or num_spec or heads
        return SpecHeadInfo(
            head_type="medusa",
            num_heads=heads,
            draft_length=draft_len,
            head_config={
                "model_type": model_type,
                "num_draft_tokens": num_draft,
                "num_speculative_tokens": num_spec,
                "medusa_num_heads": medusa_num_heads,
            },
        )

    # ── 6. No speculative heads detected ──

    return SpecHeadInfo()


def auto_configure_speculative(model_config: dict) -> SpecDecodingConfig:
    """Auto-configure speculative decoding from model config.

    Calls detect_spec_heads() and creates an appropriate SpecDecodingConfig
    based on the detected head type. If no heads are found, returns a default
    config with speculative decoding effectively disabled (draft_length=0).

    Args:
        model_config: Parsed config.json dict (typically from HuggingFace).

    Returns:
        SpecDecodingConfig configured for the detected head type.
    """
    head_info = detect_spec_heads(model_config)

    if head_info.head_type == "none":
        # No speculative heads — disable speculative decoding
        return SpecDecodingConfig(
            draft_length=0,
            bonus_token=False,
        )

    # Determine draft length from head info
    draft_length = head_info.draft_length or head_info.num_heads or 1

    # Cap draft length to reasonable maximum (vLLM default is 5)
    draft_length = min(draft_length, 10)

    # EAGLE-3 gets higher default draft length (proven 6.5x speedup)
    if head_info.head_type == "eagle3":
        draft_length = max(draft_length, 5)

    # MTP: draft_length is derived from n_predict or number of MTP layers
    # Usually 1 per layer, but can be configured higher
    if head_info.head_type == "mtp":
        draft_length = max(draft_length, 1)

    config = SpecDecodingConfig(
        draft_length=draft_length,
        bonus_token=True,
    )

    logger.info(
        f"Auto-configured speculative decoding: "
        f"type={head_info.head_type}, "
        f"num_heads={head_info.num_heads}, "
        f"draft_length={draft_length}, "
        f"head_config={head_info.head_config}"
    )

    return config


@dataclass
class SpecDecodingConfig:
    """Configuration for speculative decoding."""
    # Number of draft tokens per step (K in EAGLE-3)
    draft_length: int = 5

    # Temperature for draft sampling (slightly higher than target)
    draft_temperature: float = 1.0

    # Whether to use speculative expert prediction for MoE models
    speculate_experts: bool = False

    # Maximum draft model size (in parameters) relative to target
    # EAGLE-3 recommends draft ≈ 1/10th of target
    max_draft_ratio: float = 0.15

    # Acceptance threshold: reject if draft prob / target prob > threshold
    # Standard speculative sampling uses threshold = 1.0
    acceptance_threshold: float = 1.0

    # Bonus token: always generate one extra after acceptance
    bonus_token: bool = True


@dataclass
class DraftResult:
    """Result from the draft model."""
    token_ids: list[int]
    logprobs: list[float]  # Draft model's log probabilities for each token


@dataclass
class VerifyResult:
    """Result from verifying draft tokens against the target model."""
    accepted_count: int  # Number of accepted tokens
    accepted_ids: list[int]  # The accepted token IDs
    rejected_at: int  # Index of first rejection (-1 if all accepted)
    bonus_token_id: int  # Extra token from target after acceptance
    target_logprobs: list[float]  # Target's log probs at each position


class SpeculativeDecoder:
    """Manages speculative decoding between draft and target models.

    Usage:
        decoder = SpeculativeDecoder(target_model, draft_model, tokenizer)
        tokens = decoder.generate(input_ids, max_tokens=512)
    """

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        tokenizer: Any,
        config: SpecDecodingConfig | None = None,
        lookahead: LookaheadReasoning | None = None,
        constraint: Any | None = None,
    ) -> None:
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.config = config or SpecDecodingConfig()
        self.rng = random.Random()
        self.lookahead = lookahead
        self.constraint = constraint

        self._stats = {
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "total_steps": 0,
            "total_bonus_tokens": 0,
        }

    @property
    def acceptance_rate(self) -> float:
        if self._stats["total_draft_tokens"] == 0:
            return 0.0
        return self._stats["total_accepted_tokens"] / self._stats["total_draft_tokens"]

    def generate_draft(
        self,
        input_ids: mx.array,
        cache: list,
        skip_prefill: bool = False,
    ) -> DraftResult:
        """Generate K draft tokens from the draft model.

        Args:
            input_ids: Current sequence token IDs [1, seq_len] or [1, 1].
            cache: Draft model's KV cache.
            skip_prefill: If True, assume input_ids is a single token already
                          in cache; only use it for the first forward logits.
                          This avoids double-prefilling when the cache already
                          has the prompt.

        Returns:
            DraftResult with proposed token IDs and their logprobs
        """
        K = self.lookahead.adjust_draft_k() if self.lookahead else self.config.draft_length
        token_ids = []
        logprobs = []

        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=self.config.draft_temperature)

        current_ids = input_ids

        for _ in range(K):
            output = self.draft(current_ids, cache=cache)
            logits = output.logits[:, -1, :] if hasattr(output, 'logits') else output[:, -1, :]

            # Apply grammar constraint masking if available
            if self.constraint is not None:
                try:
                    allowed = self.constraint.get_allowed_tokens(self.tokenizer, token_ids)
                    if allowed:
                        from .json_schema import apply_json_constraint
                        logits = apply_json_constraint(logits.reshape(1, -1), allowed).reshape(logits.shape)
                except Exception:
                    pass

            # Get log probabilities (numerically stable: avoid log(softmax) underflow)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            # Sample token
            next_token = sampler(logits)
            token_id = next_token.item()
            token_logprob = log_probs[0, token_id].item()

            token_ids.append(token_id)
            logprobs.append(token_logprob)

            # Feed back for next step
            current_ids = next_token.reshape(1, 1)

        mx.eval(current_ids)
        return DraftResult(token_ids=token_ids, logprobs=logprobs)

    def verify_draft(
        self,
        draft_result: DraftResult,
        input_ids: mx.array,
        cache: list,
        temperature: float = 0.0,
    ) -> VerifyResult:
        """Verify draft tokens against the target model in one pass.

        Correct logits alignment: includes input_ids (last token) in the
        forward pass alongside draft tokens, producing K+1 logits positions.
        This gives exactly K correct comparisons plus 1 bonus position.

          Feed [last_tok, d0, d1, ..., dK-1] to target:
            logits[0] = P(. | ctx, last_tok) -> compare with d0
            logits[i] = P(. | ctx, last_tok, d0..d{i-1}) -> compare with d[i]
            logits[K] = P(. | ctx, last_tok, d0..dK-1) -> bonus token

        EAGLE-3 acceptance criterion:
          Accept token i if: U < min(1, P_target(x_i) / P_draft(x_i))
          where U ~ Uniform(0, 1)

        Args:
            draft_result: Draft tokens and logprobs from generate_draft().
            input_ids: Last token(s) to include in forward pass for alignment.
            cache: Target model's KV cache.
            temperature: Sampling temperature for bonus/correction token.
                         Must match the target model's temperature for correct
                         output distribution.

        Returns:
            VerifyResult with accepted tokens and bonus token.
        """
        K = len(draft_result.token_ids)
        if K == 0:
            return VerifyResult(
                accepted_count=0,
                accepted_ids=[],
                rejected_at=-1,
                bonus_token_id=-1,
                target_logprobs=[],
            )

        # Roll back cache by 1 to re-include last_token in the forward pass.
        # Without this, last_token gets a duplicate KV entry and logits are
        # misaligned.  After rollback, the cache is as if last_token was never
        # processed, so feeding [last_tok, d0..dK-1] produces K+1 logits
        # correctly: logits[0] verifies d0, logits[K] is the bonus.
        try:
            from mlx_lm.models.cache import trim_prompt_cache
            trim_prompt_cache(cache, 1)
        except Exception:
            trimmed = False
            for c in cache:
                if hasattr(c, "trim"):
                    c.trim(1)
                    trimmed = True
            if not trimmed:
                logger.warning("Cannot rollback KV cache — verification logits may be misaligned")

        # Build aligned input: [last_token(s), d0, d1, ..., dK-1]
        last_tok = input_ids[:, -1:]  # [1, 1] — last token from previous step
        draft_tokens = mx.array(draft_result.token_ids).reshape(1, K)
        aligned_input = mx.concatenate([last_tok, draft_tokens], axis=1)  # [1, K+1]

        # Batched forward pass: all K+1 tokens in one call
        output = self.target(aligned_input, cache=cache)
        logits = output.logits if hasattr(output, 'logits') else output

        # Get target log probabilities (numerically stable: avoid log(softmax) underflow)
        target_logprobs_full = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Verify: logits[i] predicts position i+1 -> compare with draft[i]
        # We need logits[0..K-1] for draft verification, logits[K] for bonus
        target_logprobs = target_logprobs_full[0, :K, :]  # [K, vocab]
        bonus_logits = target_logprobs_full[0, K:K+1, :]  # [1, vocab]

        # Vectorized gather: extract target logprob for each draft token
        # target_lp[i] = target_logprobs[i, draft_result.token_ids[i]]
        draft_ids_arr = mx.array(draft_result.token_ids).reshape(K, 1)
        target_lps_arr = mx.take_along_axis(
            target_logprobs, draft_ids_arr, axis=-1
        ).squeeze(-1)
        draft_lps_arr = mx.array(draft_result.logprobs)

        # Vectorized acceptance ratio: min(threshold, exp(target_lp - draft_lp))
        # Clamp exponent to avoid mx.exp overflow/inf (matches _math_exp clamping
        # in spec_draft_verifier.py).  Large positive values mean p_target >> p_draft,
        # so ratio >= 1 and acceptance is guaranteed — clamping to exp(50) preserves this.
        # Uses acceptance_threshold from config (default 1.0 = standard speculative sampling).
        log_diff = mx.clip(target_lps_arr - draft_lps_arr, -50.0, 50.0)
        raw_ratios = mx.exp(log_diff)
        threshold = self.config.acceptance_threshold
        ratios = mx.minimum(
            mx.full(K, threshold),
            raw_ratios,
        )

        # Generate uniform random numbers for all positions at once
        uniforms = mx.array([self.rng.random() for _ in range(K)])
        accepted_mask = uniforms < ratios

        # Sequential scan: find first rejection (sequential dependency)
        accepted_ids = []
        target_lps = []
        rejected_at = -1

        for i in range(K):
            if bool(accepted_mask[i].item()):
                accepted_ids.append(draft_result.token_ids[i])
                target_lps.append(float(target_lps_arr[i].item()))
            else:
                rejected_at = i
                break

        # Bonus token: from target's distribution at rejection point or last.
        # Use the caller's temperature so the output distribution matches the
        # target model's sampling distribution (not hardcoded greedy).
        bonus_pos = len(accepted_ids)
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)

        # Apply grammar constraint masking to bonus token logits
        if self.constraint is not None:
            try:
                all_accepted = accepted_ids
                allowed = self.constraint.get_allowed_tokens(self.tokenizer, all_accepted)
                if allowed:
                    from .json_schema import apply_json_constraint
                    bonus_logits_raw = logits[0, bonus_pos:bonus_pos + 1, :]
                    bonus_logits_raw = apply_json_constraint(bonus_logits_raw.reshape(1, -1), allowed).reshape(1, 1, -1)
                    logits = logits.at[0, bonus_pos:bonus_pos + 1, :].set(bonus_logits_raw.reshape(1, -1))
            except Exception:
                pass

        if rejected_at >= 0:
            # Rejected at bonus_pos: use logits at that position for resample
            bonus_token = sampler(logits[0, bonus_pos:bonus_pos + 1, :])
        else:
            # All accepted: bonus from position K (prediction after all drafts)
            bonus_token = sampler(logits[0, K:K+1, :])
        bonus_id = bonus_token.item()

        return VerifyResult(
            accepted_count=len(accepted_ids),
            accepted_ids=accepted_ids,
            rejected_at=rejected_at,
            bonus_token_id=bonus_id,
            target_logprobs=target_lps,
        )

    @staticmethod
    def _snapshot_cache(cache: list) -> list:
        """Snapshot cache state by saving tensor references (no deep copy).

        MLX is functional — operations create new tensors, not in-place
        modifications. Saving references to the original tensors is sufficient
        for rollback because they remain valid and unmodified after subsequent
        forward passes overwrite the cache slots with new tensors.

        ArraysCache (linear_attention / SSM layers): save cache.cache list
        KVCache (full_attention layers): save offset
        """
        snapshot = []
        for c in cache:
            if hasattr(c, 'cache') and isinstance(getattr(c, 'cache', None), list):
                snapshot.append(('arrays', list(c.cache)))
            elif hasattr(c, 'offset'):
                snapshot.append(('kv', c.offset))
            else:
                snapshot.append((None, None))
        return snapshot

    @staticmethod
    def _restore_cache(cache: list, snapshot: list) -> None:
        """Restore cache state from a snapshot.

        For ArraysCache: restores the original tensor references (conv_state,
        ssm_state) — these are the exact same tensor objects, not copies.
        For KVCache: restores the offset (equivalent to trim).
        """
        for i, (kind, state) in enumerate(snapshot):
            if kind == 'arrays':
                cache[i].cache = state
            elif kind == 'kv':
                cache[i].offset = state

    def generate(
        self,
        input_ids: mx.array,
        max_tokens: int = 512,
        temperature: float = 0.7,
        cancel_event: "asyncio.Event | None" = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
    ) -> list[int]:
        """Generate tokens using speculative decoding.

        Main generation loop:
        1. Prefill both models, get first token from prefill logits
        2. Snapshot draft cache state (tensor references, no copy)
        3. Draft proposes K tokens
        4. Target verifies draft tokens one-by-one
        5. Accept matched tokens + bonus token
        6. On rejection: restore draft cache snapshot, re-feed accepted+correction
        7. Repeat until max_tokens or EOS

        Args:
            input_ids: Prompt token IDs [1, seq_len]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            cancel_event: Optional asyncio.Event for cooperative cancellation.
                          is_set() is checked each iteration; thread-safe in CPython.

        Returns:
            List of generated token IDs
        """
        eos_ids = set()
        if hasattr(self.tokenizer, 'eos_token_id'):
            eid = self.tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)

        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        target_cache = make_prompt_cache(self.target)
        draft_cache = make_prompt_cache(self.draft)

        target_sampler = make_sampler(temp=temperature) if temperature > 0 else None

        # Prefill both models
        t_out = self.target(input_ids, cache=target_cache)
        t_logits = t_out.logits[:, -1, :] if hasattr(t_out, 'logits') else t_out[:, -1, :]
        if target_sampler:
            first_token = int(target_sampler(t_logits).item())
        else:
            first_token = int(t_logits.argmax(axis=-1).item())

        self.draft(input_ids, cache=draft_cache)

        generated_tokens = [first_token]
        K = self.config.draft_length
        draft_sampler = make_sampler(temp=self.config.draft_temperature)

        while len(generated_tokens) < max_tokens:
            # Cooperative cancellation check (thread-safe for asyncio.Event)
            if cancel_event is not None and (
                cancel_event.is_set() if isinstance(cancel_event, asyncio.Event)
                else cancel_event.is_set()
            ):
                break

            # Budget check: cap draft length to remaining token budget
            remaining = max_tokens - len(generated_tokens)
            effective_K = min(K, remaining)

            # Snapshot draft cache before drafting (reference-based, no copy)
            draft_snap = self._snapshot_cache(draft_cache)

            # Checkpoint grammar constraint state before drafting
            if self.constraint is not None and hasattr(self.constraint, 'checkpoint'):
                try:
                    self.constraint.checkpoint()
                except Exception:
                    pass

            # Step 1: Draft generates K tokens from last generated token
            last_tok = generated_tokens[-1]
            draft_tokens = []
            draft_probs = []
            d_input = mx.array([[last_tok]])
            for _ in range(effective_K):
                # Check cancellation inside draft loop too
                if cancel_event is not None and (
                    cancel_event.is_set() if isinstance(cancel_event, asyncio.Event)
                    else cancel_event.is_set()
                ):
                    break
                d_out = self.draft(d_input, cache=draft_cache)
                d_logits = d_out.logits[:, -1, :] if hasattr(d_out, 'logits') else d_out[:, -1, :]
                # Apply grammar constraint masking in inline draft loop
                if self.constraint is not None:
                    try:
                        allowed = self.constraint.get_allowed_tokens(self.tokenizer, draft_tokens)
                        if allowed:
                            from .json_schema import apply_json_constraint
                            d_logits = apply_json_constraint(d_logits.reshape(1, -1), allowed).reshape(d_logits.shape)
                    except Exception:
                        pass
                d_logprobs = d_logits - mx.logsumexp(d_logits, axis=-1, keepdims=True)
                next_tok = draft_sampler(d_logits)
                tok_id = int(next_tok.item())
                draft_tokens.append(tok_id)
                draft_probs.append(float(d_logprobs[0, tok_id].item()))
                d_input = next_tok.reshape(1, 1)
                if tok_id in eos_ids:
                    break

            if not draft_tokens:
                break

            self._stats["total_draft_tokens"] += len(draft_tokens)

            # Step 2: Batch verify all K draft tokens in ONE forward pass
            last_tok = generated_tokens[-1]
            last_tok_arr = mx.array([[last_tok]])
            draft_result = DraftResult(
                token_ids=draft_tokens,
                logprobs=draft_probs,
            )
            verify_result = self.verify_draft(draft_result, last_tok_arr, target_cache, temperature=temperature)

            # SP-PEN: Apply penalty/bias to bonus token via extra target forward pass
            _has_pen = (
                repetition_penalty != 1.0
                or frequency_penalty != 0.0
                or presence_penalty != 0.0
                or (logit_bias is not None and len(logit_bias) > 0)
            )
            if _has_pen and verify_result.bonus_token_id >= 0:
                from .batched_engine import _apply_spec_bonus_penalties
                _pen_input = mx.array([[generated_tokens[-1]]])
                _pen_out = self.target(_pen_input, cache=target_cache)
                _pen_logits = _pen_out.logits[0, -1, :] if hasattr(_pen_out, 'logits') else _pen_out[0, -1, :]
                _token_hist = list(input_ids.flatten()) + generated_tokens
                _pen_logits = _apply_spec_bonus_penalties(
                    _pen_logits, _token_hist, int(input_ids.size),
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                )
                _pen_bonus = int(mx.argmax(_pen_logits).item())
                verify_result = VerifyResult(
                    accepted_count=verify_result.accepted_count,
                    accepted_ids=verify_result.accepted_ids,
                    rejected_at=verify_result.rejected_at,
                    bonus_token_id=_pen_bonus,
                    target_logprobs=verify_result.target_logprobs,
                )

            accepted = verify_result.accepted_count
            all_accepted = (accepted == len(draft_tokens))

            # Append accepted tokens — but respect max_tokens budget
            for tid in verify_result.accepted_ids:
                if len(generated_tokens) >= max_tokens:
                    break
                generated_tokens.append(tid)
                if tid in eos_ids:
                    return generated_tokens

            # Append correction/bonus token
            bonus_id = verify_result.bonus_token_id
            if bonus_id >= 0 and len(generated_tokens) < max_tokens:
                generated_tokens.append(bonus_id)
                # Bonus/correction token is always produced — count it regardless
                # of whether all drafts were accepted or only a prefix matched.
                self._stats["total_bonus_tokens"] += 1
                if bonus_id in eos_ids:
                    return generated_tokens

            self._stats["total_accepted_tokens"] += accepted
            self._stats["total_steps"] += 1

            # On partial acceptance: trim target cache of rejected entries,
            # restore draft cache, re-feed accepted + correction tokens.
            # Also rollback grammar constraint on rejection.
            if accepted < len(draft_tokens):
                # Rollback grammar constraint to pre-draft state
                if self.constraint is not None and hasattr(self.constraint, 'rollback'):
                    try:
                        self.constraint.rollback()
                    except Exception:
                        pass
                # Advance constraint with accepted tokens only
                if self.constraint is not None and hasattr(self.constraint, 'advance'):
                    for tid in verify_result.accepted_ids:
                        try:
                            self.constraint.advance(self.tokenizer.decode([tid]))
                        except Exception:
                            pass
                    # Advance for the correction token
                    correction = generated_tokens[-1]
                    try:
                        self.constraint.advance(self.tokenizer.decode([correction]))
                    except Exception:
                        pass
                # verify_draft rolled back 1 then fed [last_tok, d0..dK-1] (K+1 entries).
                # After rollback, cache had N-1 entries. After forward K+1: N+K.
                # Only accepted+1 are valid (last_tok + accepted drafts).
                # Trim the rejected ones: (N+K) - (N-1+accepted+1) = K - accepted.
                trim_count = len(draft_tokens) - accepted
                try:
                    from mlx_lm.models.cache import trim_prompt_cache
                    trim_prompt_cache(target_cache, trim_count)
                except Exception:
                    for c in target_cache:
                        if hasattr(c, "trim"):
                            c.trim(trim_count)

                self._restore_cache(draft_cache, draft_snap)
                # Re-feed accepted + correction tokens to draft cache.
                refeed = generated_tokens[-(accepted + 1):]
                for tok in refeed:
                    self.draft(mx.array([[tok]]), cache=draft_cache)

                # Feed correction token to target cache so it becomes the last
                # entry.  This ensures the next verify_draft's rollback will
                # correctly remove the correction (not the last accepted draft).
                correction = generated_tokens[-1]
                self.target(mx.array([[correction]]), cache=target_cache)
            else:
                # All accepted: advance grammar constraint for all accepted + bonus
                if self.constraint is not None and hasattr(self.constraint, 'advance'):
                    for tid in draft_tokens:
                        try:
                            self.constraint.advance(self.tokenizer.decode([tid]))
                        except Exception:
                            pass
                    if bonus_id >= 0:
                        try:
                            self.constraint.advance(self.tokenizer.decode([bonus_id]))
                        except Exception:
                            pass
                # feed bonus token to target cache for the same
                # reason — the bonus must be in cache before next verify_draft.
                if bonus_id >= 0 and len(generated_tokens) < max_tokens:
                    self.target(mx.array([[bonus_id]]), cache=target_cache)

            if any(t in eos_ids for t in generated_tokens):
                return generated_tokens

        return generated_tokens

    def get_stats(self) -> dict:
        """Return speculative decoding statistics."""
        return {
            "total_steps": self._stats["total_steps"],
            "total_draft_tokens": self._stats["total_draft_tokens"],
            "total_accepted_tokens": self._stats["total_accepted_tokens"],
            "total_bonus_tokens": self._stats["total_bonus_tokens"],
            "acceptance_rate": round(self.acceptance_rate, 3),
            "avg_accepted_per_step": (
                self._stats["total_accepted_tokens"] / self._stats["total_steps"]
                if self._stats["total_steps"] > 0 else 0.0
            ),
            "effective_speedup": (
                (self._stats["total_accepted_tokens"] + self._stats["total_bonus_tokens"])
                / self._stats["total_steps"]
                if self._stats["total_steps"] > 0 else 0.0
            ),
        }


class LookaheadReasoning:
    """Lookahead reasoning for thinking models.

    During the thinking phase (<think/> tags), this module:
    1. Detects when the model is in reasoning mode
    2. Signals to n-gram / spec decode proposers to draft more aggressively
    3. Caches the thinking output for potential reuse
    4. Adjusts draft_length dynamically based on thinking state

    For models like DeepSeek-R1, Qwen3 that emit long thinking chains.

    Works independently of any specific spec decode strategy — the thinking
    state is checked by BatchedEngine's generation loops to boost spec decode
    aggressiveness during reasoning.
    """

    def __init__(
        self,
        decoder: SpeculativeDecoder | None = None,
        base_draft_k: int = 5,
        thinking_draft_k: int = 10,
    ) -> None:
        self.decoder = decoder
        self._in_thinking = False
        self._thinking_tokens: list[int] = []
        self._base_draft_k = base_draft_k
        self._thinking_draft_k = thinking_draft_k
        # Track recent acceptance rate for adaptive adjustment
        self._recent_accepts: list[int] = []
        self._window = 10

    def check_thinking_state_text(self, text_chunk: str) -> None:
        """Track thinking state from decoded text (avoids re-decoding)."""
        if "<think" in text_chunk:
            self._in_thinking = True
            self._thinking_tokens = []
        elif "</think" in text_chunk:
            self._in_thinking = False

    def record_accept(self, count: int) -> None:
        """Record acceptance count for adaptive adjustment."""
        self._recent_accepts.append(count)
        if len(self._recent_accepts) > self._window:
            self._recent_accepts.pop(0)

    @property
    def in_thinking(self) -> bool:
        return self._in_thinking

    def adjust_draft_k(self) -> int:
        """Dynamically adjust draft tokens based on thinking state + acceptance rate."""
        if not self._in_thinking:
            return self._base_draft_k

        # During thinking: boost draft length
        if self._recent_accepts:
            avg_accept = sum(self._recent_accepts) / len(self._recent_accepts)
            # If acceptance rate is high (>70% of base_k), be more aggressive
            if avg_accept >= self._base_draft_k * 0.7:
                return self._thinking_draft_k
            # Moderate acceptance: slightly boost
            return min(self._base_draft_k + 2, self._thinking_draft_k)

        return self._thinking_draft_k

    def get_stats(self) -> dict:
        result = {
            "in_thinking": self._in_thinking,
            "thinking_tokens_cached": len(self._thinking_tokens),
            "base_draft_k": self._base_draft_k,
            "thinking_draft_k": self._thinking_draft_k,
            "current_k": self.adjust_draft_k(),
            "recent_avg_accept": (
                sum(self._recent_accepts) / len(self._recent_accepts)
                if self._recent_accepts else 0.0
            ),
        }
        if self.decoder is not None:
            result["decoder_stats"] = self.decoder.get_stats()
        return result
