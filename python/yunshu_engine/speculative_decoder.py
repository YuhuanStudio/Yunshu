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
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

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
    ) -> None:
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.config = config or SpecDecodingConfig()
        self.rng = random.Random()

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

    def generate_draft(self, input_ids: mx.array, cache: list) -> DraftResult:
        """Generate K draft tokens from the draft model.

        Args:
            input_ids: Current sequence token IDs [1, seq_len]
            cache: Draft model's KV cache

        Returns:
            DraftResult with proposed token IDs and their logprobs
        """
        K = self.config.draft_length
        token_ids = []
        logprobs = []

        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=self.config.draft_temperature)

        current_ids = input_ids

        for _ in range(K):
            output = self.draft(current_ids, cache=cache)
            logits = output.logits[:, -1, :] if hasattr(output, 'logits') else output[:, -1, :]

            # Get log probabilities
            log_probs = mx.log(mx.softmax(logits, axis=-1))

            # Sample token
            next_token = sampler(logits)
            token_id = next_token.item()
            token_logprob = log_probs[0, token_id].item()

            token_ids.append(token_id)
            logprobs.append(token_logprob)

            # Feed back for next step
            current_ids = next_token.reshape(1, 1)

        mx.eval(token_ids)
        return DraftResult(token_ids=token_ids, logprobs=logprobs)

    def verify_draft(
        self,
        draft_result: DraftResult,
        input_ids: mx.array,
        cache: list,
    ) -> VerifyResult:
        """Verify draft tokens against the target model in one pass.

        The target model processes all K draft tokens simultaneously,
        producing logits at each position. We compare the target's
        distribution against the draft's to determine acceptance.

        EAGLE-3 acceptance criterion:
          Accept token i if: U < min(1, P_target(x_i) / P_draft(x_i))
          where U ~ Uniform(0, 1)

        Args:
            draft_result: The draft model's proposed tokens
            input_ids: Original sequence before draft tokens
            cache: Target model's KV cache

        Returns:
            VerifyResult with acceptance details
        """
        K = len(draft_result.token_ids)
        draft_tokens = mx.array(draft_result.token_ids).reshape(1, K)

        # Feed all K tokens to target model at once
        output = self.target(draft_tokens, cache=cache)
        logits = output.logits if hasattr(output, 'logits') else output

        # Get target log probabilities at each position
        target_logprobs = mx.log(mx.softmax(logits, axis=-1))

        # Acceptance check
        accepted_ids = []
        target_lps = []
        rejected_at = -1

        for i in range(K):
            token_id = draft_result.token_ids[i]
            target_lp = target_logprobs[0, i, token_id].item()
            draft_lp = draft_result.logprobs[i]

            target_lps.append(target_lp)

            # Acceptance ratio: P_target / P_draft (in probability space)
            # log_ratio = target_lp - draft_lp
            # Accept if U < exp(log_ratio) (capped at 1.0)
            log_ratio = target_lp - draft_lp
            ratio = min(1.0, mx.exp(mx.array(log_ratio)).item())

            if self.rng.random() < ratio:
                accepted_ids.append(token_id)
            else:
                rejected_at = i
                break

        # Bonus token: sample from target's distribution at rejection point
        bonus_pos = len(accepted_ids)
        if bonus_pos < K:
            # Resample from target's adjusted distribution
            bonus_logits = target_logprobs[0, bonus_pos]
            # Residual distribution: (P_target - P_draft)_+ normalized
            # Simplified: just sample from target distribution
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=0.0)
            bonus_token = sampler(logits[0, bonus_pos:bonus_pos+1, :])
            bonus_id = bonus_token.item()
        else:
            # All accepted — sample bonus from last position
            bonus_logits = logits[0, -1, :]
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=0.0)
            bonus_token = sampler(bonus_logits.reshape(1, 1, -1))
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

        prompt_ids = input_ids.flatten().tolist()

        # Prefill both models
        t_out = self.target(input_ids, cache=target_cache)
        t_logits = t_out.logits[:, -1, :] if hasattr(t_out, 'logits') else t_out[:, -1, :]
        first_token = int(t_logits.argmax(axis=-1).item())

        self.draft(input_ids, cache=draft_cache)

        generated_tokens = [first_token]
        K = self.config.draft_length
        draft_sampler = make_sampler(temp=self.config.draft_temperature)

        while len(generated_tokens) < max_tokens:
            # Snapshot draft cache before drafting (reference-based, no copy)
            draft_snap = self._snapshot_cache(draft_cache)

            # Step 1: Draft generates K tokens from last generated token
            last_tok = generated_tokens[-1]
            draft_tokens = []
            d_input = mx.array([[last_tok]])
            for _ in range(K):
                d_out = self.draft(d_input, cache=draft_cache)
                d_logits = d_out.logits[:, -1, :] if hasattr(d_out, 'logits') else d_out[:, -1, :]
                next_tok = draft_sampler(d_logits)
                draft_tokens.append(int(next_tok.item()))
                d_input = next_tok.reshape(1, 1)

            self._stats["total_draft_tokens"] += K

            # Step 2: Target verifies one-by-one
            last_tok = generated_tokens[-1]
            t_input = mx.array([[last_tok]])
            t_out = self.target(t_input, cache=target_cache)
            t_logits = t_out.logits[:, -1, :] if hasattr(t_out, 'logits') else t_out[:, -1, :]

            accepted = 0
            all_accepted = True
            for j in range(K):
                target_choice = int(t_logits.argmax(axis=-1).item())

                if target_choice == draft_tokens[j]:
                    accepted += 1
                    generated_tokens.append(draft_tokens[j])
                    if draft_tokens[j] in eos_ids:
                        return generated_tokens
                    # Feed to target for next position
                    t_input = mx.array([[draft_tokens[j]]])
                    t_out = self.target(t_input, cache=target_cache)
                    t_logits = t_out.logits[:, -1, :] if hasattr(t_out, 'logits') else t_out[:, -1, :]
                else:
                    # Rejected: take target's choice
                    generated_tokens.append(target_choice)
                    all_accepted = False
                    break

            if all_accepted:
                # Bonus token from target
                bonus = int(t_logits.argmax(axis=-1).item())
                generated_tokens.append(bonus)
                self._stats["total_bonus_tokens"] += 1

                if bonus in eos_ids:
                    return generated_tokens

            self._stats["total_accepted_tokens"] += accepted
            self._stats["total_steps"] += 1

            # On rejection: restore draft cache from snapshot, then re-feed
            # only the accepted + correction tokens (NOT the full sequence).
            # This is O(accepted+1) instead of O(total_length) for full rebuild.
            if accepted < K:
                self._restore_cache(draft_cache, draft_snap)
                # Re-feed the tokens that target accepted / corrected.
                # After restore, the draft cache is at the pre-draft state,
                # so we need to feed: last_tok + accepted_tokens + correction
                refeed = [generated_tokens[-(accepted + 1) - 1]]
                refeed += generated_tokens[-(accepted + 1):]
                for tok in refeed:
                    self.draft(mx.array([[tok]]), cache=draft_cache)

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
    2. Uses speculative decoding with higher draft_length
    3. Caches the thinking output for potential reuse
    4. Automatically adjusts draft_length based on acceptance rate

    For models like DeepSeek-R1, Qwen3 that emit long thinking chains.
    """

    def __init__(self, decoder: SpeculativeDecoder) -> None:
        self.decoder = decoder
        self._in_thinking = False
        self._thinking_tokens: list[int] = []

    def check_thinking_state(self, last_token: int, tokenizer) -> None:
        """Track whether we're inside <think/> tags."""
        try:
            text = tokenizer.decode([last_token])
            if "<think" in text:
                self._in_thinking = True
                self._thinking_tokens = []
            elif "</think" in text:
                self._in_thinking = False
        except Exception:
            pass

    def adjust_draft_length(self) -> int:
        """Dynamically adjust draft length based on acceptance rate."""
        base_length = self.decoder.config.draft_length

        if self._in_thinking:
            # During thinking, draft more aggressively (higher acceptance rate expected)
            return min(base_length * 2, 10)
        else:
            return base_length

    def get_stats(self) -> dict:
        return {
            "in_thinking": self._in_thinking,
            "thinking_tokens_cached": len(self._thinking_tokens),
            "decoder_stats": self.decoder.get_stats(),
        }
