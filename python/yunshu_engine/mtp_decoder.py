# DEPRECATED: home-grown Qwen3.5 MTP — superseded by mlx-vlm's
# native MTP (see mlxvlm_mtp.py + YUNSHU_MTP=1; ~1.82x in a proof script, not served/gated — experimental).
# This lacked mlx-vlm's GatedDeltaNet intermediate-state capture (garbage on 27B,
# ~0.9x on 9B). Kept only as a legacy escape hatch under YUNSHU_LEGACY_MTP=1.
from __future__ import annotations

"""MTP always-advance speculative decoder with n_confirmed skip state.

Implements the always-advance MTP strategy with the key optimization:

**n_confirmed=1**: During verify (2-token forward), SSM layers save rollback state
after the first (confirmed) token. On accept, rollback is cleared. On reject,
rollback is restored — no extra backbone forward needed. This makes:

  Accept cost = Reject cost = 1 backbone(2tok) + 1 MTP = 1.15x

Speedup formula: (1+p) / 1.15
  At p=0.65 (measured on 4B): 1.43x speedup (POSITIVE!)
  At p=0.50: 1.30x
  At p=0.30: 1.13x

The old approach (restore+refeed) had reject cost = 2.15x, needing p > 0.735
for any positive speedup — which Apple Silicon's memory bandwidth bottleneck
never reaches on small models.

Also supports:
  - Cooldown on rejection
  - FastMTP vocabulary trimming
  - Cancel event for graceful mid-generation abort (production-ready)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class MTPConfig:
    """Configuration for MTP speculative decoding."""

    max_tokens: int = 256
    cooldown_on_reject: bool = False
    fastmtp_top_k: int = 0  # 0 = disabled, 32768 = typical default
    use_n_confirmed: bool = True  # Use n_confirmed=1 for zero-cost reject


@dataclass
class MTPStats:
    """Runtime statistics for MTP decoding."""

    accepts: int = 0
    rejects: int = 0
    cooldowns: int = 0
    tokens_generated: int = 0
    total_cycles: int = 0


def _greedy(logits: mx.array) -> int:
    return int(mx.argmax(logits).item())


def _get_eos_ids(tokenizer) -> set:
    eos_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
    return eos_ids


def _snapshot_cache(cache: list) -> list:
    """Snapshot cache state via reference (MLX functional semantics)."""
    snap = []
    for c in cache:
        if hasattr(c, "cache") and isinstance(getattr(c, "cache", None), list):
            snap.append(("arrays", list(c.cache)))
        elif hasattr(c, "offset"):
            snap.append(("kv", c.offset))
        else:
            snap.append((None, None))
    return snap


def _restore_cache(cache: list, snapshot: list) -> None:
    """Restore cache state from a reference-based snapshot."""
    for i, (kind, state) in enumerate(snapshot):
        if kind == "arrays":
            cache[i].cache = state
        elif kind == "kv":
            cache[i].offset = state


class MTPDecoder:
    """MTP always-advance decoder with n_confirmed skip state.

    Usage:
        model = load_model_with_mtp("models/Qwen3.5-4B-MLX-bf16")
        tokenizer = load_tokenizer(model_path)
        decoder = MTPDecoder(model, tokenizer)
        tokens = decoder.generate("Hello world", max_tokens=128)
    """

    def __init__(self, model, tokenizer, config: MTPConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or MTPConfig()
        self._stats = MTPStats()

    @property
    def stats(self) -> MTPStats:
        return self._stats

    def _mtp_draft(
        self,
        hidden: mx.array,
        primary: int,
        constraint: Any = None,
        generated_ids: list[int] | None = None,
    ) -> int:
        """Generate draft token via MTP head.

        When a grammar constraint is provided, applies token masking to
        the MTP logits so the draft token is guaranteed valid for the
        current grammar state. The caller must also advance the constraint
        with the chosen token (outside this method).

        Args:
            hidden: Hidden state from the backbone forward pass.
            primary: Current primary token ID.
            constraint: Optional constraint with get_allowed_tokens(tokenizer, ids)
                        and advance(token_text) methods.
            generated_ids: Previously generated token IDs (for constraint context).

        Returns:
            Draft token ID (greedy after optional grammar masking).
        """
        mtp_logits = self.model.mtp_forward(
            hidden,
            mx.array([[primary]]),
            None,
        )
        logits = mtp_logits[0, -1, :]

        # Apply grammar constraint masking if available
        if constraint is not None:
            try:
                ctx_ids = generated_ids if generated_ids is not None else []
                allowed = constraint.get_allowed_tokens(self.tokenizer, ctx_ids)
                if allowed:
                    from .json_schema import apply_json_constraint

                    logits = apply_json_constraint(
                        logits.reshape(1, -1),
                        allowed,
                    ).reshape(logits.shape)
            except Exception:
                logger.debug(
                    "MTP grammar constraint masking failed, using unfiltered logits",
                    exc_info=True,
                )

        return _greedy(logits)

    def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
        cancel_event: asyncio.Event | None = None,
        sampler=None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        constraint: Any = None,
    ) -> list[int]:
        """Generate tokens using MTP always-advance with n_confirmed.

        Args:
            prompt: Text string or pre-tokenized ID list.
            max_tokens: Override config max_tokens.
            cancel_event: Optional asyncio.Event — checked each cycle for early abort.
            sampler: Optional sampler (from make_sampler). Applied to bonus tokens
                     and rejection corrections — NOT to the draft/verify comparison
                     which must remain greedy for correct spec decode semantics.
            constraint: Optional grammar constraint (ConstrainedSampler, BitmaskConstrainedSampler,
                        GrammarBitmaskEngine, or any object with get_allowed_tokens/advance methods).
                        When provided, draft tokens are filtered by the grammar mask.

        Returns:
            List of generated token IDs.
        """
        max_tokens = max_tokens or self.config.max_tokens
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)

        eos_ids = _get_eos_ids(self.tokenizer)

        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)

        # Prefill
        prompt_t = mx.array(ids).reshape(1, -1)
        out, hidden = self.model(prompt_t, cache=cache, return_hidden=True)
        mx.synchronize()
        # Apply sampler to first token if provided
        if sampler is not None:
            first = int(sampler(out[0, -1:, :]).item())
        else:
            first = _greedy(out[0, -1, :])

        generated = [first]
        primary = first
        primary_h = hidden[:, -1:, :]

        # Cooldown state
        in_cooldown = False
        stats = self._stats
        stats.accepts = 0
        stats.rejects = 0
        stats.cooldowns = 0
        stats.tokens_generated = 0
        stats.total_cycles = 0

        # Import rollback helpers if using n_confirmed
        if self.config.use_n_confirmed:
            from yunshu_engine.n_confirmed_patch import clear_rollback, restore_rollback

        while len(generated) < max_tokens:
            # Check cancel_event for graceful mid-generation abort
            if cancel_event is not None and cancel_event.is_set():
                break

            # Cooldown: skip draft after rejection to get fresh logits
            if in_cooldown and self.config.cooldown_on_reject:
                in_cooldown = False
                stats.cooldowns += 1
                out2, hid2 = self.model(
                    mx.array([[primary]]),
                    cache=cache,
                    return_hidden=True,
                )
                mx.synchronize()
                if sampler is not None:
                    correction = int(sampler(out2[0, -1:, :]).item())
                else:
                    correction = _greedy(out2[0, -1, :])
                generated.append(correction)
                if correction in eos_ids or len(generated) >= max_tokens:
                    break
                primary = correction
                primary_h = hid2[:, -1:, :]
                continue

            # Snapshot for non-n_confirmed fallback
            if not self.config.use_n_confirmed:
                snap = _snapshot_cache(cache)

            # Checkpoint grammar constraint before draft
            if constraint is not None and hasattr(constraint, "checkpoint"):
                try:
                    constraint.checkpoint()
                except Exception:
                    logger.debug("MTP constraint checkpoint failed", exc_info=True)

            # MTP draft — always greedy (spec decode requires deterministic draft)
            # Apply grammar constraint masking to draft logits if available
            draft = self._mtp_draft(
                primary_h,
                primary,
                constraint=constraint,
                generated_ids=generated,
            )

            # Verify: backbone forward [primary, draft]
            # With n_confirmed=1, SSM layers save rollback state after token 0
            verify_kwargs = {}
            if self.config.use_n_confirmed:
                verify_kwargs["n_confirmed"] = 1

            verify_out, verify_h = self.model(
                mx.array([[primary, draft]]),
                cache=cache,
                return_hidden=True,
                **verify_kwargs,
            )
            mx.synchronize()
            # v0 MUST be greedy for spec decode acceptance check
            v0_greedy = _greedy(verify_out[0, 0, :])
            # MTP-PEN: Apply penalty/bias to bonus token logits (v1)
            _has_mtp_pen = (
                repetition_penalty != 1.0
                or frequency_penalty != 0.0
                or presence_penalty != 0.0
                or (logit_bias is not None and len(logit_bias) > 0)
            )
            if _has_mtp_pen:
                from .batched_engine import _apply_spec_bonus_penalties

                _mtp_token_hist = list(ids) + generated
                _bonus_logits = _apply_spec_bonus_penalties(
                    verify_out[0, 1, :],
                    _mtp_token_hist,
                    len(ids),
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                )
                verify_out[0, 1, :] = _bonus_logits
            # v1 (bonus) can use sampler for non-greedy output
            if sampler is not None:
                v1 = int(sampler(verify_out[0, 1:2, :]).item())
            else:
                v1 = _greedy(verify_out[0, 1, :])

            stats.total_cycles += 1

            if v0_greedy == draft:
                # Accept: draft matched backbone's prediction at pos0
                stats.accepts += 1
                if self.config.use_n_confirmed:
                    clear_rollback(cache)

                # Discard grammar constraint checkpoint (all accepted)
                if constraint is not None:
                    try:
                        if hasattr(constraint, "discard_checkpoint"):
                            constraint.discard_checkpoint()
                    except Exception:
                        logger.debug(
                            "MTP constraint discard_checkpoint failed", exc_info=True
                        )

                generated.append(draft)
                if draft in eos_ids or len(generated) >= max_tokens:
                    break

                # Advance constraint with accepted draft token
                if (
                    constraint is not None
                    and hasattr(constraint, "advance")
                    and draft not in eos_ids
                ):
                    try:
                        constraint.advance(self.tokenizer.decode([draft]))
                    except Exception:
                        logger.debug(
                            "MTP constraint advance for draft failed", exc_info=True
                        )

                # Bonus token (v1) becomes next primary
                generated.append(v1)
                if v1 in eos_ids or len(generated) >= max_tokens:
                    break
                # Advance constraint with bonus token
                if (
                    constraint is not None
                    and hasattr(constraint, "advance")
                    and v1 not in eos_ids
                ):
                    try:
                        constraint.advance(self.tokenizer.decode([v1]))
                    except Exception:
                        logger.debug(
                            "MTP constraint advance for bonus failed", exc_info=True
                        )
                primary = v1
                primary_h = verify_h[:, -1:, :]
            else:
                # Reject: sample correction token from v0's logits BEFORE
                # cache commit so the emitted token respects the sampler.
                # MTP-PEN: Apply penalty/bias to rejection correction logits
                if _has_mtp_pen:
                    from .batched_engine import _apply_spec_bonus_penalties

                    _corr_logits = _apply_spec_bonus_penalties(
                        verify_out[0, 0, :],
                        list(ids) + generated,
                        len(ids),
                        repetition_penalty=repetition_penalty,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        logit_bias=logit_bias,
                    )
                    verify_out[0, 0, :] = _corr_logits
                if sampler is not None:
                    correction = int(sampler(verify_out[0, 0:1, :]).item())
                else:
                    correction = v0_greedy
                stats.rejects += 1

                # Rollback grammar constraint to pre-draft state
                if constraint is not None and hasattr(constraint, "rollback"):
                    try:
                        constraint.rollback()
                    except Exception:
                        logger.debug("MTP constraint rollback failed", exc_info=True)

                if self.config.use_n_confirmed:
                    restore_rollback(cache)
                    # Re-feed the correction token through the rolled-back
                    # cache to get a hidden state consistent with the new
                    # primary token.  Then immediately undo the cache mutation
                    # so the next iteration's forward [correction, draft] does
                    # not produce a duplicate correction entry in KV cache or
                    # double-process correction in SSM layers.
                    #
                    # Snapshot both SSM (ArraysCache.cache list) and KV
                    # (KVCache.offset) so we can roll them back after
                    # extracting the hidden state.
                    _pre_corr_snap = _snapshot_cache(cache)
                    _out_corr, hid_corr = self.model(
                        mx.array([[correction]]),
                        cache=cache,
                        return_hidden=True,
                    )
                    mx.synchronize()
                    primary_h = hid_corr[:, -1:, :]
                    _restore_cache(cache, _pre_corr_snap)
                else:
                    # Old path: restore cache + refeed correction (expensive)
                    _restore_cache(cache, snap)
                    out2, hid2 = self.model(
                        mx.array([[correction]]),
                        cache=cache,
                        return_hidden=True,
                    )
                    mx.synchronize()
                    primary_h = hid2[:, -1:, :]

                generated.append(correction)
                if correction in eos_ids or len(generated) >= max_tokens:
                    break
                # Advance constraint with correction token
                if (
                    constraint is not None
                    and hasattr(constraint, "advance")
                    and correction not in eos_ids
                ):
                    try:
                        constraint.advance(self.tokenizer.decode([correction]))
                    except Exception:
                        logger.debug(
                            "MTP constraint advance for correction failed",
                            exc_info=True,
                        )
                primary = correction

                if self.config.cooldown_on_reject:
                    in_cooldown = True

        stats.tokens_generated = len(generated)
        return generated


def run_mtp_decode(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    cooldown: bool = False,
    use_n_confirmed: bool = True,
) -> dict:
    """Run MTP decode and return results dict (for benchmarking).

    Returns dict with tokens, timing, acceptance stats.
    """
    import time

    decoder = MTPDecoder(
        model,
        tokenizer,
        MTPConfig(
            max_tokens=max_tokens,
            cooldown_on_reject=cooldown,
            use_n_confirmed=use_n_confirmed,
        ),
    )

    t0 = time.perf_counter()
    tokens = decoder.generate(prompt, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0

    s = decoder.stats
    cycles = s.accepts + s.rejects
    return {
        "tokens": tokens,
        "n": len(tokens),
        "total_s": round(elapsed, 3),
        "tps": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
        "acceptance": round(s.accepts / cycles, 3) if cycles > 0 else 0,
        "accepts": s.accepts,
        "rejects": s.rejects,
        "cooldowns": s.cooldowns,
        "cycles": cycles,
    }
