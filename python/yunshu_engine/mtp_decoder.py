from __future__ import annotations
"""MTP always-advance speculative decoder with n_confirmed skip state.

Implements the always-advance MTP strategy with the key optimization from oMLX:

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
  - Cooldown on rejection (from llama.cpp PR #20700)
  - FastMTP vocabulary trimming (from llama.cpp PR #20700)
  - Cancel event for graceful mid-generation abort (production-ready)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class MTPConfig:
    """Configuration for MTP speculative decoding."""
    max_tokens: int = 256
    cooldown_on_reject: bool = False
    fastmtp_top_k: int = 0  # 0 = disabled, 32768 = llama.cpp default
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
    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id'):
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
        if hasattr(c, 'cache') and isinstance(getattr(c, 'cache', None), list):
            snap.append(('arrays', list(c.cache)))
        elif hasattr(c, 'offset'):
            snap.append(('kv', c.offset))
        else:
            snap.append((None, None))
    return snap


def _restore_cache(cache: list, snapshot: list) -> None:
    """Restore cache state from a reference-based snapshot."""
    for i, (kind, state) in enumerate(snapshot):
        if kind == 'arrays':
            cache[i].cache = state
        elif kind == 'kv':
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

    def _mtp_draft(self, hidden: mx.array, primary: int) -> int:
        """Generate draft token via MTP head."""
        mtp_logits = self.model.mtp_forward(
            hidden, mx.array([[primary]]), None,
        )
        return _greedy(mtp_logits[0, -1, :])

    def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
        cancel_event: Optional["asyncio.Event"] = None,
    ) -> list[int]:
        """Generate tokens using MTP always-advance with n_confirmed.

        Args:
            prompt: Text string or pre-tokenized ID list.
            max_tokens: Override config max_tokens.
            cancel_event: Optional asyncio.Event — checked each cycle for early abort.

        Returns:
            List of generated token IDs.
        """
        import asyncio
        max_tokens = max_tokens or self.config.max_tokens
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt)
        else:
            ids = list(prompt)

        eos_ids = _get_eos_ids(self.tokenizer)

        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)

        # Prefill
        prompt_t = mx.array(ids).reshape(1, -1)
        out, hidden = self.model(prompt_t, cache=cache, return_hidden=True)
        mx.synchronize()
        first = _greedy(out[0, -1, :])

        generated = [first]
        primary = first
        primary_h = hidden[:, -1:, :]

        # Cooldown state
        in_cooldown = False
        stats = self._stats

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
                    mx.array([[primary]]), cache=cache, return_hidden=True,
                )
                mx.synchronize()
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

            # MTP draft
            draft = self._mtp_draft(primary_h, primary)

            # Verify: backbone forward [primary, draft]
            # With n_confirmed=1, SSM layers save rollback state after token 0
            verify_kwargs = {}
            if self.config.use_n_confirmed:
                verify_kwargs["n_confirmed"] = 1

            verify_out, verify_h = self.model(
                mx.array([[primary, draft]]), cache=cache,
                return_hidden=True, **verify_kwargs,
            )
            mx.synchronize()
            v0 = _greedy(verify_out[0, 0, :])
            v1 = _greedy(verify_out[0, 1, :])

            stats.total_cycles += 1

            if v0 == draft:
                # Accept: draft matched backbone's prediction at pos0
                stats.accepts += 1
                if self.config.use_n_confirmed:
                    clear_rollback(cache)

                generated.append(draft)
                if draft in eos_ids or len(generated) >= max_tokens:
                    break

                # Bonus token (v1) becomes next primary
                generated.append(v1)
                if v1 in eos_ids or len(generated) >= max_tokens:
                    break
                primary = v1
                primary_h = verify_h[:, -1:, :]
            else:
                # Reject
                stats.rejects += 1

                if self.config.use_n_confirmed:
                    # n_confirmed path: restore rollback (no extra forward!)
                    restore_rollback(cache)
                    # Use hidden at pos 0 (confirmed/primary position)
                    # for next MTP draft — same as oMLX pattern
                    primary_h = verify_h[:, 0:1, :]
                else:
                    # Old path: restore cache + refeed primary (expensive)
                    _restore_cache(cache, snap)
                    out2, hid2 = self.model(
                        mx.array([[primary]]), cache=cache, return_hidden=True,
                    )
                    mx.synchronize()
                    primary_h = hid2[:, -1:, :]

                # In both paths, v0 is the correct token at position 0
                # (backbone's greedy choice at primary position)
                generated.append(v0)
                if v0 in eos_ids or len(generated) >= max_tokens:
                    break
                primary = v0

                if self.config.cooldown_on_reject:
                    in_cooldown = True

        stats.tokens_generated = len(generated)
        return generated


def run_mtp_decode(
    model, tokenizer, prompt: str,
    max_tokens: int = 128,
    cooldown: bool = False,
    use_n_confirmed: bool = True,
) -> dict:
    """Run MTP decode and return results dict (for benchmarking).

    Returns dict with tokens, timing, acceptance stats.
    """
    import time

    decoder = MTPDecoder(model, tokenizer, MTPConfig(
        max_tokens=max_tokens,
        cooldown_on_reject=cooldown,
        use_n_confirmed=use_n_confirmed,
    ))

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
