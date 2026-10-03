# Upstream (derived): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/mtp.py @ v0.7.3
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""MTP rounds for the speculative lane's single greedy request.

Same contract as ``mlx_vlm.speculative.mtp._mtp_rounds_batch`` for one row
(drafts are proposals, the target verifies every committed token, so output is
the target's own greedy stream), with the cycle rearranged so the GPU is not
idle or serialised behind host decisions:

- **Early absorb.** The MTP head reads ``(embed(token[p + 1]), hidden[p])`` and
  drafts ``token[p + 2]``. In a verify window every row ``i`` already has its
  ``hidden[i]`` and the target's next token ``t[i]`` on the GPU, so the head is
  run over *all* rows inside the verify's graph and read out for each row; once
  the host knows how many drafts landed (``a``) it keeps row ``a``'s result. The
  upstream loop instead waits for ``a`` and then runs a second dependent pass.
- **One read per cycle.** Drafts, the target's tokens and every row's next-draft
  seed come back in a single host read.
- **Chain first.** The next cycle's draft chain is submitted before the verify's
  rollback (``commit``) is built, so the host builds while the GPU drafts.

Only used where the lane is exact and greedy (no processors, no thinking
budget); everything else stays on upstream's loop.

**Tool-call guide.** A request with tools carries a ``ToolCallGuide``
(``tool_call_grammar``): free until the model emits the tool-call start marker,
then constrained to the exact call grammar. In a free stretch the lane is
unchanged except that a round is cut right after a marker token (the positions
after it were verified without the grammar's mask; the cache is rolled back to
the marker like any partial acceptance). In a constrained stretch the verify
window's target tokens are the argmax of the *masked* logits, one mask per
position along the draft path, so the round is exactly what plain masked decode
would produce token by token.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import Any

import mlx.core as mx

from .copy_cost import RoundCostClock
from .copy_drafter import CopyDrafter
from .keyed_sampling import KeyedSampler

logger = logging.getLogger(__name__)

_STATE: dict = {
    "installed": False,
    "enabled": True,
    "window": 0,
    "profile": None,
    "guide": None,
    "context": None,  # the request's FULL prompt ids (not the tail after a prefix hit)
    "copy_rows": 16,  # verify rows a copy round may use (0: copy rounds off)
}


def set_guide(guide: Any) -> None:
    """The tool-call guide of the request the lane is stepping (None: none)."""
    _STATE["guide"] = guide


def set_context(ids: Any) -> None:
    """The full prompt ids of the request the lane is stepping (None: none). The
    copy drafter indexes all of them, however much of the prompt a prefix-cache
    hit skipped."""
    _STATE["context"] = ids


def verify_max_rows(lane_projections: bool, language_model: Any = None) -> int:
    """Widest verify window whose rows equal plain decode: with the lane
    projections and the tile attention kernel the fused GDN, tile attention and
    lane matmul are row-invariant to 32 rows; otherwise (sg8 / packed
    projections) 8."""
    from .kernels import ragged_attention as ra

    if not lane_projections or not ra.tile_ready():
        return 8
    if language_model is not None:
        for layer in language_model.model.layers:
            # Expert projections are not converted to LaneLinear; their
            # wider-row arithmetic has not been certified by this lane.
            if hasattr(getattr(layer, "mlp", None), "switch_mlp"):
                return 8
            if layer.is_linear:
                continue
            attention = layer.self_attn
            if (
                attention.head_dim != 256
                or attention.num_attention_heads // attention.num_key_value_heads > 8
            ):
                return 8
    return ra.MAX_WINDOW


def set_copy_rows(rows: int, limit: int | None = None) -> int:
    """Verify rows a copy round may use (drafts <= rows - 1); 0 turns copy rounds
    off. ``limit`` is the verify's real maximum width (``verify_max_rows``); the
    rows are capped to it. Returns the rows in effect."""
    rows = max(0, int(rows))
    if limit is not None:
        rows = min(rows, max(0, int(limit)))
    _STATE["copy_rows"] = rows
    return rows


def copy_rows_for_model(language_model: Any, rows: int | None = None) -> int:
    """Recheck the current target, even if another loaded engine set the lane's
    process-global copy preference after this engine was constructed."""
    rows = int(_STATE["copy_rows"] if rows is None else rows)
    if rows < 3:
        return rows
    layers = getattr(getattr(language_model, "model", None), "layers", None)
    if not layers:  # unknown projection geometry keeps the conservative bound
        return min(rows, 8)
    from .kernels.lane_linear import LaneLinear

    layer = layers[0]
    projection = (
        layer.linear_attn.in_proj_qkv if layer.is_linear else layer.self_attn.q_proj
    )
    return min(
        rows, verify_max_rows(isinstance(projection, LaneLinear), language_model)
    )


def can_guide(draft_model: Any) -> bool:
    """True when ``rounds`` can apply a tool-call guide with this drafter."""
    return bool(
        _STATE["enabled"]
        and _STATE["installed"]
        and getattr(draft_model, "supports_greedy_draft_argmax", False)
    )


def _masked_targets(lm: Any, hidden: mx.array, masks: Any, dtype: mx.Dtype) -> mx.array:
    """Target tokens of a verify window under per-position token masks."""
    from .tool_call_grammar import apply_bitmask

    logits = lm.speculative_logits_from_hidden(hidden)
    return mx.argmax(apply_bitmask(logits, masks), axis=-1).reshape(1, -1).astype(dtype)


def set_profile(enabled: bool) -> dict | None:
    """Collect per-phase host wall time of the rounds (research aid): returns the
    accumulator dict (phase -> seconds, ``cycles``) or None."""
    _STATE["profile"] = {"cycles": 0} if enabled else None
    return _STATE["profile"]


def set_enabled(enabled: bool) -> None:
    _STATE["enabled"] = bool(enabled)


def set_window(window: int) -> None:
    """Prompt positions the head absorbs (0: all of them)."""
    _STATE["window"] = max(0, int(window))


def rounds(
    model: Any,
    draft_model: Any,
    prompt_cache: list,
    hidden: mx.array,
    *,
    prompt_tokens: mx.array | None,
    first_bonus: int,
    max_tokens: int,
    sampler: Any,
    draft_block_size: int | None,
    token_dtype: mx.Dtype,
    stop_check: Any,
    eos_token_ids: set | None,
    guide: Any = None,
    keyed: Any = None,
) -> Generator[tuple[list, dict | None]]:
    import mlx_vlm.speculative.mtp as mtp
    from mlx_vlm.speculative.common import (
        _dflash_block_total,
        _record_speculative_round,
    )

    lm = model.language_model if hasattr(model, "language_model") else model
    block_total = _dflash_block_total(draft_model, draft_block_size)
    draft_model.reset(model)
    window = int(_STATE["window"])
    if prompt_tokens is not None:
        length = int(prompt_tokens.shape[1])
        if window and length > window:
            # Drafts only propose: the head sees the last ``window`` prompt
            # positions (absolute positions kept), so its per-pass attention and
            # its prefill stop growing with the prompt.
            lo = length - window
            bonus = mx.array([[first_bonus]], dtype=token_dtype)
            shifted = mx.concatenate(
                [prompt_tokens[:, 1:].astype(token_dtype), bonus], axis=1
            )
            draft_model._next_position = lo
            h = draft_model._forward_tokens(
                shifted[:, lo:], hidden[:, lo:length, :], token_dtype
            )
            draft_model._set_seed_from_hidden(h[:, -1:, :], sampler, True)
        else:
            draft_model.prefill_from_target_hidden(
                prompt_tokens,
                hidden,
                first_bonus,
                sampler,
                token_dtype,
                greedy=True,
            )
    readout = draft_model._greedy_token
    vocab = getattr(draft_model, "_draft_vocab", None)
    seed_tok, seed_h = draft_model._seed_token, draft_model._seed_hidden
    draft_model._seed_token = draft_model._seed_hidden = None
    if seed_tok is None:  # no prompt tokens to absorb: seed from the last hidden
        seed_h = hidden[:, -1:, :]
        seed_tok = readout(seed_h)
    mx.eval(seed_tok, seed_h)  # the head's pass over the prompt, before the rounds

    def chain(seed_tok, seed_h, bs):
        """The seed plus ``bs - 2`` chained head passes: ``[1, bs - 1]`` drafts."""
        toks = [seed_tok.astype(token_dtype)]
        tok, h = toks[0], seed_h
        for _ in range(bs - 2):
            h = draft_model._forward_token(tok, h, token_dtype)
            tok = readout(h).astype(token_dtype)
            toks.append(tok)
        out = mx.concatenate(toks, axis=1)
        mx.async_eval(out)
        return out, bs - 2

    emitted = 1  # the caller already emitted the first bonus
    b = int(first_bonus)
    copy = None
    context = _STATE["context"]
    copy_rows = copy_rows_for_model(lm)
    if copy_rows >= 3 and context is not None:
        copy = CopyDrafter(max_draft=copy_rows - 1)
        copy.extend(context)
        copy.extend([b])

    def plan_copy(done: int) -> list[int]:
        """Copy draft for the next round (``[]``: the model drafts)."""
        if copy is None:
            return []
        cap = min(max_tokens - done, copy.max_draft)
        return copy.draft(cap) if cap >= 2 else []

    next_copy = plan_copy(emitted)
    if guide is not None:
        guide.feed(b)
    finished = False
    queued = None  # (drafts, chained entries, block) built ahead of the rollback

    prof = _STATE["profile"]
    clock = time.perf_counter
    cost_clock = (
        RoundCostClock() if copy is not None and copy.costs is not None else None
    )

    def mark(name, since):
        now = clock()
        if prof is not None:
            prof[name] = prof.get(name, 0.0) + now - since
        return now

    while emitted < max_tokens and not finished:
        is_copy = bool(next_copy)
        bs = (
            len(next_copy) + 1
            if is_copy
            else min(block_total, max_tokens - emitted + 1)
        )
        if bs <= 1:
            break
        t = clock()
        if is_copy:
            # A copied run is verified like any draft; the head's chain is not built.
            draft_tokens, chained = mx.array([next_copy], dtype=token_dtype), 0
            next_copy = []
            if queued is not None:  # cannot happen (no chain is queued before a copy)
                chained = queued[1]
        elif queued is not None and queued[2] == bs:
            draft_tokens, chained, _ = queued
        else:
            draft_tokens, chained = chain(seed_tok, seed_h, bs)
        queued = None
        verify = None
        t = mark("chain_build", t)
        try:
            # the verify's graph is built while the chain runs on the GPU
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
            )
            masks = None
            if guide is not None and guide.constrained:
                # Each position's mask depends on the drafts before it: read the
                # chain back first (only inside a tool call, a short stretch).
                masks = guide.plan(draft_tokens.reshape(-1).tolist(), bs)
                guide.lane_rounds += masks is not None
            if keyed is None and masks is None:
                verify = mtp._mtp_verify_target(
                    lm, verify_input, prompt_cache, sampler, sample_target_tokens=True
                )
                target = verify.target_tokens.reshape(1, -1).astype(token_dtype)
            else:
                # Masked and/or sampled request. A sampled row r draws generation index
                # ``emitted + r`` with the keyed sampler, so a draft is accepted exactly
                # when serial sampling would have produced it (see keyed_sampling); a
                # tool-call mask is applied to the logits first, as serial decoding does.
                verify = mtp._mtp_verify_target(
                    lm, verify_input, prompt_cache, sampler, sample_target_tokens=False
                )
                logits = lm.speculative_logits_from_hidden(verify.hidden)
                if masks is not None:
                    from .tool_call_grammar import apply_bitmask

                    logits = apply_bitmask(logits, masks)
                if keyed is None:
                    target = (
                        mx.argmax(logits, axis=-1).reshape(1, -1).astype(token_dtype)
                    )
                else:
                    logits = logits[0]
                    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    target = keyed.sample_positions(
                        logprobs, list(range(emitted, emitted + bs))
                    ).reshape(1, -1)
                    target = target.astype(token_dtype)
            # Early absorb: every row through the head with the true tokens.
            # The chain's own entries (built from the head's hidden) go first.
            if chained:
                for c in draft_model._cache:
                    c.trim(chained)
                draft_model._next_position = draft_model._next_position - chained
            head_out = draft_model._forward_tokens(target, verify.hidden, token_dtype)
            seeds = readout(head_out).astype(token_dtype)  # [1, bs]
            flat = mx.concatenate([draft_tokens, target, seeds], axis=1)
            t = mark("verify_build", t)
            mx.async_eval(flat, head_out)
            t = mark("submit", t)
            values = flat.reshape(-1).tolist()
            t = mark("gpu_wait", t)
            round_ms = cost_clock.readback(t) if cost_clock is not None else None
            drafted = values[: bs - 1]
            tgt = values[bs - 1 : 2 * bs - 1]
            next_seed = values[2 * bs - 1 :]
            accepted = 0
            while accepted < bs - 1 and drafted[accepted] == tgt[accepted]:
                accepted += 1
            new_tokens = (drafted[:accepted] + [tgt[accepted]])[: max_tokens - emitted]
            if guide is not None:
                # Keep the tokens up to the first one that arms a constraint (the
                # ones after it were verified without its mask), and advance the
                # guide over what is kept.
                kept = guide.advance(new_tokens)
                if kept < len(new_tokens):
                    new_tokens = new_tokens[:kept]
                    accepted = kept - 1
            # per-round drafted / accepted counts (drafter lifetime counters; the runner
            # diffs them per request for x_yunshu.speculative)
            _record_speculative_round(draft_model, accepted, bs - 1)
            # Rows 0..accepted stay in the head's cache; later rows go.
            rejected = bs - (accepted + 1)
            if rejected:
                for c in draft_model._cache:
                    c.trim(rejected)
                draft_model._next_position = draft_model._next_position - rejected
            seed_tok = mx.array([[next_seed[accepted]]], dtype=token_dtype)
            seed_h = head_out[:, accepted : accepted + 1, :]
            if vocab is not None:  # the reduced draft readout follows the script
                vocab.learn(new_tokens)
            t = mark("walk", t)
            # The next chain is queued before the rollback is built.
            if copy is not None:
                if is_copy:
                    copy.observe_copy(bs - 1, accepted, round_ms)
                    draft_model.copy_total_rounds = (
                        getattr(draft_model, "copy_total_rounds", 0) + 1
                    )
                    draft_model.copy_total_tokens = getattr(
                        draft_model, "copy_total_tokens", 0
                    ) + len(new_tokens)
                else:
                    copy.observe_model(len(new_tokens), round_ms)
                copy.extend(new_tokens)
                next_copy = plan_copy(emitted + len(new_tokens))
            nb = min(block_total, max_tokens - (emitted + len(new_tokens)) + 1)
            if nb > 1 and len(new_tokens) == accepted + 1 and not next_copy:
                queued = (*chain(seed_tok, seed_h, nb), nb)
            t = mark("next_chain_build", t)
            verify.commit(lm, prompt_cache, accepted, bs)
            t = mark("commit_build", t)
            if prof is not None:
                prof["cycles"] += 1
                prof["tokens"] = prof.get("tokens", 0) + len(new_tokens)
        except BaseException:
            if verify is not None:
                verify.abort()
            abort = getattr(draft_model, "abort_draft_round", None)
            if callable(abort):
                abort()
            raise

        n = len(new_tokens)
        for pos, tok in enumerate(new_tokens):
            emitted += 1
            if emitted >= max_tokens:
                finished = True
            if eos_token_ids is not None and tok in eos_token_ids:
                finished = True
            if stop_check is not None and stop_check(0, tok):
                finished = True
            publication = clock() if cost_clock is not None else None
            yield [tok], {"round_pos": pos, "round_len": n}
            if cost_clock is not None and publication is not None:
                cost_clock.published((clock() - publication) * 1e3)
            if finished:
                break
        b = new_tokens[-1] if new_tokens else b
        if emitted % 256 == 0:
            mx.clear_cache()


def install() -> bool:
    """Serve single-row greedy MTP requests of ``BatchGenerator`` with
    ``rounds``; other requests keep upstream's loop."""
    if _STATE["installed"]:
        return True
    from mlx_vlm.generate import ar
    from mlx_vlm.speculative import utils as spec_utils

    original = ar.run_speculative_server_rounds

    def run(model, draft_model, prompt_cache, hidden, **kw):
        first = kw.get("first_bonus")
        if (
            _STATE["enabled"]
            and kw.get("draft_kind") == "mtp"
            and (
                kw.get("greedy_sampling") or isinstance(kw.get("sampler"), KeyedSampler)
            )
            and first is not None
            and int(first.shape[0]) == 1
            and getattr(draft_model, "supports_greedy_draft_argmax", False)
        ):
            return rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                prompt_tokens=kw.get("prompt_tokens"),
                first_bonus=int(first.reshape(-1).item()),
                max_tokens=kw["max_tokens"],
                sampler=kw["sampler"],
                draft_block_size=kw.get("draft_block_size"),
                token_dtype=kw.get("token_dtype", mx.int32),
                stop_check=kw.get("stop_check"),
                eos_token_ids=kw.get("eos_token_ids"),
                guide=_STATE["guide"],
                keyed=kw["sampler"]
                if isinstance(kw.get("sampler"), KeyedSampler)
                else None,
            )
        if _STATE["guide"] is not None:
            logger.warning(
                "tool-call guide dropped: the speculative lane's rounds are not in use"
            )
        return original(model, draft_model, prompt_cache, hidden, **kw)

    ar.run_speculative_server_rounds = run
    spec_utils.run_speculative_server_rounds = run
    _STATE["installed"] = True
    return True


__all__ = ["install", "rounds", "set_enabled", "set_window"]
