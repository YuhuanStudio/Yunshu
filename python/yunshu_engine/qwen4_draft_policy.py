# Upstream (inspired): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/mtp.py _mtp_rounds_batch @ 92b31ad3
# Prior art (inspired): Strata draft policy (docs/DETAILS.md "Speculation": MTP drafts up to 3 tokens)
"""Confidence-gated drafting for the Qwen4 native MTP head (single greedy row).

The head chains up to ``block - 1`` drafts. This driver stops the chain when the
head's top-1 probability for a draft is below ``YUNSHU_QWEN4_DRAFT_MIN_PROB``:
that draft and the ones after it are not proposed, so the verify window is
shorter. Drafts are only proposals and the target still verifies every
committed token, so the output is the target's own greedy stream for every
depth and threshold. The first draft is always proposed. Threshold 0 never
stops early and is the upstream fixed-depth round.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import mlx.core as mx

_STATE: dict[str, Any] = {"installed": False, "stats": None}


def _top1_prob(logits: mx.array) -> mx.array:
    """Softmax probability of the argmax, shape ``[B]``."""
    x = logits.astype(mx.float32)
    return 1.0 / mx.sum(mx.exp(x - mx.max(x, axis=-1, keepdims=True)), axis=-1)


def track_seed_prob(draft_model: Any) -> None:
    """Make the head's seed draft carry its top-1 probability in ``_seed_prob``."""
    if getattr(draft_model, "_seed_prob_tracked", False):
        return

    def set_seed(hidden, sampler, greedy):
        logits = draft_model._lm_head_fn(hidden)
        draft_model._seed_prob = _top1_prob(logits[:, -1, :])
        draft_model._seed_token = (
            mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        )
        draft_model._seed_hidden = hidden

    draft_model._set_seed_from_hidden = set_seed
    draft_model._seed_prob = None
    draft_model._seed_prob_tracked = True


def draft_chain(
    draft_model: Any,
    bonus: int,
    hidden: mx.array,
    max_draft: int,
    min_prob: float,
    token_dtype: mx.Dtype,
) -> mx.array:
    """Up to ``max_draft`` greedy drafts ``[1, k]`` (``k >= 1``).

    A draft whose top-1 probability is below ``min_prob`` is dropped together
    with everything after it (the first draft is always kept). Mirrors
    ``MTPDraftBase.draft_block`` (greedy) and its bookkeeping.
    """
    tok = mx.array([[bonus]], dtype=token_dtype)
    h_prev = hidden
    tokens: list[mx.array] = []
    probs: list[mx.array | None] = []
    draft_model._round_appended = 0
    seed_tok = getattr(draft_model, "_seed_token", None)
    seed_h = getattr(draft_model, "_seed_hidden", None)
    if seed_tok is not None and seed_h is not None:
        tok = seed_tok.astype(token_dtype)
        h_prev = seed_h
        tokens.append(tok)
        probs.append(getattr(draft_model, "_seed_prob", None))
        draft_model._seed_token = None
        draft_model._seed_hidden = None
        draft_model._seed_prob = None
    gated = min_prob > 0.0
    while len(tokens) < max_draft:
        if gated and probs and probs[-1] is not None:
            if float(probs[-1].item()) < min_prob:
                break  # the last draft is dropped below
        logits_hidden, h_prev = draft_model._forward_tokens(tok, h_prev, token_dtype)
        draft_model._round_appended += 1
        logits = draft_model._lm_head_fn(logits_hidden)
        tok = mx.argmax(logits, axis=-1)
        tokens.append(tok)
        probs.append(_top1_prob(logits[:, -1, :]) if gated else None)
    if gated and len(tokens) > 1 and probs[-1] is not None:
        if float(probs[-1].item()) < min_prob:
            tokens.pop()
    draft_model._draft_round += 1
    return mx.concatenate(tokens, axis=1)


def rounds(
    model: Any,
    draft_model: Any,
    prompt_cache: list,
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    prompt_tokens: mx.array | None,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Any,
    draft_block_size: int | None,
    token_dtype: mx.Dtype,
    stop_check: Any,
    eos_token_ids: set | None,
    min_prob: float,
    row_ids: list[int] | None = None,
) -> Generator[tuple[list, dict | None]]:
    """Single-row greedy MTP rounds with a variable draft length."""
    from mlx_vlm.speculative.common import (
        _batch_cache_left_padding,
        _dflash_block_total,
    )
    from mlx_vlm.speculative.mtp import (
        _mtp_cache_positions,
        _mtp_draft_hidden,
        _mtp_draft_position,
        _mtp_verify_target,
        _slice_shared_kv_after_reject,
        _speculative_walk_batch_deferred_greedy,
        generation_stream,
    )

    lm = model.language_model if hasattr(model, "language_model") else model
    row_ids = [0] if row_ids is None else list(row_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    length, positions = _mtp_cache_positions(prompt_cache, 1)
    left_padding = [length - p for p in positions]
    if getattr(draft_model, "supports_ragged_batch_acceptance", False):
        draft_model.reset(model, left_padding=left_padding)
    else:
        draft_model.reset(model)
    track_seed_prob(draft_model)
    prefill = getattr(draft_model, "prefill_from_target_hidden", None)
    if callable(prefill) and prompt_tokens is not None:
        kw: dict[str, Any] = {"greedy": True}
        if getattr(draft_model, "supports_left_padded_prefill", False):
            kw["left_padding"] = left_padding
        prefill(prompt_tokens, hidden, first_bonus, sampler, token_dtype, **kw)
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    hidden = _mtp_draft_hidden(lm, hidden)
    draft_model.set_shared_kv(
        shared_kv_states,
        kv_offset=length,
        position=_mtp_draft_position(mx.array(positions)),
        kv_valid_len=mx.array(positions),
        left_padding=_batch_cache_left_padding(prompt_cache),
    )
    b = int(first_bonus.reshape(-1).item())
    emitted = 1
    position = positions[0]
    stats = _STATE["stats"]
    while True:
        bs = min(block_total, max_tokens - emitted + 1)
        if bs <= 1:
            break
        verify = None
        try:
            draft_tokens = draft_chain(
                draft_model, b, hidden, bs - 1, min_prob, token_dtype
            )
            bs = int(draft_tokens.shape[1]) + 1
            with mx.stream(generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
                )
                verify = _mtp_verify_target(
                    lm, verify_input, prompt_cache, sampler, sample_target_tokens=False
                )
                hidden_full = verify.hidden
            accepted_list, new_list = _speculative_walk_batch_deferred_greedy(
                lm,
                hidden_full,
                draft_tokens,
                sampler,
                [max_tokens - emitted],
                row_ids,
                [emitted],
            )
            if stats is not None:
                stats.append((bs - 1, accepted_list[0]))
            accept_verified = getattr(draft_model, "accept_verified_tokens_batch", None)
            if callable(accept_verified):
                accept_verified(
                    hidden_full,
                    draft_tokens,
                    accepted_list,
                    new_list,
                    sampler,
                    token_dtype,
                    greedy=True,
                )
            verify.commit(lm, prompt_cache, accepted_list, bs)
        except BaseException:
            if verify is not None:
                verify.abort()
            abort_draft = getattr(draft_model, "abort_draft_round", None)
            if callable(abort_draft):
                abort_draft()
            raise
        accepted, new_tokens = accepted_list[0], new_list[0]
        hidden = _mtp_draft_hidden(lm, hidden_full[:, accepted : accepted + 1, :])
        finished = False
        for pos, tok in enumerate(new_tokens):
            emitted += 1
            if emitted >= max_tokens:
                finished = True
            if eos_token_ids is not None and tok in eos_token_ids:
                finished = True
            if stop_check is not None and stop_check(0, tok):
                finished = True
            yield [tok], {"round_pos": pos, "round_len": len(new_tokens)}
            if finished:
                return
        b = new_tokens[-1] if new_tokens else b
        position += accepted + 1
        next_kv = _slice_shared_kv_after_reject(
            verify.shared_kv_states, bs - (accepted + 1)
        )
        draft_model.set_shared_kv(
            next_kv,
            kv_offset=position,
            position=_mtp_draft_position(mx.array([position])),
            kv_valid_len=mx.array([position]),
            left_padding=_batch_cache_left_padding(prompt_cache),
        )
        if emitted % 256 == 0:
            mx.clear_cache()


def install() -> bool:
    """Route single-row greedy qwen4 MTP requests through :func:`rounds` when
    ``YUNSHU_QWEN4_DRAFT_MIN_PROB`` > 0; everything else keeps upstream's loop."""
    if _STATE["installed"]:
        return True
    from mlx_vlm.generate import ar
    from mlx_vlm.speculative import utils as spec_utils

    from . import settings

    original = ar.run_speculative_server_rounds

    def run(model, draft_model, prompt_cache, hidden, **kw):
        min_prob = float(settings.get("YUNSHU_QWEN4_DRAFT_MIN_PROB") or 0.0)
        first = kw.get("first_bonus")
        if (
            min_prob > 0.0
            and kw.get("draft_kind") == "mtp"
            and kw.get("greedy_sampling")
            and type(draft_model).__name__ == "Qwen4ExpMTPDraftModel"
            and first is not None
            and int(first.shape[0]) == 1
        ):
            return rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                kw["shared_kv_states"],
                prompt_tokens=kw.get("prompt_tokens"),
                first_bonus=first,
                max_tokens=kw["max_tokens"],
                sampler=kw["sampler"],
                draft_block_size=kw.get("draft_block_size"),
                token_dtype=kw.get("token_dtype", mx.int32),
                stop_check=kw.get("stop_check"),
                eos_token_ids=kw.get("eos_token_ids"),
                min_prob=min_prob,
                row_ids=kw.get("row_ids"),
            )
        return original(model, draft_model, prompt_cache, hidden, **kw)

    ar.run_speculative_server_rounds = run
    spec_utils.run_speculative_server_rounds = run
    _STATE["installed"] = True
    return True


__all__ = ["draft_chain", "install", "rounds", "track_seed_prob"]
