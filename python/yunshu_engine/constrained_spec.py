# Upstream (derived): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/dflash.py @ v0.7.3
# Patches upstream symbols (see vendor.json); target rows always use serial arithmetic.
"""Request-local grammar transactions and accepted target probabilities.

The serial processor and speculative verifier share the very same constraint;
walking proposals is a transaction, never a change to the committed grammar.
Only singleton token masks are jumped over: forced bytes can have multiple BPE
tokenizations, so llguidance's byte fast-forward alone is not token-exact.
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import numpy as np

from .tool_call_grammar import apply_bitmask

logger = logging.getLogger(__name__)
_REQUEST: Any = None
_SERIAL_VERIFY = False


def set_request(request: Any) -> None:
    global _REQUEST
    _REQUEST = request


def current_request() -> Any:
    return _REQUEST


def prepare_serial_prefill(gen):
    """Use the same singleton KV representation as ordinary AR prefill."""
    prompt = getattr(gen, "_prompt_batch", None)
    if prompt is None:
        return
    prepare_serial_prompt(prompt)


def prepare_serial_prompt(prompt):
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    for i, cache in enumerate(prompt.prompt_cache):
        if isinstance(cache, BatchKVCache):
            prompt.prompt_cache[i] = KVCache() if cache.empty() else cache.extract(0)
        elif isinstance(cache, ArraysCache):
            # The batch factory adds an all-zero padding mask even to one row.
            # Native AR has no mask; the masked GDN path can round differently.
            cache.left_padding = None
            cache.lengths = None


def finish_serial_prefill(gen):
    """Convert stock-prefilled KV to the unpadded dense verify representation."""
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    if getattr(gen, "_prompt_batch", None) is not None:
        return
    batch = getattr(gen, "_generation_batch", None)
    if batch is None:
        return
    for i, cache in enumerate(getattr(batch, "prompt_cache", [])):
        if isinstance(cache, KVCache):
            dense = BatchKVCache.merge([cache])
            dense.left_padding = None
            dense.offset = int(cache.offset)
            batch.prompt_cache[i] = dense


@contextmanager
def exact_verify(request=None, guide=None):
    """Constrained/logprob rows use serial arithmetic, including non-winning logits.

    Greedy token equality of the ordinary fast verify is insufficient for keyed
    sampling and logprobs. Reuse the existing one-row arithmetic kernel routing;
    leave unqualified requests on their established fast path.
    """
    global _SERIAL_VERIFY
    from .kernels import batch_invariant

    if not batch_invariant.is_installed() or not (
        guide is not None or (request and request.logprobs)
    ):
        yield
        return
    from .kernels import omlx
    from .kernels.omlx import qwen35_verify_qmm as qmm

    active = batch_invariant._STATE["active"]
    serial = _SERIAL_VERIFY
    row_exact = omlx._STATE.get("row_exact", False)
    armed, exact_armed = qmm._is_armed(), qmm.is_row_exact_armed()
    batch_invariant.set_active(False)
    omlx._STATE["row_exact"] = True
    qmm.set_verify_qmm_armed(True, row_exact=True)
    _SERIAL_VERIFY = True
    if request is not None:
        if not request.serial_verify_rounds:
            logger.info(
                "Serial target-row verify engaged: row_exact=True invariant=False"
            )
        request.serial_verify_rounds += 1
    try:
        yield
    finally:
        batch_invariant.set_active(active)
        _SERIAL_VERIFY = serial
        omlx._STATE["row_exact"] = row_exact
        qmm.set_verify_qmm_armed(armed or exact_armed, row_exact=exact_armed)


class ConstraintGuide:
    """A serial ConstraintProcessor exposed as a transactional bitmask guide."""

    constrained = True

    def __init__(self, processor: Any, vocab_size: int):
        self.processor = processor
        self.vocab_size = vocab_size
        self.words = (vocab_size + 31) // 32
        self.lane_rounds = 0
        c = processor._constraint
        self._saved_arg = bool(inspect.signature(c.rollback).parameters)

    def mask(self):
        p = self.processor
        c = p._constraint
        # llguidance's native bitmask is identical to its allowlist, including
        # the serial backend's accepting-state EOS addition.
        if getattr(c, "_matcher", None) is not None:
            import llguidance.numpy as lnp

            from .constraint_eos import normalize_eos_ids

            if c._dead:
                raise ValueError("Grammar constraint has no valid next token")
            out = np.zeros((self.words,), dtype=np.int32)
            if not c._done and not c._matcher.is_stopped():
                # llguidance requires exactly its tokenizer width, while the
                # model head can have padded vocabulary rows. Keep those zero.
                lnp.fill_next_token_bitmask(c._matcher, out[None, : c._words], 0)
            if c._done or c._matcher.is_stopped() or c._matcher.is_accepting():
                for tok in normalize_eos_ids(p._tokenizer):
                    out.view(np.uint32)[tok >> 5] |= np.uint32(1 << (tok & 31))
        else:
            fast = getattr(c, "allowed_mask", None)
            mask = fast(p._tokenizer, self.vocab_size) if fast else None
            if mask is None:
                allowed = c.get_allowed_tokens(p._tokenizer, p._generated)
                bits = np.zeros(self.words * 32, dtype=np.uint8)
                bits[allowed] = 1
            else:
                bits = np.zeros(self.words * 32, dtype=np.uint8)
                values = np.asarray(mask, dtype=np.uint8)
                bits[: min(len(values), self.vocab_size)] = values[: self.vocab_size]
            out = np.packbits(bits, bitorder="little").view(np.int32)
        if not np.any(out):
            raise ValueError("Grammar constraint has no valid next token")
        return out

    def checkpoint(self):
        return self.processor._constraint.checkpoint(), len(self.processor._generated)

    def restore(self, saved):
        cp, n = saved
        c = self.processor._constraint
        if self._saved_arg:
            c.rollback(cp)
            discard = getattr(c, "discard_checkpoint", None)
            if discard:
                discard()
        else:
            c.rollback()
        del self.processor._generated[n:]

    def feed(self, token):
        p = self.processor
        p._generated.append(int(token))
        p._constraint.advance(p._tokenizer.decode([int(token)]))
        return True

    def advance(self, tokens):
        for token in tokens:
            self.feed(token)
        return len(tokens)

    def plan(self, drafts, n_pos):
        cp = self.checkpoint()
        out = np.full((n_pos, self.words), -1, dtype=np.int32)
        try:
            for i in range(n_pos):
                out[i] = self.mask()
                if i == len(drafts):
                    break
                tok = drafts[i]
                if not (out[i, tok >> 5] >> (tok & 31)) & 1:
                    break
                self.feed(tok)
        finally:
            self.restore(cp)
        return out


class CombinedGuide:
    """Intersect response-format and tool masks without dropping either contract."""

    def __init__(self, text, tool):
        self.guides = (text, tool)
        self.words = text.words
        self.lane_rounds = 0

    @property
    def constrained(self):
        return any(g.constrained for g in self.guides)

    def mask(self):
        out = np.full(self.words, -1, dtype=np.int32)
        for g in self.guides:
            row = g.mask()
            if row is not None:
                out &= row
        if not np.any(out):
            raise ValueError("Grammar constraint has no valid next token")
        return out

    def checkpoint(self):
        return [g.checkpoint() for g in self.guides]

    def restore(self, cp):
        for g, saved in zip(self.guides, cp, strict=True):
            g.restore(saved)

    def feed(self, token):
        return all([g.feed(token) for g in self.guides])

    def advance(self, tokens):
        for i, token in enumerate(tokens):
            arms = any(getattr(g, "arms", lambda t: False)(token) for g in self.guides)
            self.feed(token)
            if arms:
                return i + 1
        return len(tokens)

    def plan(self, drafts, n_pos):
        return ConstraintGuide.plan(self, drafts, n_pos)


def forced_tokens(guide: Any, limit: int) -> list[int]:
    """Return a singleton-token prefix without advancing the live guide."""
    if guide is None or not guide.constrained:
        return []
    cp = guide.checkpoint()
    out = []
    probes = [g for g in getattr(guide, "guides", (guide,)) if hasattr(g, "_planning")]
    for g in probes:
        g._planning += 1
    try:
        for _ in range(limit):
            row = guide.mask()
            if row is None:
                break
            # Count bits without expanding an entire vocabulary-sized mask.
            words = np.asarray(row, dtype=np.int32).view(np.uint32)
            nonzero = np.flatnonzero(words)
            if len(nonzero) != 1:
                break
            word = int(words[nonzero[0]])
            if word & (word - 1):
                break
            token = int(nonzero[0]) * 32 + word.bit_length() - 1
            out.append(token)
            if not guide.feed(token) or not guide.constrained:
                break
    finally:
        guide.restore(cp)
        for g in probes:
            g._planning -= 1
    request = current_request()
    if len(out) >= 2 and request is not None:
        request.forced_windows += 1
        if request.forced_windows == 1:
            logger.info("Forced-token verify window engaged: %d drafts", len(out))
    return out


def normalize_rows(logits):
    """Use serial [1,V] reductions at each position, including bf16 rounding."""
    return mx.concatenate(
        [row - mx.logsumexp(row, axis=-1, keepdims=True) for row in logits[0, :, None]],
        axis=0,
    )


def target_rows(logits, masks, emitted, keyed, request=None):
    if masks is not None:
        logits = apply_bitmask(logits, masks)
    if request is not None and request.logprobs:
        logits = logits.astype(mx.float32)
    probs = normalize_rows(logits)
    target = (
        keyed.sample_positions(probs, list(range(emitted, emitted + logits.shape[1])))
        if keyed is not None
        else mx.argmax(probs, axis=-1)
    )
    return target.reshape(1, -1), probs


class SpecRequest:
    """Own first-token logits and bounded accepted-row reports for one request."""

    def __init__(self, logprobs=False, top_logprobs=0):
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.first_probs = None
        self.pending: deque = deque()
        self.guide = None
        self.forced_windows = 0
        self.serial_verify_rounds = 0

    def __call__(self, tokens, logits):
        if self.logprobs:
            logits = logits.astype(mx.float32)
        return logits

    def record(self, probs, tokens):
        self.pending.extend(self.reports(probs, tokens))

    def reports(self, probs, tokens):
        if not self.logprobs:
            return []
        probs = probs[: len(tokens)]
        ids = mx.array(tokens, mx.int32)
        chosen = mx.take_along_axis(probs, ids[:, None], axis=-1).reshape(-1)
        k = min(self.top_logprobs, probs.shape[-1])
        if k:
            # Match AR ordering, including ties; reduce only committed rows.
            top = mx.argsort(probs, axis=-1)[:, -k:][:, ::-1].astype(mx.int32)
            vals = mx.take_along_axis(probs, top, -1)
            top_ids, top_vals = top.tolist(), vals.tolist()
        else:
            top_ids = top_vals = [[] for _ in tokens]
        return [
            (int(tok), float(lp), list(zip(ts, vs, strict=True)))
            for tok, lp, ts, vs in zip(
                tokens, chosen.tolist(), top_ids, top_vals, strict=True
            )
        ]

    def take(self, token):
        if self.first_probs is not None:
            self.record(self.first_probs, [token])
            self.first_probs = None
        tok, lp, top = self.pending.popleft()
        if tok != token:
            raise RuntimeError("speculative logprob row/token mismatch")
        return lp, top


def dflash_rounds(model, draft_model, prompt_cache, hidden, *, request, **kw):
    """Single-row DFlash chain with the same target mask/sampler as serial decode.

    Keep the upstream cache transaction and context preparation; drafts (including
    copied/forced proposals) are never used as probability sources.
    """
    from mlx_vlm.speculative.cache_state import (
        abort_speculative_round,
        commit_speculative_round,
    )
    from mlx_vlm.speculative.common import (
        _dflash_block_total,
        _record_speculative_round,
    )
    from mlx_vlm.speculative.dflash import (
        _dflash_next_block_size,
        _reserve_dflash_target_cache,
    )

    from . import mtp_lane
    from .copy_drafter import CopyDrafter
    from .keyed_sampling import KeyedSampler

    lm = getattr(model, "language_model", model)
    guide = request.guide
    keyed = kw["sampler"] if isinstance(kw["sampler"], KeyedSampler) else None
    dtype = kw.get("token_dtype", mx.int32)
    maximum = kw["max_tokens"]
    block = _dflash_block_total(draft_model, kw.get("draft_block_size"))
    ceiling = getattr(draft_model, "choose_block_ceiling", None)
    if ceiling:
        block = ceiling(int(hidden.shape[1]), block)
    initial = getattr(draft_model, "dflash_initial_block_size", None)
    choose_initial = getattr(draft_model, "choose_initial_block_size", None)
    if choose_initial:
        initial = choose_initial(int(hidden.shape[1]), block)
    cache = draft_model.reset(model)
    _reserve_dflash_target_cache(prompt_cache, block)
    prepare = getattr(draft_model, "prepare_target_hidden", None)
    if prepare:
        hidden = prepare(hidden)
    bonus = int(kw["first_bonus"].reshape(-1).item())
    emitted = 1
    if guide is not None:
        guide.feed(bonus)
    copy = None
    context = mtp_lane._STATE["context"]
    rows = mtp_lane._STATE["copy_rows"]
    if rows >= 3 and context is not None:
        copy = CopyDrafter(max_draft=rows - 1)
        copy.extend(context)
        copy.extend([bonus])
    while emitted < maximum:
        bs = _dflash_next_block_size(draft_model, block, maximum - emitted + 1, initial)
        proposals = forced_tokens(guide, min(bs - 1, maximum - emitted))
        copied = False
        if len(proposals) < 2:
            proposals = (
                copy.draft(min(copy.max_draft, maximum - emitted)) if copy else []
            )
            copied = len(proposals) >= 2
        if len(proposals) >= 2:
            drafts = mx.array([proposals], dtype=dtype)
            bs = len(proposals) + 1
        else:
            # The draft distribution need not be constrained or sampled. Matching
            # the position-keyed target is the sole acceptance criterion.
            fn = getattr(draft_model, "draft_block_greedy", draft_model.draft_block)
            drafts = fn(
                bonus,
                hidden,
                cache,
                bs,
                lambda x: mx.argmax(x, axis=-1),
                dtype,
                **({"target_hidden_prepared": True} if prepare else {}),
            )
        mx.async_eval(drafts)
        drafted = drafts.reshape(-1).tolist()
        masks = (
            guide.plan(drafted, bs) if guide is not None and guide.constrained else None
        )
        if masks is not None:
            guide.lane_rounds += 1
        unabsorbed = hidden if len(proposals) >= 2 else None
        states = None
        try:
            inputs = mx.concatenate([mx.array([[bonus]], dtype=dtype), drafts], axis=1)
            with exact_verify(request, guide):
                captured, final, states = lm.speculative_verify_dflash_hidden(
                    inputs, prompt_cache, list(draft_model.config.target_layer_ids)
                )
                logits = lm.speculative_logits_from_hidden(final)
            targets, probs = target_rows(logits, masks, emitted, keyed, request)
            hidden = mx.concatenate(captured, axis=-1)
            mx.async_eval(targets, hidden)
            tgt = targets.reshape(-1).tolist()
            accepted = 0
            while accepted < bs - 1 and drafted[accepted] == tgt[accepted]:
                accepted += 1
            tokens = (drafted[:accepted] + [tgt[accepted]])[: maximum - emitted]
            if guide is not None:
                kept = guide.advance(tokens)
                if kept < len(tokens):
                    tokens = tokens[:kept]
                    accepted = kept - 1
            _record_speculative_round(draft_model, accepted, bs - 1)
            reports = request.reports(probs, tokens)
            hidden = hidden[:, : accepted + 1, :]
            commit_speculative_round(lm, prompt_cache, states, accepted, bs)
        except BaseException:
            abort_speculative_round(states)
            raise
        if copy:
            if copied:
                copy.observe_copy(bs - 1, accepted)
                draft_model.copy_total_rounds = (
                    getattr(draft_model, "copy_total_rounds", 0) + 1
                )
                draft_model.copy_total_tokens = getattr(
                    draft_model, "copy_total_tokens", 0
                ) + len(tokens)
            else:
                copy.observe_model(len(tokens))
            copy.extend(tokens)
        if prepare:
            hidden = prepare(hidden)
        if unabsorbed is not None:
            # A forced/copy window skipped the drafter's context projection.
            # Retain its pending context until a model-draft round absorbs it.
            hidden = mx.concatenate([unabsorbed, hidden], axis=1)
        if prepare or unabsorbed is not None:
            mx.async_eval(hidden)
        bonus = tokens[-1]
        for pos, token in enumerate(tokens):
            emitted += 1
            if reports:
                request.pending.append(reports[pos])
            yield [token], {"round_pos": pos, "round_len": len(tokens)}
            stop = kw.get("stop_check")
            if emitted >= maximum or (stop and stop(0, token)):
                return


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from mlx_vlm.generate import ar
    from mlx_vlm.speculative import utils

    original = ar.run_speculative_server_rounds
    sample = ar._sample_with_positions
    prompt_init = ar.PromptProcessingBatch.__init__

    def serial_prompt_init(self, *args, **kwargs):
        prompt_init(self, *args, **kwargs)
        request = current_request()
        if request is not None and (request.guide is not None or request.logprobs):
            # Upstream may create AND prefill the batch inside one next() call.
            # Preparing it only at the next runner slice is already too late.
            prepare_serial_prompt(self)

    ar.PromptProcessingBatch.__init__ = serial_prompt_init

    # GDN's small floating-point gates are not LaneLinear projections. The
    # verifier's dense reduction can differ from ordinary one-row GEMV even
    # when every quantized projection is row-invariant.
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    linear = Qwen3_5BatchInvariantForward._linear
    linears = Qwen3_5BatchInvariantForward._linears

    def serial_linear(self, projection, x):
        if (
            _SERIAL_VERIFY
            and isinstance(projection, nn.Linear)
            and x.ndim == 3
            and x.shape[1] > 1
        ):
            return mx.concatenate(
                [projection(x[:, p : p + 1]) for p in range(x.shape[1])], axis=1
            )
        return linear(self, projection, x)

    def serial_linears(self, projections, x):
        if _SERIAL_VERIFY and any(isinstance(p, nn.Linear) for p in projections):
            return tuple(serial_linear(self, p, x) for p in projections)
        return linears(self, projections, x)

    Qwen3_5BatchInvariantForward._linear = serial_linear
    Qwen3_5BatchInvariantForward._linears = serial_linears

    from .kernels.omlx import qwen35_gdn_verify_fused

    fused_gdn = qwen35_gdn_verify_fused.fused_eligible

    def serial_gdn(*args):
        return not _SERIAL_VERIFY and fused_gdn(*args)

    qwen35_gdn_verify_fused.fused_eligible = serial_gdn

    def sample_first(sampler, logprobs, *, row_ids, positions):
        request = current_request()
        if request is not None and request.logprobs and positions == [0]:
            # Retain the exact reduction already used to sample the first token.
            request.first_probs = logprobs
            logger.info("First target distribution captured from prefill sampler")
        return sample(sampler, logprobs, row_ids=row_ids, positions=positions)

    ar._sample_with_positions = sample_first

    def run(model, draft_model, prompt_cache, hidden, **kw):
        request = current_request()
        if (
            request is not None
            and (request.guide is not None or request.logprobs)
            and kw.get("draft_kind") == "dflash"
            and kw["first_bonus"].size == 1
        ):
            return dflash_rounds(
                model, draft_model, prompt_cache, hidden, request=request, **kw
            )
        return original(model, draft_model, prompt_cache, hidden, **kw)

    ar.run_speculative_server_rounds = utils.run_speculative_server_rounds = run
    _INSTALLED = True
