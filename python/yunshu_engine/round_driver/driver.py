# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/engine/lane_engine.py, src/tensorfold/engine/lane_family.py @ 34bae79a
"""Round driver: rows advance in packed forwards, prefill and decode interleaved.

See docs/guides/ROUND_DRIVER.md. Each step is one forward of one kind:

- a **decode step**: every decoding row's window (pending token + drafts) in
  one launch per layer (``batch.DecodeBatch``: shared KV slot buffers and GDN
  state arrays);
- a **prefill step**: fixed ``chunk``-token chunks of waiting prompts —
  several prompts in one forward, each on its own caches — up to
  ``idle_budget`` tokens when no row decodes, one chunk while rows
  decode. A prompt that completes joins the decode batch.

While rows decode and prompts wait, prefill and decode steps alternate
one-to-one, so the chunk size sets the trade between the waiting prompt's
TTFT and the decoding rows' rate (27B, 4 rows + a 16K prompt: decode rows and
a prefill chunk packed into one forward measured no better than alternating
separate forwards at the same chunk size, docs/research/runs/
2026-09-29-fused-prefill/, so steps stay single-kind). A decode step:

1. runs one forward over the rows' windows (right-padded to the longest) and
   one LM-head call over the real window positions;
2. samples: greedy rows take the argmax of every window position and keep the
   drafts up to the first mismatch plus the target's token there; sampled rows
   with a keyed sampler draw every window position at its generation index
   (kept while equal to the draft), rows with logits processors run one
   token a step through their own sampler / processors (never drafting);
3. commits each window (KV lengths advance, rows that kept part of a window
   continue their GDN state from that position), runs stop / length checks;
4. keeps every draftable row's MTP head current with the kept positions, then
   drafts for the next step with the cost-aware allocation (``allocate``) over
   the measured step cost curve.

The driver never looks at other rows to decide a row's tokens: greedy output is
the same alone, in any batch, with any draft depth (row-invariant forward).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from .. import keyed_sampling
from ..cache_prefill import reserve_prompt_kv
from ..kernels import preempt
from ..keyed_sampling import top_k_filter, top_p_filter
from ..serving.work_scheduler import (
    DECODE_EARLY_TOKENS,
    DECODE_FIRST_BURST_S,
    decode_quantum,
)
from ..text_logprobs import top_logprobs_device
from . import replay
from .allocate import CostCurve, allocate, chain
from .batch import MAX_WINDOW, DecodeBatch
from .forward import Segment, lane_forward, logits

logger = logging.getLogger(__name__)

MAX_DECODE_TOKENS = MAX_WINDOW  # pending token + up to 7 drafts
FINISH_STEPS = (
    4  # extra prefill steps a step may run for rows within a chunk of their end
)
LONG_FROM = 16384  # positions from here on prefill in LONG_CHUNK atoms
LONG_CHUNK = 2048
CHUNK = 512  # default prefill chunk: fixed spans from the prompt start
SHORT_FIRST = 512  # a prompt with at most this many rows left runs before longer ones
IDLE_BUDGET = 2048  # prefill tokens per prefill step when no row decodes
ACCEPT_PRIOR = 0.7  # per-depth draft acceptance before a row has history
ACCEPT_EMA = 0.15
LANE_ROWS = 128  # rows one lane-matmul call keeps row-invariant
# MLX's freed-buffer cache: long prompts allocate transient buffers of ever
# new sizes (KV growth, attention scores), and left unbounded the cache grew
# to ~95 GiB in a 4 x 32K run, after which every step took 5-10x as long
# (docs/guides/ROUND_DRIVER.md). The driver keeps it under this bound.
CACHE_LIMIT = 8 * 2**30


def cache_buffers(cache: list) -> list:
    """A row's cache arrays (KV buffers, GDN conv / recurrent state)."""
    out = []
    for c in cache:
        if getattr(c, "keys", None) is not None:
            out += [c.keys, c.values]
        else:
            out += [a for a in getattr(c, "cache", ()) if isinstance(a, mx.array)]
    return out


@dataclass
class Request:
    """What a row needs from the caller (the runner's job)."""

    ids: list[int]
    max_tokens: int
    sampling: Any = None  # RowParams (vlm_batch_runner) or None: greedy
    processors: list = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int = 0
    draft: bool = True  # greedy without processors / logprobs may draft
    guide: Any = None  # invariant tool-call guide, advanced on emitted tokens
    budget: Any = None  # mlx_vlm ThinkingBudgetCriteria (forces "\n</think>")
    handle: Any = None
    extra_hash: int = 0  # APC salt (model / processor identity)
    use_apc: bool = True  # the runner's capacity check for this prompt
    cache_store: bool = True  # False: look the prefix up, store nothing
    # Media prompt: [L, D] prompt embeddings (visual features scattered in), [3, L] mRoPE
    # positions and the decode position offset (``rope_deltas``); None / 0 for plain text.
    embeds: Any = None
    position_ids: Any = None
    rope_delta: int = 0


@dataclass
class Event:
    handle: Any
    token: int | None
    logprob: dict | None = None
    finish: str | None = None
    cached_tokens: int = 0


@dataclass
class _Row:
    req: Request
    cache: list | None  # prefill caches (None once the row joined the batch)
    mtp_cache: list | None
    done: int = 0  # prompt tokens prefilled
    pending: int | None = None  # committed token not yet in the cache
    generated: int = 0
    drafts: list = field(default_factory=list)  # host ints of this step's window
    draft_dev: Any = None  # the next window's drafts: a (maybe lazy) int32 array
    nd: int = 0  # how many drafts ``draft_dev`` holds
    rates: list = field(default_factory=list)
    key: Any = None
    keyed: Any = None  # KeyedSampler of a sampled row that drafts
    base: int | None = None  # keyed-sampling seed constant (None: stateful key)
    context: list = field(default_factory=list)
    force: list = field(default_factory=list)  # the budget's next token
    finished: bool = False
    n: int = 0  # positions in the decode batch's KV (after joining)
    slot: int | None = None  # KV slot in the decode batch
    hslot: int | None = None  # KV slot in the MTP head's batch
    hn: int = 0  # positions absorbed by the MTP head (after joining)
    drafting: bool = False  # the row has an MTP head (prefill or batch)
    hit: int = 0  # prompt tokens restored from the APC prefix cache
    ckpts: list = field(default_factory=list)  # APC checkpoint lengths ahead
    salt: int = 0  # APC key of this row's cache layout
    apc: Any = None  # this row's APC coordinator (its checkpoint plan is per request)
    head_state: Any = None  # MTP head state of a just-joined row: its first draft


@dataclass
class _Item:
    kind: str  # "d": decode window, "p": prefill chunk
    row: _Row
    seg: Segment | None = None
    start: int = 0
    end: int = 0
    at: int = 0  # first packed / padded position of the row's tokens
    count: int = 0  # tokens of the row in the step


class RoundDriver:
    """Rows of one Qwen3.5-family target (projections converted to lane)."""

    def __init__(
        self,
        model: Any,
        *,
        drafter: Any = None,
        dflash_drafter: Any = None,
        plain_verify: bool = False,
        stop_tokens=None,
        chunk: int | None = None,
        apc: Any = None,
    ):
        from .mtp import MTPHead

        # Fixed prefill span per prompt (ROUND_PREFILL_CHUNK in vlm_engine.py). While rows
        # decode a prefill step is one span, so this is also the longest a
        # decoding row waits behind a prompt.
        self.chunk = int(chunk or CHUNK)
        self.idle_budget = max(IDLE_BUDGET, self.chunk)
        # Atom size by absolute position (so it depends on the prompt alone, and a
        # session extended from a stored checkpoint continues on the same grid):
        # the attention re-reads the whole KV per query block, so 512-token blocks
        # cost +7 s of a 38 s 32K prefill; beyond LONG_FROM blocks are LONG_CHUNK.
        self.long_from = LONG_FROM if self.chunk == CHUNK else 1 << 60
        self.long_chunk = LONG_CHUNK

        self.model = model
        self.lm = model.language_model if hasattr(model, "language_model") else model
        # Placeholder ids of image / video / audio tokens (APC never cuts or restores across them)
        self._media_ids: set[int] = set()
        config = getattr(model, "config", None)
        if config is not None:
            try:
                from mlx_vlm import apc as _apc

                self._media_ids = set(_apc.multimodal_token_ids_from_config(config))
            except Exception:  # a model without media tokens
                self._media_ids = set()
        self.stop_tokens = set(stop_tokens or ())
        self.head = (
            MTPHead(drafter, self.lm, lambda h: logits(self.lm, h))
            if drafter is not None
            else None
        )
        prev = mx.set_cache_limit(CACHE_LIMIT)
        if prev < CACHE_LIMIT:
            mx.set_cache_limit(prev)  # a tighter limit set by the host stays
        self.dflash = None
        if dflash_drafter is not None or (plain_verify and drafter is None):
            from .dflash import DFlashBatch

            self.dflash = DFlashBatch(model, dflash_drafter)
        self.batch: Any = (
            self.dflash if self.dflash is not None else DecodeBatch(self.lm)
        )
        # APC prefix cache (mlx_vlm ``APCManager``): a prompt starts from the
        # longest stored checkpoint (target KV + GDN state, plus the MTP
        # head's KV for rows that draft) and stores checkpoints at chunk
        # boundaries while it prefills.
        self.apc = None
        self._apc_manager = None
        if apc is not None:
            coordinator = self._new_coordinator(apc)
            if coordinator.enabled and coordinator.is_checkpoint:
                self.apc = coordinator
                self._apc_manager = apc
        self.rows: list[_Row] = []
        self._pending_join: list = []  # rows with their first token out, not yet in the decode batch
        self._pending_heads: dict = {}
        self._deferred_apcs: list = []  # coordinators holding captured, unflushed checkpoints
        self.cost = CostCurve()
        self.steps = 0
        self.drafted = 0
        self.accepted = 0
        self._prefilled_last = False
        self._decode_credit = 0.0
        self.chain_ms = 0.0  # measured drafting cost per chain depth
        self._chain_t = 0.0  # when the last chain was submitted
        self._chain_deep = 0  # its depth while it may still be running
        self._calls = 0
        self.decode_steps = 0  # telemetry since the batch last went idle
        self.step_s = 0.0
        self.gap_s = 0.0
        self._t_end = 0.0
        # per draft depth: drafted / landed (telemetry)
        self.depth_drafted = [0] * (MAX_DECODE_TOKENS - 1)
        self.depth_landed = [0] * (MAX_DECODE_TOKENS - 1)

    # ── rows ─────────────────────────────────────────────────────────────
    def add(self, req: Request) -> int:
        """Queue a row; returns the prompt tokens restored from the APC."""
        if len(req.ids) < 1:
            raise ValueError("round driver: empty prompt")
        self.flush_checkpoints()  # a lookup must see what earlier requests captured
        keyed = None
        if (
            req.sampling is not None
            and req.draft
            and not req.processors
            and (not req.logprobs or self.dflash is not None)
            and keyed_sampling.supports(req.sampling)
        ):
            # A sampled row drafts too (and draws the same tokens whether or
            # not it does): its token at generation index g is a
            # pure function of (logits, seed, g), so a draft is kept exactly
            # when serial sampling would have drawn it (keyed_sampling).
            seed = getattr(req.sampling, "seed", None)
            keyed = keyed_sampling.KeyedSampler(
                req.sampling,
                seed if seed is not None else int.from_bytes(os.urandom(8), "little"),
            )
        draft = bool(
            req.draft
            and self.can_draft
            and not req.processors
            and (req.sampling is None or keyed is not None)
        )
        row = _Row(
            req=req,
            cache=self.lm.make_cache(),
            mtp_cache=self.head.make_cache()
            if draft and self.head is not None
            else None,
            rates=[ACCEPT_PRIOR] * (MAX_DECODE_TOKENS - 1),
            context=list(req.ids),
            drafting=draft,
            keyed=keyed,
        )
        seed = getattr(req.sampling, "seed", None)
        if req.sampling is not None and keyed_sampling.supports(req.sampling):
            # Position-keyed draws, like the shared batch and the speculative lane.
            if seed is None:
                seed = int.from_bytes(os.urandom(8), "little")
            row.base = keyed_sampling.seed_base(seed)
        elif seed is not None:
            row.key = mx.random.key(int(seed))
        self._restore(row)
        vocab = self.head.vocab if self.head is not None and draft else None
        if vocab is not None:
            if self.rows:
                vocab.add_context(req.ids)
            else:
                vocab.set_context(req.ids)
        self.rows.append(row)
        return row.hit

    # ── APC prefix cache ─────────────────────────────────────────────────
    def _new_coordinator(self, manager: Any):
        """The manager's own coordinator (Yunshu's stores only the planned
        checkpoints, not every span cut) or upstream's for a stock manager."""
        bind = getattr(manager, "coordinator", None)
        if callable(bind):
            return bind(self.lm)
        from mlx_vlm.apc_coordinator import APCCoordinator

        return APCCoordinator(manager, self.lm)

    @property
    def can_draft(self) -> bool:
        return self.head is not None or (
            self.dflash is not None and self.dflash.can_draft
        )

    def will_draft(self, *, sampling, processors, logprobs, draft: bool) -> bool:
        """Whether a request with these fields gets an MTP head row (as ``add``)."""
        return bool(
            draft
            and self.can_draft
            and (
                sampling is None
                or (
                    not processors
                    and (not logprobs or self.dflash is not None)
                    and keyed_sampling.supports(sampling)
                )
            )
        )

    def apc_salt(self, extra_hash: int, drafting: bool) -> int:
        """APC key of this driver's cache layout. Entries carry the head's KV only
        for rows that draft and come from this driver's numerics, so they are kept
        apart from the upstream runner's entries; the prefill arithmetic (NAX /
        chunked GDN) is part of the key, a checkpoint is not read back by a run
        without those kernels."""
        from ..kernels import gdn_prefill, nax_prefill

        if self.dflash is not None:
            # DFlash rings are request-local proposal state, not target APC KV.
            drafting = False
        arith = gdn_prefill.kernel_id() + (
            nax_prefill.arithmetic_id() if nax_prefill.enabled() else "lane"
        )
        mix = f"{extra_hash}:round-driver:{int(drafting)}:{arith}"
        return (
            int.from_bytes(hashlib.blake2b(mix.encode(), digest_size=8).digest(), "big")
            >> 1
        )

    def _restore(self, row: _Row) -> None:
        """Start ``row`` from the longest stored checkpoint of its prompt and
        plan the checkpoints it stores on the way."""
        if self.apc is None or not row.req.use_apc:
            return
        ids = row.req.ids
        # one coordinator per row: its checkpoint plan is request state
        row.apc = self._new_coordinator(self._apc_manager)
        row.apc.media_checkpoint = row.req.embeds is not None
        row.salt = self.apc_salt(int(row.req.extra_hash), row.mtp_cache is not None)
        media = self._media_ids
        try:
            from mlx_vlm import apc as _apc

            # A media prompt restores only a prefix that leaves a text-only suffix (the media
            # placeholders, whose embeddings are computed per request, lie inside the cached
            # part), and no checkpoint cuts a media span (main: PromptProcessingBatch callbacks).
            hit = row.apc.lookup(
                ids,
                extra_hash=row.salt,
                safe_lookup_min=max(0, _apc.media_safe_prefix_min(ids, media) - 1),
                suffix_is_text_only=lambda n: _apc.prefix_leaves_text_only_suffix(
                    ids, n, media
                ),
                prefix_has_media=lambda n: _apc.prefix_contains_media_tokens(
                    ids, n, media
                ),
            )
            n = len(row.cache)
            want = n + (len(row.mtp_cache) if row.mtp_cache is not None else 0)
            warm = hit.get("warm_cache") if hit else None
            prefix = int(hit["prefix_len"]) if hit else 0
            if warm is not None and len(warm) == want and 0 < prefix < len(ids):
                row.cache = list(warm[:n])
                if row.mtp_cache is not None:
                    row.mtp_cache = list(warm[n:])
                row.done = row.hit = prefix
                with row.apc.manager.lock:
                    row.apc.manager.stats.restored_tokens += prefix
            row.apc.prepare_prefill(
                [len(ids)], prefill_step_size=self.chunk, prefix_lengths=[row.hit]
            )
            self._seed_window(row)
            floor = row.apc.manager.exact_cache_min_tokens
            row.ckpts = [
                c
                for c in row.apc.checkpoint_lengths(ids, media)
                if row.hit < c < len(ids) and c >= floor
            ]
        except Exception:
            logger.warning("APC lookup failed; prefilling from scratch", exc_info=True)
            if row.hit == 0:
                row.cache = self.lm.make_cache()
                if row.mtp_cache is not None:
                    row.mtp_cache = self.head.make_cache()
            row.ckpts = []

    def atom(self, pos: int) -> int:
        """Atom size of the prompt span that starts at ``pos``."""
        return self.long_chunk if pos >= self.long_from else self.chunk

    def _span_end(self, row: _Row, start: int) -> int:
        """End of the prompt span that starts at ``start``: the next multiple
        of ``chunk``, cut at the row's next APC checkpoint and at the prompt's
        end. The plan depends on the prompt alone (grid plus its
        checkpoints), so a row restored at a checkpoint on the grid continues
        with the spans a cold prefill would have run."""
        a = self.atom(start)
        end = min((start // a + 1) * a, len(row.req.ids))
        for c in row.ckpts:
            if start < c < end:
                end = c
                break
        return end

    def _run_end(self, row: _Row, start: int, budget: int) -> int:
        """End of the segment a prefill step runs for ``row`` from ``start``:
        consecutive atoms (as many as ``budget`` takes) become ONE segment, so a
        step runs large matmuls and a restored prompt's remainder is one forward
        (not one per grid cell). A run stops at a checkpoint (its state is stored
        there) and before the prompt's tail (the rows after the last 64-aligned
        boundary: the tail runs apart, on the tail kernels, as in a restored
        run). Prefill arithmetic is split-invariant at 64-aligned boundaries
        (nax matmul, BQ32 attention, chunked GDN), so the segment size never
        changes a prompt's output."""
        end = self._span_end(row, start)
        limit = (len(row.req.ids) - 1) // 64 * 64
        while end not in row.ckpts and end < limit:
            nxt = self._span_end(row, end)
            # a run keeps one atom size (the long-position atoms read the KV in
            # fewer, larger passes)
            if (
                nxt > limit
                or nxt - start > budget
                or self.atom(end) != self.atom(start)
            ):
                break
            end = nxt
        return end

    def _store_window(self, row: _Row, end: int) -> None:
        if self.dflash is None or not self.dflash.can_draft:
            return
        try:
            from .. import dflash_apc_window as win

            layers = self.dflash.window(row)
            if layers is None:
                return
            win.put(
                win.key_for(row.salt, row.req.ids[:end]),
                layers,
                sum(int(x.nbytes) for x in layers),
            )
        except Exception:  # an optimisation only: a miss leaves the suffix-only context
            logger.debug("drafter window store failed", exc_info=True)

    def _seed_window(self, row: _Row) -> None:
        if self.dflash is None or not self.dflash.can_draft or row.hit <= 0:
            return
        from .. import dflash_apc_window as win

        layers = win.get(win.key_for(row.salt, row.req.ids[: row.hit]))
        if layers is None:
            return
        self.dflash.seed(row, layers)
        logger.info(
            "Round driver: DFlash drafter window restored from APC hit: %d positions"
            " (prefix %d)",
            int(layers[0].shape[1]),
            row.hit,
        )

    def _store_checkpoint(self, row: _Row, end: int) -> None:
        if not row.req.cache_store:
            return  # X-Yunshu-Cache-Store: off: the plan and lookup stay an ordinary request's
        self._store_window(row, end)
        caches = list(row.cache)
        if row.mtp_cache is not None:
            caches += row.mtp_cache
        apc = row.apc
        # A lone row captures its checkpoint as lazy detached arrays and the copy is
        # evaluated by flush_checkpoints after its first token is out (the runner's
        # single-job protocol): the copy no longer sits on the first-token path.
        defer = len(self.rows) == 1 and hasattr(apc, "flush_deferred_checkpoints")
        try:
            if defer:
                apc.defer_checkpoint_stores = True
            apc.store_checkpoint(row.req.ids[:end], caches, extra_hash=row.salt)
            if defer and apc not in self._deferred_apcs:
                self._deferred_apcs.append(apc)
        except Exception:
            logger.warning("APC checkpoint store failed", exc_info=True)
        finally:
            if defer:
                apc.defer_checkpoint_stores = False

    def planned_checkpoints(self, handle: Any) -> tuple[int, list[int]] | None:
        """``(apc salt, checkpoint lengths this row's prefill will store)`` of the row owned by
        ``handle``, or None (the runner's prefix single-flight waits for them)."""
        for r in self.rows:
            if r.req.handle is handle and r.apc is not None:
                return r.salt, list(r.ckpts)
        return None

    def flush_checkpoints(self) -> None:
        """Evaluate and publish the captured checkpoints (after the step's tokens were emitted)."""
        pending, self._deferred_apcs = self._deferred_apcs, []
        for apc in pending:
            try:
                apc.flush_deferred_checkpoints()
            except Exception:
                logger.warning("APC deferred checkpoint flush failed", exc_info=True)

    def remove(self, handle: Any) -> None:
        gone = [r for r in self.rows if r.req.handle is handle]
        for r in gone:  # a cancelled prefill drops its captures, as the runner does
            apc = getattr(r, "apc", None)
            if apc is not None and apc in self._deferred_apcs:
                apc.discard_deferred_checkpoints()
                self._deferred_apcs.remove(apc)
        self.rows = [r for r in self.rows if r.req.handle is not handle]
        self._release(gone)
        if self.dflash is not None:
            self.dflash.leave(gone)

    def _release(self, gone: list[_Row]) -> None:
        """Drop rows' slots in the decode batch and the head's."""
        if len(gone) == 1 and not self.rows and gone[0].cache:
            # The last row leaves with its own K/V buffers (the DFlash batch decodes in the row's
            # caches): pool them for the next request's cache hit, as the runner's _finish does
            # (restore writes into a pooled buffer instead of allocating a context-sized one).
            from ..kernels import buffer_cache

            buffer_cache.offer_restore_buffers(gone[0].cache)
        if any(r.slot is not None for r in gone):
            self.batch.leave([r for r in gone if r.slot is not None])
        if self.head is not None and any(r.hslot is not None for r in gone):
            self.head.leave([r for r in gone if r.hslot is not None])

    def busy(self) -> bool:
        return bool(self.rows)

    # ── sampling ─────────────────────────────────────────────────────────
    def _processed(self, row: _Row, lg: mx.array, first: bool) -> mx.array:
        for p in row.req.processors:
            if not first and hasattr(p, "process_last_token"):
                lg = p.process_last_token(row.context[-1], lg)
            else:
                lg = p(mx.array(row.context), lg)
        return lg

    def _sample(self, row: _Row, logprobs: mx.array) -> mx.array:
        from mlx_lm.sample_utils import apply_min_p, apply_xtc

        p = row.req.sampling
        if p is None:
            return mx.argmax(logprobs, axis=-1)
        if row.base is not None:
            # row.generated is this draw's generation index: draws happen one per committed token
            return keyed_sampling.sample_rows(
                logprobs,
                p,
                mx.array(np.array([row.base], dtype=np.uint64)),
                mx.array([row.generated]),
            )
        x = logprobs
        if p.top_p < 1.0:
            x = top_p_filter(x, max(p.top_p, 0.0))
        if p.min_p:
            x = apply_min_p(x, p.min_p)
        if p.xtc_probability > 0.0:
            x = apply_xtc(
                x, p.xtc_probability, p.xtc_threshold, list(p.xtc_special_tokens or [])
            )
        x = top_k_filter(x, p.top_k)
        x = x * (1 / p.temperature)
        if row.key is not None:
            row.key, sub = mx.random.split(row.key)
            return mx.random.categorical(x, key=sub)
        return mx.random.categorical(x)

    def _draw(self, items: list[_Item], hidden: mx.array, target_logits=None):
        """Logits at every decode window position and the last position of a
        prompt finished this step; the row's token(s) at each. Returns
        ``[(item, token ids array, logprob arrays or None)]``."""
        pick, outs = [], []
        for it in items:
            if it.kind == "d":
                outs.append((it, len(pick), it.count))
                pick.extend(range(it.at, it.at + it.count))
            elif it.end == len(it.row.req.ids):
                outs.append((it, len(pick), 1))
                pick.append(it.at + it.count - 1)
        if not pick:
            return []
        indices = mx.array(pick, dtype=mx.int32)
        lg = (
            logits(self.lm, hidden[indices])
            if target_logits is None
            else target_logits[indices]
        )
        # Serial greedy decoding samples ``logits - logsumexp(logits)`` in the logits
        # dtype, so its argmax runs over rounded log-probabilities: two adjacent bf16
        # logits can land on one value and the tie goes to the lower id. Take the same
        # argmax so a verified row picks the token serial decode would (omlx 26375259).
        greedy = mx.argmax(lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1)
        draws: list[tuple[_Item, Any, Any]] = []
        for it, off, n in outs:
            row = it.row
            req = row.req
            if req.guide is not None:
                from ..tool_call_grammar import apply_bitmask

                mask = req.guide.mask()
                if mask is not None:
                    # This row's slice only; other packed distributions stay intact.
                    masked = apply_bitmask(lg[off : off + n], mask)
                    lg = mx.concatenate([lg[:off], masked, lg[off + n :]], axis=0)
                    greedy = mx.argmax(
                        lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1
                    )
            if row.force:
                # the thinking budget's token at this position, not a sample
                draws.append((it, mx.array(row.force[:1], dtype=mx.int32), None))
                row.force = row.force[1:]
                continue
            if req.sampling is None and not req.processors and not req.logprobs:
                draws.append((it, greedy[off : off + n], None))
                continue
            if row.keyed is not None:
                # window position j is generation index generated + j
                x = lg[off : off + n]
                if self.dflash is None or req.logprobs:
                    x = x.astype(mx.float32)
                lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
                tok = row.keyed.sample_positions(
                    lp, list(range(row.generated, row.generated + n))
                )
                extra = None
                if req.logprobs:
                    extra = [lp[0, tok[0]]]
                    if req.top_logprobs > 0:
                        extra += list(top_logprobs_device(lp, req.top_logprobs))
                draws.append((it, tok, extra))
                continue
            x = self._processed(row, lg[off : off + 1], first=it.kind == "p")
            if req.logprobs or (row.base is not None and self.dflash is None):
                # DFlash matches the shared batch: retain bf16 normalization
                # unless reporting logprobs explicitly requests float32.
                x = x.astype(mx.float32)
            lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
            tok = self._sample(row, lp)
            extra = None
            if req.logprobs:
                extra = [lp[0, tok[0]]]
                if req.top_logprobs > 0:
                    extra += list(top_logprobs_device(lp, req.top_logprobs))
            draws.append((it, tok, extra))
        return draws

    # ── one step ─────────────────────────────────────────────────────────
    def _finish_join(self) -> None:
        """Move rows whose prompt finished into the decode batch (deferred from the prefill
        step so their first token is not delayed by it)."""
        pending, self._pending_join = self._pending_join, []
        heads, self._pending_heads = self._pending_heads, {}
        live = [r for r in pending if r in self.rows and not r.finished]
        if not live:
            return
        self.batch.join(live)
        arrays = list(self.batch.arrays())
        ready = [r for r in live if r.drafting] if self.head is not None else []
        if ready:
            self.head.join(ready)
            arrays += self.head.slots.arrays()
            # The first draft waits for the next decode step: the row's first token is out
            # already, and drafting (a chain of head forwards) would only delay it.
            for r in ready:
                r.head_state = heads[id(r)]
                arrays.append(r.head_state)
        mx.eval(*arrays)

    def step(self) -> list[Event]:
        self.flush_checkpoints()  # the previous step's tokens have been handed out
        self._finish_join()
        if not self.rows:
            return []
        if self.dflash is not None:
            from ..mrope import clear_rope_state

            clear_rope_state(self.model)
        if self._will_decode():
            self._prefilled_last = False
            t0 = time.perf_counter()
            if self._t_end:
                self.gap_s += t0 - self._t_end  # the host's time between steps
            out = self._decode_step()
            self._t_end = time.perf_counter()
            self.step_s += self._t_end - t0
            if self.dflash is not None:
                self._decode_credit = max(0.0, self._decode_credit - (self._t_end - t0))
            self.decode_steps += 1
            if not self.rows:
                logger.info(
                    "round driver idle: %d decode steps, %.1f ms in a step, "
                    "%.1f ms between steps",
                    self.decode_steps,
                    self.step_s * 1e3 / self.decode_steps,
                    self.gap_s * 1e3 / self.decode_steps,
                )
                self.decode_steps, self.step_s, self.gap_s, self._t_end = (
                    0,
                    0.0,
                    0.0,
                    0.0,
                )
            return out
        self._t_end = 0.0
        self._prefilled_last = True
        decoding = bool(self.batch.rows)
        prefill_started = time.perf_counter()
        waiting = [r for r in self.rows if r.pending is None]
        # Shortest remainder first: a restored prompt (a few rows to run) would
        # otherwise ride behind a whole 2048-row span of a cold one (~1.9 s on the
        # 27B); it finishes in tens of milliseconds, so it goes ahead.
        short = [r for r in waiting if len(r.req.ids) - r.done <= SHORT_FIRST]
        out = self._prefill_step(short or waiting, decoding)
        # A checkpoint ends a row's span for the step (the snapshot is taken from
        # the cache after it), so a prompt's last atoms arrive one step apart and
        # its first token waited a whole multi-row step per atom (8 x 1K: 8.3 s for
        # a prefill that ended at 4.7 s). Rows within a chunk of their end run
        # those atoms now: the same spans, so the same bits.
        for _ in range(FINISH_STEPS):
            near = [
                r
                for r in self.rows
                if r.pending is None
                and r.done > 0
                and 0 < len(r.req.ids) - r.done <= self.atom(r.done)
            ]
            if not near:
                break
            out += self._prefill_step(near, decoding)
        if self.dflash is not None:
            self._credit_prefill(time.perf_counter() - prefill_started, decoding)
        return out

    def _will_decode(self) -> bool:
        """Whether the next step is a decode step: rows decode and either no
        prompt waits or the last step was a prefill."""
        if not self.batch.rows:
            return False
        return (
            self._prefilled_last
            or (self.dflash is not None and self._decode_credit > 0)
            or all(r.pending is not None for r in self.rows)
        )

    def _credit_prefill(self, elapsed: float, was_decoding: bool) -> None:
        """The shared scheduler policy, measured in GPU service time, not rounds.

        Credit is replaced after each prefill episode, never accumulated. A
        newly joined first row gets the existing bounded first-token burst.
        """
        live = self.batch.rows
        if not live:
            self._decode_credit = 0.0
            return
        early = any(r.generated < DECODE_EARLY_TOKENS for r in live)
        credit = decode_quantum(elapsed, early)
        self._decode_credit = (
            credit if was_decoding else min(credit, DECODE_FIRST_BURST_S)
        )

    @staticmethod
    def _preempt_event(items: list[_Item]):
        """The cancel event of a lone background (rehearsal) request: its prefill may stop
        between layer groups when the event is set (kernels/preempt.py, as in the runner's
        ``_next_preemptible``). None for any other step."""
        # one row (a step may hold several of its segments: a span and the prompt's tail)
        if len({id(it.row) for it in items}) != 1 or not preempt.supported():
            return None
        event = getattr(items[0].row.req.handle, "cancel_event", None)
        return (
            event
            if event is not None and getattr(event, "preemptible", False)
            else None
        )

    def _preempted(self, items: list[_Item], stop) -> list[Event]:
        """A preemptible prefill was cancelled inside a span: nothing of the partial span is
        kept; checkpoints captured by its completed spans are published, then the rows leave."""
        from .. import next_turn

        self.flush_checkpoints()
        for it in items:
            self.remove(it.row.req.handle)
        logger.info(
            "background prefill preempted at layer group %d (after %d layers)",
            stop.group,
            stop.layers,
        )
        next_turn.record("preempted")
        return []

    def _replay_key(self, it: _Item):
        """Replay key of a prompt's final block, or None when ``it`` is not one."""
        row = it.row
        ids = row.req.ids
        # An MTP head row replays too: the head's cache already holds the restored
        # prefix and its tail is re-absorbed below from the stored hidden rows (a
        # one-layer forward, as on a real step), so only the target forward is skipped.
        if row.finished or not replay.eligible(it.start, it.end, len(ids)):
            return None
        from .. import dflash_apc_window

        return (
            id(self.lm),
            self.dflash is not None,
            dflash_apc_window.key_for(row.salt, ids),
        )

    def _record_tail(self, items: list[_Item], hidden, captures) -> None:
        """Keep the final block a real forward produced for the next identical prompt."""
        for it in items:
            key = self._replay_key(it)
            if key is None or it.seg is None:
                continue
            try:
                replay.store(
                    key,
                    it.row.cache,
                    it.start,
                    it.end,
                    hidden[it.at : it.at + it.count],
                    [c[it.at : it.at + it.count] for c in captures],
                )
            except Exception:  # an optimisation only: the next repeat runs the forward
                logger.debug("round driver replay store failed", exc_info=True)
                replay.clear()

    def _prefill_step(self, waiting: list[_Row], decoding: bool) -> list[Event]:
        items: list[_Item] = []
        budget = self.chunk if decoding else self.idle_budget
        at = 0
        blocked = False
        for r in waiting:
            ids = r.req.ids
            if blocked and len(ids) - r.done > budget:
                # Oldest prompt first: a long prompt that cannot finish in
                # this step is not interleaved with younger long prompts
                # (that would delay every first token to the last one's);
                # prompts that fit in what is left of the budget still ride along.
                continue
            while budget > 0 and r.done < len(ids):
                start = r.done
                end = self._run_end(r, start, budget)
                chunk = mx.array(ids[start:end], dtype=mx.int32)
                items.append(
                    _Item(
                        "p",
                        r,
                        Segment(
                            r.cache,
                            chunk,
                            self.chunk,
                            self.long_from,
                            self.long_chunk,
                            embeds=None
                            if r.req.embeds is None
                            else r.req.embeds[start:end],
                            position_ids=None
                            if r.req.position_ids is None
                            else r.req.position_ids[:, start:end],
                        ),
                        start,
                        end,
                        at,
                        end - start,
                    )
                )
                r.done = end
                at += end - start
                budget -= end - start
                if end in r.ckpts:
                    break  # the checkpoint is stored after this span
            if r.done < len(ids):
                blocked = True
        if not items:
            return []
        captures: list[mx.array] = []
        entry = None
        tail_key = None
        if len(items) == 1 and items[0].seg is not None:
            tail_key = self._replay_key(items[0])
            if tail_key is not None:
                entry = replay.lookup(
                    tail_key, items[0].row.cache, items[0].start, items[0].end
                )
        if entry is not None:
            replay.restore(entry, tail_key, items[0].row.cache)
            hidden = entry.hidden
            captures = list(entry.captures)
        else:
            ev = self._preempt_event(items)
            try:
                with preempt.scope(ev) if ev is not None else contextlib.nullcontext():
                    # Every segment runs the engine's own prefill forward (nax matmul, prefill
                    # kernels, chunked GDN under prefill_scope), as the lane does: a segment's
                    # bits are what it would be alone in a step, so one forward per segment
                    # keeps the arithmetic of the packed form. The prompt's KV is reserved once.
                    parts: list[mx.array] = []
                    per_seg: list[list[mx.array]] = []
                    for it in items:
                        if it.seg is None:
                            continue
                        reserve_prompt_kv(it.row.cache, len(it.row.req.ids))
                        got: list[mx.array] = []
                        parts.append(
                            lane_forward(
                                self.lm,
                                it.seg,
                                capture_ids=self.dflash.capture_ids
                                if self.dflash
                                else (),
                                captured=got if self.dflash else None,
                            )
                        )
                        per_seg.append(got)
                    hidden = (
                        parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
                    )
                    if self.dflash is not None and per_seg:
                        captures = [
                            mx.concatenate([g[layer] for g in per_seg], axis=0)
                            for layer in range(len(per_seg[0]))
                        ]
            except preempt.PreemptedError as stop:
                return self._preempted(items, stop)
        draws = self._draw(items, hidden)
        # Evaluate every cache the step advanced, not only what feeds a token:
        # a prompt chunk that emits nothing would otherwise stay a lazy graph
        # on top of the previous chunk's, and a long prompt becomes one graph
        # holding every chunk's KV buffer version until its last chunk.
        advanced = {id(it.row): it.row for it in items}
        mx.eval(
            *[d[1] for d in draws],
            *[a for d in draws if d[2] for a in d[2]],
            *[a for r in advanced.values() for a in cache_buffers(r.cache)],
        )
        if entry is None:
            self._record_tail(items, hidden, captures)
        events: list[Event] = []
        for it, tok, extra in draws:
            self._emit(it.row, [int(tok.tolist()[0])], extra, events)
        if self.dflash is not None and captures:
            self.dflash.absorb(items, captures)
        # prompt chunks of draftable rows feed their MTP head, in order
        absorb: dict[int, tuple] = {}
        for it in items:
            row = it.row
            if row.mtp_cache is None or row.finished:
                continue
            ids = row.req.ids
            nxt = ids[it.start + 1 : it.end]
            nxt.append(ids[it.end] if it.end < len(ids) else row.pending)
            entry = absorb.setdefault(id(row), (row, [], [], []))
            entry[1].extend(nxt)
            entry[2].append(hidden[it.at : it.at + it.count])
            if row.req.position_ids is not None:
                entry[3].append(row.req.position_ids[:, it.start : it.end])
        live = list(absorb.values())
        heads = {}
        if live:
            outs = self.head.absorb_prompt(
                [
                    (r, t, mx.concatenate(h, axis=0))
                    + ((mx.concatenate(pos, axis=1),) if pos else ())
                    for r, t, h, pos in live
                ]
            )
            heads = {id(e[0]): o for e, o in zip(live, outs, strict=True)}
        for it in items:
            if it.end in it.row.ckpts and not it.row.finished:
                self._store_checkpoint(it.row, it.end)
        gone = [r for r in self.rows if r.finished]
        self.rows = [r for r in self.rows if not r.finished]
        # a request that ends at its first token (a rehearsal, max_tokens 1) leaves from here: its
        # K/V buffers go to the restore pool like those of a row that decoded to its end
        self._release(gone)
        joined = [
            r
            for r in advanced.values()
            if r.pending is not None and not r.finished and r.done == len(r.req.ids)
        ]
        for r in joined:
            r.n = len(r.req.ids)
        # Joining the decode batch (slot copy, the drafter's context ring: a drafter forward over
        # the window) is not needed for the first token, which is out below: it runs at the start
        # of the next step, after the caller has handed the token on.
        self._pending_join.extend(joined)
        self._pending_heads.update(
            {id(r): heads[id(r)] for r in joined if id(r) in heads}
        )
        arrays = [a for r in self.rows if r.cache for a in cache_buffers(r.cache)]
        arrays += [
            a for r in self.rows if r.mtp_cache for a in cache_buffers(r.mtp_cache)
        ]
        arrays += list(self._pending_heads.values())
        mx.eval(*arrays)
        self.steps += 1
        return events

    def _decode_step(self) -> list[Event]:
        if self.dflash is not None:
            return self.dflash.step(self)
        started = time.perf_counter()
        late = [r for r in self.batch.rows if r.head_state is not None]
        if late:
            self._draft(late, mx.stack([r.head_state for r in late]))
            for r in late:
                r.head_state = None
        rows = list(self.batch.rows)
        # The drafts are still a lazy graph (the previous step queued the
        # chain and returned before it ran): the window's tokens are built on
        # the device, so this step's graph is built while the chain runs.
        lens, parts, used_drafts = [], [], []
        for r in rows:
            use = r.nd if r.draft_dev is not None and not r.force else 0
            parts.append(mx.array([r.pending], dtype=mx.int32))
            if use:
                parts.append(r.draft_dev)
            lens.append(1 + use)
            used_drafts.append(use)
        hidden = self.batch.forward(lens, mx.concatenate(parts))
        items, at = [], 0
        for r, n in zip(rows, lens, strict=True):
            items.append(_Item("d", r, None, 0, 0, at, n))
            at += n
        draws = self._draw(items, hidden)

        # Early absorb: the head reads (target token, hidden) at every window
        # position inside this step's graph, and its greedy readout is the
        # row's next first draft whichever position the row keeps. The host
        # only picks which position, so drafting the next window starts at
        # once instead of after a second dependent pass.
        head_rows = [(b, r) for b, r in enumerate(rows) if r.hslot is not None]
        early_out: mx.array | None = None
        early_seeds: mx.array | None = None
        head = self.head
        if head_rows and head is not None:
            T = max(lens)
            idx = mx.array(
                [
                    [items[b].at + min(j, lens[b] - 1) for j in range(T)]
                    for b, _ in head_rows
                ],
                dtype=mx.int32,
            )
            flat = mx.concatenate([d[1] for d in draws]).astype(mx.int32)
            early_out, early_seeds = head.absorb_window(
                [r for _, r in head_rows], flat[idx], hidden[idx]
            )
        mx.async_eval(
            *[d[1] for d in draws],
            *[a for d in draws if d[2] for a in d[2]],
            *self.batch.arrays(),
            *(
                [early_out, early_seeds, *head.slots.arrays()]
                if early_out is not None and early_seeds is not None and head
                else []
            ),
        )
        window_tokens: list[Any] = [d[1].tolist() for d in draws]  # waits for the step
        seed_vals: Any = early_seeds.tolist() if early_seeds is not None else []
        # forward cost: from when the GPU could start it (the chain before it
        # is not part of the forward) to now
        gpu_free = self._chain_t + self.chain_ms * self._chain_deep / 1e3
        ms = (time.perf_counter() - max(started, gpu_free)) * 1e3
        if ms > 0:
            self.cost.observe(at, ms)
        self._chain_deep = 0

        events: list[Event] = []
        used_all: list[int] = []
        for (it, _tok, extra), toks, use in zip(
            draws, window_tokens, used_drafts, strict=True
        ):
            row = it.row
            window = [row.pending, *(row.draft_dev.tolist() if use else [])]
            keep = 1
            while keep < len(window) and window[keep] == toks[keep - 1]:
                keep += 1
            if len(window) > 1:
                self._observe(row, len(window) - 1, keep - 1)
            # a stop, the length limit or the thinking budget can cut the
            # window short: the cache keeps what was emitted
            used = self._emit(row, toks[:keep], extra, events)
            used_all.append(used)
        self.batch.commit(used_all)
        gone = [r for r in rows if r.finished]
        self.rows = [r for r in self.rows if not r.finished]
        self._release(gone)
        if early_out is not None:
            alive = [(i, b, r) for i, (b, r) in enumerate(head_rows) if not r.finished]
            for _, b, r in alive:
                r.hn += used_all[b]  # the positions it kept stay in the head's KV
            if alive:
                at_last = [used_all[b] - 1 for _, b, _ in alive]
                heads = early_out[
                    mx.array([i for i, _, _ in alive], dtype=mx.int32),
                    mx.array(at_last, dtype=mx.int32),
                ]
                first = mx.array(
                    [
                        seed_vals[i][u]
                        for (i, _, _), u in zip(alive, at_last, strict=True)
                    ],
                    dtype=mx.int32,
                )
                self._draft([r for _, _, r in alive], heads, first)
        self.steps += 1
        return events

    def _draft(
        self,
        rows: list[_Row],
        heads: mx.array,
        first: mx.array | None = None,
        sync: bool = False,
    ) -> None:
        """Queue the next window's drafts for ``rows``. The chain is submitted
        to the GPU here (async) and the caller returns to build the next
        step's graph while it runs; ``sync`` (or every 16th call, to keep the
        chain's cost measured) waits for it."""
        assert self.head is not None
        depths = self._depths(rows)
        drafted_at = time.perf_counter()
        drafts = self.head.draft(rows, heads, depths, first)
        deepest = max(depths, default=0)
        for r, d, n in zip(rows, drafts, depths, strict=True):
            r.draft_dev, r.nd, r.drafts = d, n, []
            self.drafted += n
        self._chain_deep = 0
        if deepest:
            self._calls += 1
            live = [d for d, n in zip(drafts, depths, strict=True) if n]
            if sync or self._calls % 16 == 1:
                mx.eval(*live, *self.head.slots.arrays())
                ms = (time.perf_counter() - drafted_at) * 1e3 / deepest
                self.chain_ms = (
                    ms
                    if not self.chain_ms
                    else self.chain_ms + ACCEPT_EMA * (ms - self.chain_ms)
                )
            else:
                mx.async_eval(*live, *self.head.slots.arrays())
                self._chain_deep = deepest
        self._chain_t = time.perf_counter()

    def _emit(self, row: _Row, committed: list[int], extra, out: list) -> int:
        """Emit ``committed`` in order until a stop, the length limit or the
        thinking budget cuts it; returns how many tokens were kept."""
        lp = None
        if extra is not None:
            lp = {
                "token_id": committed[0],
                "logprob": float(extra[0].item()),
                "top_logprobs": [],
            }
            if len(extra) > 1:
                lp["top_logprobs"] = [
                    {"token_id": int(t), "logprob": float(v)}
                    for t, v in zip(
                        extra[1][0].tolist(), extra[2][0].tolist(), strict=True
                    )
                ]
        row.drafts, row.draft_dev, row.nd = [], None, 0
        used = 0
        for t in committed:
            used += 1
            row.generated += 1
            row.context.append(t)
            guide_cut = False
            if row.req.guide is not None:
                guide_cut = row.req.guide.arms(t)
                row.req.guide.feed(t)
            finish = None
            if t in self.stop_tokens:
                finish = "stop"
            elif row.generated >= row.req.max_tokens:
                finish = "length"
            out.append(Event(row.req.handle, t, lp, finish))
            lp = None
            if finish is not None:
                row.finished = True
                break
            budget = row.req.budget
            if budget is not None:
                budget(t)
                forced = budget.pop_forced_token_id()
                if forced is not None:
                    # the budget writes the next position: later drafts go
                    row.force = [int(forced)]
                    break
            if guide_cut:
                break
        row.pending = committed[used - 1]
        if (
            self.head is not None
            and self.head.vocab is not None
            and row.drafting
            and len(self.rows) == 1
        ):
            self.head.vocab.learn(committed[:used])
        return used

    # ── drafting policy ──────────────────────────────────────────────────
    def _observe(self, row: _Row, drafted: int, landed: int) -> None:
        self.accepted += landed
        st = getattr(row.req.handle, "stats", None)
        if st is not None:
            st.spec_drafted += drafted
            st.spec_rounds += 1
            st.spec_accepted += landed
            from ..spec_metrics import observe

            observe(st, landed, drafted)
        for j in range(drafted):
            self.depth_drafted[j] += 1
            self.depth_landed[j] += int(j < landed)
        for j in range(min(drafted, landed + 1)):
            hit = 1.0 if j < landed else 0.0
            row.rates[j] += ACCEPT_EMA * (hit - row.rates[j])

    def _depths(self, ready: list[_Row]) -> list[int]:
        """Next step's drafts per ready row (cost-aware allocation over the
        packed rows of the step)."""
        if not ready:
            return []
        fixed = len(self.batch.rows)
        probs = []
        for r in ready:
            room = min(MAX_DECODE_TOKENS - 1, r.req.max_tokens - r.generated - 1)
            probs.append(chain(r.rates, room) if room > 0 else [])
        return allocate(fixed, probs, self.cost, LANE_ROWS, self.chain_ms, padded=True)


__all__ = ["CHUNK", "Event", "Request", "RoundDriver", "cache_buffers"]
