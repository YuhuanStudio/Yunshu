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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import numpy as np

from .. import fn_ttft, hit_stages, keyed_sampling, ple_prefetch
from ..cache_prefill import reserve_prompt_kv
from ..constraint_eos import GrammarDeadEndError
from ..kernels import preempt
from ..keyed_sampling import top_k_filter, top_p_filter
from ..prompt_ids import ids_array
from ..serving.work_scheduler import (
    DECODE_EARLY_TOKENS,
    DECODE_FIRST_BURST_S,
    decode_quantum,
)
from ..text_logprobs import top_logprobs_device
from . import flashnext, replay
from .allocate import CostCurve, allocate, chain
from .batch import MAX_WINDOW, DecodeBatch
from .forward import Segment
from .layer import FlashNextLayer, GenericLayer, PackedLayer

logger = logging.getLogger(__name__)

MAX_DECODE_TOKENS = MAX_WINDOW  # pending token + up to 7 drafts
FINISH_STEPS = (
    4  # extra prefill steps a step may run for rows within a chunk of their end
)
LONG_FROM = 16384  # positions from here on prefill in LONG_CHUNK atoms
LONG_CHUNK = 2048
CHUNK = 512  # default prefill chunk: fixed spans from the prompt start
SHORT_FIRST = 512  # a prompt with at most this many rows left runs before longer ones
# A prompt this much shorter than the longest waiting one also goes first (up to SHORT_JUMP_MAX
# rows): 1K prompts no longer wait out a whole 128K prefill (real-1104 mixed cell: 207-243 s).
SHORT_JUMP_RATIO = 8
SHORT_JUMP_MAX = 8192
SHORT_STREAK_MAX = (
    16  # after this many prefill steps of jumping prompts the oldest gets one
)
IDLE_BUDGET = 2048  # prefill tokens per prefill step when no row decodes
ACCEPT_PRIOR = 0.7  # per-depth draft acceptance before a row has history
ACCEPT_EMA = 0.15
COST_WARM_STEPS = 3  # decode steps a row runs before its step times feed the cost curve
LANE_ROWS = 128  # rows one lane-matmul call keeps row-invariant
# MLX's freed-buffer cache: long prompts allocate transient buffers of ever
# new sizes (KV growth, attention scores), and left unbounded the cache grew
# to ~95 GiB in a 4 x 32K run, after which every step took 5-10x as long
# (docs/guides/ROUND_DRIVER.md). The driver keeps it under this bound.
CACHE_LIMIT = 8 * 2**30


def _clear_verify_state() -> None:
    """Drop the verify kernels' per-forward group-sum tables (thread-local, keyed by array).

    The packed / Flash-Next layers register every GDN block output there; only the tree-verify
    paths used to clear it, so each decode step of a driver row left its 48 layers' outputs (and
    the Metal buffers behind them) alive: ~1600 arrays per second for the life of the server."""
    from ..kernels.omlx import qwen35_verify_qmm as q35
    from ..kernels.omlx_q4 import qwen35_verify_qmm as q4

    q35.clear_group_sums()
    q4.clear_group_sums()


HEAD_LAZY_MAX = 512 << 20  # prompt HC bytes a row's queued head prefill may retain


def cache_buffers(cache: list) -> list:
    """A row's cache arrays (KV buffers, GDN conv / recurrent state)."""
    out = []
    for c in cache:
        if getattr(c, "keys", None) is not None:
            out += [c.keys, c.values]
            if getattr(c, "k_scales", None) is not None:  # int8 row cache
                out += [c.k_scales, c.v_scales]
            for name in ("index_keys", "index_position_ids", "index_block_keys"):
                a = getattr(c, name, None)  # Flash-Next QSA indexer state
                if isinstance(a, mx.array):
                    out.append(a)
        elif getattr(c, "caches", None) is not None:  # a CacheList of sub-caches
            out += cache_buffers(list(c.caches))
        else:
            out += [a for a in getattr(c, "cache", ()) if isinstance(a, mx.array)]
    return out


def static_mask_only(processors) -> bool:
    """Every processor is a static token ban (``suppress_tokens`` / ``ignore_eos``): it never needs the
    token history or the logits, so a drafting row applies it to every verify row (``pure_at``, the same
    elementwise mask a plain row applies) and keeps its speculation."""
    return bool(processors) and all(
        getattr(p, "is_static", False) is True and hasattr(p, "pure_at")
        for p in processors
    )


def draft_penalty(processors):
    """The frequency / presence penalty of a row that keeps drafting with one: the processors are static
    bans plus one ``GeneratedTokenPenalty`` (the non-thinking default ``presence_penalty`` 1.5 of the Qwen
    cards). A drafting row applies it to every window row over the window's own tokens (the serial
    ``window_row`` arithmetic, as the speculative lane does) and counts the tokens a round kept; anything
    else returns None."""
    from ..text_logit_processors import GeneratedTokenPenalty

    pens = [p for p in processors or () if isinstance(p, GeneratedTokenPenalty)]
    rest = [p for p in processors or () if not isinstance(p, GeneratedTokenPenalty)]
    if len(pens) != 1 or (rest and not static_mask_only(rest)):
        return None
    return pens[0]


def _mlx_closure(p) -> bool:
    return getattr(p, "__module__", "").startswith("mlx_lm.sample_utils")


def windowable(processors) -> bool:
    """Every processor can be applied to one row of a speculative window with nothing but that
    row's position and path: a pure mask (``pure_at``: token masks with min_tokens / top-n-sigma,
    logit bias), at most one ``GeneratedTokenPenalty`` (``window_row``), or an mlx-lm closure
    over the last tokens (repetition penalty)."""
    penalties = 0
    for p in processors or ():
        if hasattr(p, "pure_at"):
            continue
        if hasattr(p, "window_row"):
            penalties += 1
            continue
        if _mlx_closure(p):
            continue
        return False
    return penalties <= 1


def generic_processors(
    processors, guided: bool = False, logprobs: bool = False
) -> bool:
    """Processors the row applies one window row at a time (``window_processed``): windowable,
    and not served by the device fast paths (static bans; one penalty over static bans,
    unguided, unreported). A reporting row always takes this path: the serial step applies the
    processors in the logits dtype and only then promotes to fp32 for the report."""
    if not processors or not windowable(processors):
        return False
    if guided or logprobs:
        return True
    return not static_mask_only(processors) and draft_penalty(processors) is None


def _row_path(tokens, parents, j) -> list:
    """Window tokens on the root-to-``j`` path (root first, ``j`` included)."""
    if parents is None:
        return list(tokens[: j + 1])
    chain, at = [], j
    while at >= 0:
        chain.append(tokens[at])
        at = parents[at]
    return chain[::-1]


def window_processed(
    processors, logits: mx.array, context, generated: int, tokens=None, parents=None
) -> mx.array:
    """The request's processors on every row of a window, in list order, each row exactly as the
    serial step that draws it: ``logits`` [n, V]; ``context`` the row's token context (ending
    with the window root); ``generated`` the draw index of row 0; ``tokens`` the window's tokens
    (root first) and ``parents`` its tree (None: a chain), or ``tokens`` None for the prompt's
    first draw (nothing fed yet). Row j sees the committed context plus its path, and a stateful
    token mask counts ``generated + depth`` draws, as ``process_last_token`` would have."""
    from ..constrained_spec import PENALTY_CONTEXT

    n = int(logits.shape[0])
    rows = []
    for j in range(n):
        path = _row_path(tokens, parents, j) if tokens is not None else []
        depth = max(len(path) - 1, 0)
        row = logits[j : j + 1]
        for p in processors:
            if hasattr(p, "pure_at"):
                row = p.pure_at(generated + depth, row)
            elif hasattr(p, "window_row"):
                row = p.window_row(path, row)
            else:
                history = [*context[-PENALTY_CONTEXT:], *path[1:]][-PENALTY_CONTEXT:]
                row = p(mx.array(history, dtype=mx.int32), row)
        rows.append(row)
    return rows[0] if n == 1 else mx.concatenate(rows, axis=0)


class _PerRow(list):
    """Reported logprobs of an n-row window: entry j is row j's ``[logprob, *top]``."""


def row_logprobs(x: mx.array) -> mx.array:
    """Log-probabilities of the rows of ``x`` [n, V], each through its own ``[1, V]`` reduction.

    A verify window of n rows and n plain steps must report the same bits: the sum inside
    ``logsumexp`` is one fixed reduction per row only when it is called on one row, so a window's
    rows are never reduced as one ``[n, V]`` block (``constrained_spec.normalize_rows``'s rule)."""
    n = int(x.shape[0])
    if n == 1:
        return x - mx.logsumexp(x, axis=-1, keepdims=True)
    return mx.concatenate(
        [
            x[j : j + 1] - mx.logsumexp(x[j : j + 1], axis=-1, keepdims=True)
            for j in range(n)
        ],
        axis=0,
    )


def _report(req, lp: mx.array, tok: mx.array):
    """The logprob report of a draw over the rows of ``lp`` [n, V]: ``[lp, (ids, values)]`` of the
    one row of a plain step, a ``_PerRow`` of them for a window (an MTP chain)."""
    if not req.logprobs:
        return None
    n = int(lp.shape[0])

    def one(j: int) -> list:
        extra = [lp[j, tok[j]]]
        if req.top_logprobs > 0:
            extra += list(top_logprobs_device(lp[j : j + 1], req.top_logprobs))
        return extra

    return one(0) if n == 1 else _PerRow(one(j) for j in range(n))


def window_penalty(penalty, logits: mx.array, window: mx.array) -> mx.array:
    """``penalty`` on a chain window's rows ``logits`` [n, V]: row j counts the committed tokens plus
    ``window[: j + 1]`` (the tokens fed up to it), the counts ``window_row`` builds, then the same
    elementwise ``_apply`` on each row."""
    vocab = int(logits.shape[-1])
    tokens = window.reshape(-1).astype(mx.int32)
    n = int(tokens.shape[0])
    valid = ((tokens >= 0) & (tokens < vocab)).astype(mx.float32)
    hits = (
        mx.zeros((n, vocab), dtype=mx.float32)
        .at[mx.arange(n), mx.clip(tokens, 0, vocab - 1)]
        .add(valid)
    )
    counts = penalty._counts(vocab)[None] + mx.cumsum(hits, axis=0)
    return cast(mx.array, penalty._apply(counts, logits))


# eq=False: a request and a row are entities, compared by identity. The generated field-wise
# __eq__ made `row in self.rows` compare prompt kwargs holding mx arrays (pixel_values, audio
# features), which raised as soon as two media rows were live (2026-11-10, OCR A/B: 500s).
@dataclass(eq=False)
class Request:
    """What a row needs from the caller (the runner's job)."""

    ids: list[int]
    max_tokens: int
    sampling: Any = None  # RowParams (vlm_batch_runner) or None: greedy
    processors: list = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int = 0
    prompt_logprobs: int | None = None
    draft: bool = True  # greedy without processors / logprobs may draft
    guide: Any = None  # invariant tool-call guide, advanced on emitted tokens
    budget: Any = None  # mlx_vlm ThinkingBudgetCriteria (forces "\n</think>")
    handle: Any = None
    extra_hash: int = 0  # APC salt (model / processor identity)
    use_apc: bool = True  # the runner's capacity check for this prompt
    cache_store: bool = True  # False: look the prefix up, store nothing
    # Explicit cache seams (Anthropic ``cache_control``, rendered prompt-cache endpoints): the APC
    # coordinator's per-request policy (lookup points, checkpoint writes); None: automatic checkpoints.
    cache_plan: Any = None
    # Media prompt: [L, D] prompt embeddings (visual features scattered in), [3, L] mRoPE
    # positions and the decode position offset (``rope_deltas``); None / 0 for plain text.
    embeds: Any = None
    position_ids: Any = None
    rope_delta: int = 0
    # Generic layer only: the model's prompt inputs (``generic.PromptFeatures`` input: the runner's
    # ``get_input_embeddings(...).to_dict()`` or ``prepare_media`` output, dict of arrays).
    prompt_kwargs: Any = None
    media: bool = (
        False  # the prompt carries media (APC checkpoints take the media rules)
    )
    # n-gram / suffix speculation (generic layer, greedy rows; opt-in: per-request spec_decode or
    # YUNSHU_NGRAM_DEFAULT). Ignored where the row cannot take it (``RoundDriver.ngram_ok``).
    ngram: bool = False


@dataclass
class Event:
    handle: Any
    token: int | None
    logprob: dict | None = None
    finish: str | None = None
    cached_tokens: int = 0
    prompt_logprobs: list | None = None


@dataclass(eq=False)
class _Row:
    req: Request
    cache: list | None  # prefill caches (None once the row joined the batch)
    mtp_cache: list | None
    tm: Any = None  # fn_ttft stage timer (cold Flash-Next requests)
    head_lazy: int = (
        0  # bytes of prompt HC states the head's queued (unevaluated) prefill holds
    )
    prompt_scores: list = field(default_factory=lambda: [None])
    done: int = 0  # prompt tokens prefilled
    pending: int | None = None  # committed token not yet in the cache
    generated: int = 0
    drafts: list = field(default_factory=list)  # host ints of this step's window
    draft_dev: Any = None  # the next window's drafts: a (maybe lazy) int32 array
    forced_windows: int = 0
    ngram_logged: bool = False
    ngram: Any = None  # NgramDrafter of a greedy generic-layer row that asked for n-gram speculation
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
    head_cache: list | None = (
        None  # the head's KV cut out of its slots as the row leaves finished
    )
    drafting: bool = False  # the row has an MTP head (prefill or batch)
    hit: int = 0  # prompt tokens restored from the APC prefix cache
    ckpts: list = field(default_factory=list)  # APC checkpoint lengths ahead
    salt: int = 0  # APC key of this row's cache layout
    apc: Any = None  # this row's APC coordinator (its checkpoint plan is per request)
    head_state: Any = None  # MTP head state of a just-joined row: its first draft
    penalty: Any = (
        None  # GeneratedTokenPenalty of a drafting / keyed row (applied per window row)
    )
    generic: bool = False  # processors applied per window row by ``window_processed``
    features: Any = None  # generic layer: ``generic.PromptFeatures`` of the prompt
    state: Any = None  # generic layer: ``generic.RowState`` (the model's per-row state)
    ahead_logged: bool = False
    ahead: Any = (
        None  # lone generic row: the next step's token, already queued on the device
    )


@dataclass
class _Item:
    kind: str  # "d": decode window, "p": prefill chunk
    row: _Row
    seg: Segment | None = None
    start: int = 0
    end: int = 0
    at: int = 0  # first packed / padded position of the row's tokens
    count: int = 0  # tokens of the row in the step
    tokens: Any = None  # a decode window's tokens (pending + drafts); None: just the pending token


def _prefix(ids, n: int):
    """The first ``n`` ids as a view of the prompt's int32 array (one conversion per prompt)."""
    arr = ids_array(ids)
    return arr[:n] if arr is not None else ids[:n]


def round_readback(
    targets: list[mx.array],
    rows: list,
    used_drafts: list[int],
    *,
    read: Callable[[mx.array], list[int]] | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """One host transfer for target windows and the drafts compared with them.

    The head's seed table stays on the device; callers select its last kept
    position there. Packing integer IDs adds no floating-point arithmetic.
    """
    arrays = [t.reshape(-1).astype(mx.int32) for t in targets]
    sizes = [int(t.size) for t in arrays]
    draft_arrays = [
        r.draft_dev[:n] for r, n in zip(rows, used_drafts, strict=True) if n
    ]
    packed = mx.concatenate([*arrays, *draft_arrays])
    values = cast(list[int], packed.tolist()) if read is None else read(packed)
    windows, drafts, at = [], [], 0
    for n in sizes:
        windows.append(values[at : at + n])
        at += n
    for n in used_drafts:
        drafts.append(values[at : at + n])
        at += n
    return windows, drafts


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
        kv_format: str = "bf16",
        max_draft: int | None = None,
        generic_layer: bool = False,
        row_drafters: Any = None,
        layer: Any = None,
    ):
        from ..kernels.gdn_qk_eps import install_lossless_default

        install_lossless_default()
        # Fixed prefill span per prompt (ROUND_PREFILL_CHUNK in vlm_engine.py). While rows
        # decode a prefill step is one span, so this is also the longest a
        # decoding row waits behind a prompt.
        self.chunk = int(chunk or CHUNK)
        self.idle_budget = max(IDLE_BUDGET, self.chunk)
        self._short_streak = 0
        # Atom size by absolute position (so it depends on the prompt alone, and a
        # session extended from a stored checkpoint continues on the same grid):
        # the attention re-reads the whole KV per query block, so 512-token blocks
        # cost +7 s of a 38 s 32K prefill; beyond LONG_FROM blocks are LONG_CHUNK.
        self.long_from = LONG_FROM if self.chunk == CHUNK else 1 << 60
        self.long_chunk = LONG_CHUNK

        # called with the last finished row before its K/V goes to the restore pool (the runner
        # stores the reply's decode cache there)
        self.finish_hook: Any = None
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
        # the wrapper's own placeholder ids (a backbone config may omit them: Gemma 4)
        from ..vlm_batch_runner import media_token_ids

        with contextlib.suppress(
            Exception
        ):  # a model object without a config (unit fakes)
            self._media_ids |= set(media_token_ids(model))
        self.stop_tokens = set(stop_tokens or ())
        # The model layer (round_driver/layer.py) owns the family's head, DFlash batch, default draft width
        # and decode set. ``generic_layer=`` builds the same layer the driver always built.
        if generic_layer and layer is not None:
            raise ValueError("round driver: pass layer= or generic_layer=, not both")
        if layer is None:
            if generic_layer:
                layer = GenericLayer(self.lm)
            elif flashnext.is_flash_next(self.lm):
                layer = FlashNextLayer(self.lm)
            else:
                layer = PackedLayer(self.lm)
        if layer.lm is not self.lm:
            raise ValueError(
                "round driver: the layer is bound to another language model"
            )
        self.layer = layer
        self.flash = isinstance(layer, FlashNextLayer)
        # Generic layer: the model's own upstream forward, one call per row (``generic.py``); no drafter,
        # logits instead of hidden states. Mutually exclusive with the two packed model layers.
        self.generic = isinstance(layer, GenericLayer)
        # draft LM / Gemma-4 assistant drafters of the generic layer (``draftlm.RowDrafters``); n-gram otherwise
        self.row_drafters = row_drafters if self.generic else None
        if self.generic and (
            self.flash or drafter is not None or dflash_drafter is not None
        ):
            raise ValueError(
                "round driver: the generic layer takes no drafter and no Flash-Next model"
            )
        prev = mx.set_cache_limit(CACHE_LIMIT)
        if prev < CACHE_LIMIT:
            mx.set_cache_limit(prev)  # a tighter limit set by the host stays
        drafters = layer.drafters(
            model,
            drafter=drafter,
            dflash_drafter=dflash_drafter,
            plain_verify=plain_verify,
            kv_format=kv_format,
        )
        self.head: Any = drafters.head
        self.dflash: Any = drafters.dflash
        # drafts per round: the Qwen3.5 window is up to MAX_WINDOW - 1 positions, a Flash-Next window is the
        # pending token plus one draft (the packed verify's chain length; block 2 decodes fastest there)
        self.max_draft = int(max_draft if max_draft is not None else drafters.max_draft)
        self.kv_format = kv_format
        # the drafter batch (DFlash rows own their step) replaces the layer decode set when there is one
        self.batch: Any = layer.decode_set(kv_format, self.dflash)
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
        self._lazy_ids: dict[
            int, bool
        ] = {}  # row id -> its head prefill was left queued
        self._deferred_apcs: list = []  # coordinators holding captured, unflushed checkpoints
        self.cost = CostCurve()
        self._ngram_untrimmable_logged = False
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

    def _logits(self, hidden: mx.array) -> mx.array:
        """Logits of the target's hidden rows (Flash-Next: HC states through the mixer and the head)."""
        return self.layer.logits(hidden)

    def _new_cache(self) -> list:
        return self.layer.new_cache()

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
            and self._processors_draft(req.processors, req.logprobs, req.guide)
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
            and self._processors_draft(req.processors, req.logprobs, req.guide)
            and (req.sampling is None or keyed is not None)
        )
        ngram = self.ngram_ok(req)
        row = _Row(
            req=req,
            cache=self._new_cache(),
            mtp_cache=self.head.make_cache()
            if draft and self.head is not None
            else None,
            rates=[ACCEPT_PRIOR] * (MAX_DECODE_TOKENS - 1),
            context=list(req.ids),
            drafting=draft,
            keyed=keyed,
            # a keyed row draws in the window branch (no ``_processed``) whether or not it drafts
            penalty=draft_penalty(req.processors)
            if draft or keyed is not None or ngram
            else None,
        )
        if ngram:
            from .. import settings
            from .ngram import NgramDrafter

            row.ngram = (
                self.row_drafters.make() if self.row_drafters is not None else None
            ) or NgramDrafter(settings.get("YUNSHU_SPEC_PROPOSER"))
            st = getattr(req.handle, "stats", None)
            if st is not None:
                st.spec_mode = row.ngram.mode
        if (draft or keyed is not None or ngram) and generic_processors(
            req.processors, req.guide is not None, bool(req.logprobs)
        ):
            row.generic = True
            row.penalty = next(
                (p for p in req.processors if hasattr(p, "window_row")), None
            )
        seed = getattr(req.sampling, "seed", None)
        if req.sampling is not None and keyed_sampling.supports(req.sampling):
            # Position-keyed draws, like the shared batch and the speculative lane.
            if seed is None:
                seed = int.from_bytes(os.urandom(8), "little")
            row.base = keyed_sampling.seed_base(seed)
        elif seed is not None:
            row.key = mx.random.key(int(seed))
        self.layer.admit(row, req)
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

    def ngram_ok(self, req: Request) -> bool:
        """Whether ``req`` takes the model-free n-gram / suffix drafter: the generic layer only (packed
        layers keep their batch-invariant tree verify), a greedy row without guide, thinking budget or
        logprobs, with windowable processors."""
        return bool(
            req.ngram
            and self.generic
            and req.sampling is None
            and req.guide is None
            and req.budget is None
            and not req.logprobs
            and req.prompt_logprobs is None
            and windowable(req.processors)
        )

    def _ngram_window(self, row: _Row) -> list[int]:
        """The drafts of this step's window for an n-gram row. Only a lone decoding row speculates:
        rejected positions are cut off the cache by ``trim`` (free on one row; rows of a batch cache
        would each need their own cut). Caches that cannot trim (rotating, recurrent) decode plain."""
        if row.ngram is None or row.force:
            return []
        if len(self.batch.rows) != 1:
            return []
        if row.ngram.needs_target_cache:
            row.ngram.cache = self.batch.cache
        if not self.batch.can_trim(row.ngram.k + 1):
            if not self._ngram_untrimmable_logged:
                self._ngram_untrimmable_logged = True
                logger.info(
                    "Round driver n-gram speculation off: the model's cache cannot trim"
                )
            return []
        room = min(row.ngram.k, row.req.max_tokens - row.generated - 1)
        drafts: list[int] = row.ngram.propose(row.context, room)
        if drafts and not row.ngram_logged:
            row.ngram_logged = True
            logger.info(
                "Round driver %s window engaged: %d drafts", row.ngram.name, len(drafts)
            )
        return drafts

    def _processors_draft(self, processors, logprobs=False, guide=None) -> bool:
        """A row with these processors may draft: none, static token bans (every verify row gets the
        same ban, DFlash and MTP rows alike), or bans plus one frequency / presence penalty without
        logprobs or a guide (``draft_penalty``, as the speculative lane's ``tree_penalty``)."""
        # pure masks, repetition penalty, one frequency / presence penalty, in any order, with
        # or without a guide or logprobs: applied one window row at a time
        # (``window_processed``; fast device paths for static bans and one penalty)
        return windowable(processors)

    def will_draft(self, *, sampling, processors, logprobs, draft: bool) -> bool:
        """Whether a request with these fields gets an MTP head row (as ``add``)."""
        return bool(
            draft
            and self.can_draft
            and (
                sampling is None
                or (
                    self._processors_draft(processors, logprobs)
                    and keyed_sampling.supports(sampling)
                )
            )
        )

    def apc_salt(self, extra_hash: int, drafting: bool, ngram: bool = False) -> int:
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
        arith += self.layer.apc_suffix()
        if ngram:
            # T>1 windows round differently from T=1 on stock kernels: their replies stay out of the plain namespace
            arith += ":ngram"
        mix = f"{extra_hash}:round-driver:{int(drafting)}:{arith}"
        return (
            int.from_bytes(hashlib.blake2b(mix.encode(), digest_size=8).digest(), "big")
            >> 1
        )

    def _restore(self, row: _Row) -> None:
        """Start ``row`` from the longest stored checkpoint of its prompt and
        plan the checkpoints it stores on the way."""
        if (
            self.apc is None
            or not row.req.use_apc
            or row.req.prompt_logprobs is not None
        ):
            return
        ids = row.req.ids
        # one coordinator per row: its checkpoint plan is request state
        row.apc = self._new_coordinator(self._apc_manager)
        row.apc.media_checkpoint = row.req.embeds is not None or row.req.media
        if row.req.cache_plan is not None:
            # the policy steers this coordinator's lookup and checkpoint plan (it is per row)
            row.apc.set_request(ids, row.req.cache_plan)
        row.salt = self.apc_salt(
            int(row.req.extra_hash), row.mtp_cache is not None, row.ngram is not None
        )
        # A prompt with no media token (the usual one) needs none of the media checks, each of which
        # walks every prompt id in Python (~34K ids at 32K context, several times per lookup): one
        # C-level set test decides, and an empty media set makes every check trivially true / empty.
        media = self._media_ids if not self._media_ids.isdisjoint(ids) else frozenset()
        try:
            from mlx_vlm import apc as _apc

            # A media prompt restores only a prefix that leaves a text-only suffix (the media
            # placeholders, whose embeddings are computed per request, lie inside the cached
            # part), and no checkpoint cuts a media span (main: PromptProcessingBatch callbacks).
            t_lookup = time.perf_counter()
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
            hit_stages.add("lookup", time.perf_counter() - t_lookup)
            assert row.cache is not None
            n = len(row.cache)
            want = n + (len(row.mtp_cache) if row.mtp_cache is not None else 0)
            warm = hit.get("warm_cache") if hit else None
            prefix = int(hit["prefix_len"]) if hit else 0
            if warm is not None and len(warm) == want and 0 < prefix < len(ids):
                from ..kernels import int8_kv

                row.cache = [
                    int8_kv.to_bf16_cache(c) if int8_kv.is_int8_cache(c) else c
                    for c in warm[:n]
                ]
                if row.mtp_cache is not None:
                    row.mtp_cache = list(warm[n:])
                row.done = row.hit = prefix
                with row.apc.manager.lock:
                    row.apc.manager.stats.restored_tokens += prefix
            row.apc.prepare_prefill(
                [len(ids)], prefill_step_size=self.chunk, prefix_lengths=[row.hit]
            )
            with hit_stages.stage("window"):
                self._seed_window(row)
            t_plan = time.perf_counter()
            floor = row.apc.manager.exact_cache_min_tokens
            row.ckpts = [
                c
                for c in row.apc.checkpoint_lengths(ids, media)
                if row.hit < c < len(ids) and c >= floor
            ]
            hit_stages.add("plan", time.perf_counter() - t_plan)
        except Exception:
            logger.warning("APC lookup failed; prefilling from scratch", exc_info=True)
            if row.hit == 0:
                row.cache = self._new_cache()
                if row.mtp_cache is not None:
                    row.mtp_cache = self.head.make_cache()
            row.ckpts = []

    def atom(self, pos: int) -> int:
        """Atom size of the prompt span that starts at ``pos``."""
        return self.long_chunk if pos >= self.long_from else self.chunk

    def _store_window(self, row: _Row, end: int) -> None:
        if self.dflash is None or not self.dflash.can_draft:
            return
        try:
            from .. import dflash_apc_window as win

            layers = self.dflash.window(row)
            if layers is None:
                return
            win.put(
                win.key_for(row.salt, _prefix(row.req.ids, end)),
                layers,
                sum(int(x.nbytes) for x in layers),
            )
        except Exception:  # an optimisation only: a miss leaves the suffix-only context
            logger.debug("drafter window store failed", exc_info=True)

    def _seed_window(self, row: _Row) -> None:
        if self.dflash is None or not self.dflash.can_draft or row.hit <= 0:
            return
        from .. import dflash_apc_window as win

        layers = win.get(win.key_for(row.salt, _prefix(row.req.ids, row.hit)))
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
        assert row.cache is not None
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
        t0 = time.perf_counter()
        try:
            self._flush_pending(pending)
        finally:
            if pending:
                logger.info(
                    "round driver: deferred checkpoints flushed in %.1f ms (%d apc), "
                    "active %.0f MiB cache %.0f MiB, wall %.3f",
                    (time.perf_counter() - t0) * 1e3,
                    len(pending),
                    mx.get_active_memory() / 2**20,
                    mx.get_cache_memory() / 2**20,
                    time.time() % 1000,
                )

    def _flush_pending(self, pending: list) -> None:
        for apc in pending:
            try:
                apc.flush_deferred_checkpoints()
            except Exception:
                logger.warning("APC deferred checkpoint flush failed", exc_info=True)

    def remove(self, handle: Any) -> None:
        self._drop_ahead()
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
        if gone and self.head is not None and any(r.generated for r in gone):
            # what the draft-depth allocator believed when this reply ended: per-width step cost,
            # per-depth chain cost and the rows' acceptance (compare with the measured tok/s)
            logger.info(
                "Round driver draft cost: chain %.2f ms/depth, steps %s, rates %s",
                self.chain_ms,
                {k: round(v, 1) for k, v in sorted(self.cost.points.items())},
                [round(x, 2) for x in gone[0].rates[: self.max_draft]],
            )
        for r in gone:
            ng = getattr(r, "ngram", None)
            if ng is not None and ng.windows:
                logger.info(
                    "Round driver %s row: %d windows, %d drafted, %d accepted, %d tokens",
                    ng.name,
                    ng.windows,
                    ng.drafted,
                    ng.landed,
                    r.generated,
                )
        if (
            self.generic
            and len(gone) == 1
            and not self.rows
            and gone[0].slot is not None
            and gone[0].finished
            and gone[0].apc is not None
            and self.finish_hook is not None
        ):
            # the last row leaves a batch cache: cut its own cache out so the reply can be stored
            self.batch.restore(gone[0])
        if (
            type(self.batch) is DecodeBatch
            and len(gone) == 1
            and not self.rows
            and gone[0].slot is not None
            and gone[0].finished
            and gone[0].apc is not None
            and self.finish_hook is not None
            and gone[0].req.use_apc
            and gone[0].req.cache_store
            and len(gone[0].context) > len(gone[0].req.ids)
        ):
            # the packed batch decodes in shared slots: cut the row's own caches (and its head's) out
            # before they are released, so the reply is stored like a DFlash-batch row's
            row = gone[0]
            try:
                self.batch.restore(row)
                if self.head is not None and row.drafting and row.hslot is not None:
                    row.head_cache = self.head.restore(row)
            except Exception:
                logger.warning("decode cache cut-out failed", exc_info=True)
                row.cache = None
        if len(gone) == 1 and not self.rows and gone[0].cache:
            # The last row leaves with its own K/V buffers (the DFlash batch decodes in the row's
            # caches): pool them for the next request's cache hit, as the runner's _finish does
            # (restore writes into a pooled buffer instead of allocating a context-sized one).
            from ..kernels import buffer_cache

            row = gone[0]
            held = False
            if row.finished and self.finish_hook is not None:
                # the reply's cache is complete here: the runner holds it for the next turn and
                # offers the buffers itself once stored
                if self.dflash is not None:
                    self.dflash.finish_tail(row)
                try:
                    held = bool(self.finish_hook(row))
                except Exception:
                    logger.warning("decode cache store failed", exc_info=True)
            if not held:
                buffer_cache.offer_restore_buffers(row.cache)
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

    def _draw(self, items: list[_Item], hidden: mx.array | None, target_logits=None):
        """Logits at every decode window position and the last position of a
        prompt finished this step; the row's token(s) at each. Returns
        ``[(item, token ids array, logprob arrays or None)]``."""
        pick: list[int] = []
        outs = []
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
        assert (
            hidden is not None or target_logits is not None
        )  # a picked row has a hidden row
        lg = (
            self._logits(cast(mx.array, hidden)[indices])
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
            host = None  # the window's tokens (root first) on the host, when a row needs them
            if it.kind == "d" and (row.generic or req.guide is not None):
                host = (
                    [row.pending]
                    if it.tokens is None
                    else [int(t) for t in it.tokens.reshape(-1).tolist()]
                )
            if req.guide is not None:
                from ..tool_call_grammar import apply_bitmask

                # a window's position j is masked by the guide state after its path: ``plan``
                try:
                    mask = (
                        req.guide.plan(host[1:], n)
                        if host is not None and n > 1
                        else req.guide.mask()
                    )
                except GrammarDeadEndError as exc:
                    # this row's grammar has no continuation: record it on the guide (the runner fails
                    # this one request: ``_row_fault``) and keep the step going for the other rows
                    req.guide.fault = exc
                    mask = None
                if mask is not None:
                    # This row's slice only; other packed distributions stay intact.
                    masked = apply_bitmask(lg[off : off + n], mask)
                    lg = mx.concatenate([lg[:off], masked, lg[off + n :]], axis=0)
                    greedy = mx.argmax(
                        lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1
                    )
            if row.generic:
                # the request's processors on every window row, in list order, after the guide's
                # mask (a mask is -inf: the order only matters to top-n-sigma, and the plain row
                # and the window take the same one)
                masked = window_processed(
                    req.processors,
                    lg[off : off + n],
                    row.context,
                    row.generated,
                    host,
                    None,
                )
                lg = mx.concatenate([lg[:off], masked, lg[off + n :]], axis=0)
                greedy = mx.argmax(
                    lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1
                )
            if req.processors and not row.generic and static_mask_only(req.processors):
                # static bans: every window row gets the mask a plain row applies (``pure_at``), so a
                # drafting row verifies against the same distributions
                masked = lg[off : off + n]
                for p in req.processors:
                    masked = p.pure_at(row.generated, masked)
                lg = mx.concatenate([lg[:off], masked, lg[off + n :]], axis=0)
                greedy = mx.argmax(
                    lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1
                )
            if row.penalty is not None and not row.generic and it.kind == "d":
                # a drafting row's penalty: every window row over the tokens fed up to it (a prompt's
                # last position has nothing generated yet, as the serial first call)
                window = (
                    it.tokens
                    if it.tokens is not None
                    else mx.array([row.pending], dtype=mx.int32)
                )
                masked = window_penalty(row.penalty, lg[off : off + n], window)
                lg = mx.concatenate([lg[:off], masked, lg[off + n :]], axis=0)
                greedy = mx.argmax(
                    lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1
                )
            if row.force:
                # the thinking budget's token at this position, not a sample
                draws.append((it, mx.array(row.force[:1], dtype=mx.int32), None))
                row.force = row.force[1:]
                continue
            if (
                req.sampling is None
                and (
                    not req.processors
                    or static_mask_only(req.processors)
                    or row.penalty is not None
                    or row.generic
                )
                and not req.logprobs
            ):
                draws.append((it, greedy[off : off + n], None))
                continue
            if row.keyed is not None:
                # window position j is generation index generated + j
                x = lg[off : off + n]
                if self.dflash is None or req.logprobs:
                    x = x.astype(mx.float32)
                lp = row_logprobs(x)
                tok = row.keyed.sample_positions(
                    lp, list(range(row.generated, row.generated + n))
                )
                draws.append((it, tok, _report(req, lp, tok)))
                continue
            if row.generic or not req.processors:
                # already processed above (or nothing to apply): every row of the window
                x = lg[off : off + n]
            else:
                x = self._processed(row, lg[off : off + 1], first=it.kind == "p")
            if req.logprobs or (row.base is not None and self.dflash is None):
                # DFlash matches the shared batch: retain bf16 normalization
                # unless reporting logprobs explicitly requests float32.
                x = x.astype(mx.float32)
            lp = row_logprobs(x)
            tok = self._sample(row, lp)
            draws.append((it, tok, _report(req, lp, tok)))
        return draws

    # ── one step ─────────────────────────────────────────────────────────
    def _finish_join(self) -> None:
        """Move rows whose prompt finished into the decode batch (deferred from the prefill
        step so their first token is not delayed by it)."""
        t_join = time.perf_counter()
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
        logger.info(
            "round driver: %d row(s) joined the decode batch in %.1f ms, wall %.3f",
            len(live),
            (time.perf_counter() - t_join) * 1e3,
            time.time() % 1000,
        )

    def step(self) -> list[Event]:
        try:
            return self._step_impl()
        finally:
            _clear_verify_state()

    def _step_impl(self) -> list[Event]:
        # The previous step's tokens have been handed out: publish the captured checkpoints. While a
        # row is still prefilling nothing has been handed out yet and the capture stays lazy (the
        # runner flushes after the first token, as for the batch path): flushing at every prefill
        # step start put a 32K checkpoint copy (~45 ms) in the middle of a rehearsal's prefill.
        if all(r.pending is not None for r in self.rows):
            self.flush_checkpoints()
        if any(r.ahead is not None for r in self.rows) and not (
            len(self.rows) == 1 and self.rows[0].pending is not None
        ):
            return (
                self._settle_ahead()
            )  # a second row arrives: this step finishes first
        self._finish_join()
        if not self.rows:
            return []
        self.layer.begin_step(self.model, self.dflash is not None)
        if self._will_decode():
            self._prefilled_last = False
            t0 = time.perf_counter()
            if self._t_end:
                self.gap_s += t0 - self._t_end  # the host's time between steps
            out = self._decode_step()
            self._t_end = time.perf_counter()
            self.step_s += self._t_end - t0
            if self.layer.caps.fn_trace and self.decode_steps < 3:
                logger.info(
                    "round driver: decode step %d of the burst took %.1f ms",
                    self.decode_steps + 1,
                    (self._t_end - t0) * 1e3,
                )
            if self.dflash is not None:
                self._decode_credit = max(0.0, self._decode_credit - (self._t_end - t0))
            self.decode_steps += 1
            if not self.rows:
                logger.info(
                    "round driver idle: %d decode steps (%d pipelined), %.1f ms in a step, "
                    "%.1f ms between steps",
                    self.decode_steps,
                    self.ahead_steps,
                    self.step_s * 1e3 / self.decode_steps,
                    self.gap_s * 1e3 / self.decode_steps,
                )
                self.ahead_steps = 0
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
        out = self._prefill_step(self._short_first(waiting), decoding)
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

    def _short_first(self, waiting: list) -> list:
        """Prompts much shorter than the longest waiting one run first, shortest first; the
        split-invariant prefill keeps every row's bits whatever order the spans run in."""

        def left(r):
            return len(r.req.ids) - r.done

        longest = max((left(r) for r in waiting), default=0)
        limit = max(SHORT_FIRST, min(SHORT_JUMP_MAX, longest // SHORT_JUMP_RATIO))
        short = sorted((r for r in waiting if left(r) <= limit), key=left)
        if not short or len(short) == len(waiting):
            self._short_streak = 0
            return short or waiting
        self._short_streak += 1
        if self._short_streak > SHORT_STREAK_MAX:
            self._short_streak = 0
            return waiting
        return short

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
        if (
            not self.layer.caps.replay
            or row.req.prompt_logprobs is not None
            or row.finished
            or not replay.eligible(it.start, it.end, len(ids))
        ):
            return None
        from .. import dflash_apc_window

        return (
            id(self.lm),
            self.dflash is not None,
            dflash_apc_window.key_for(row.salt, ids),
        )

    def _head_lazy(self, row: _Row, advanced: dict, hidden, items: list) -> bool:
        """Keep ``row``'s head prefill queued past this step's first token (Flash-Next rows,
        bounded by HEAD_LAZY_MAX of retained prompt HC states); False evaluates it now."""
        if not self.layer.caps.lazy_head or hidden is None or id(row) not in advanced:
            return False
        row.head_lazy += sum(
            it.count * int(hidden.shape[-1]) * hidden.dtype.size
            for it in items
            if it.row is row
        )
        lazy = row.head_lazy <= HEAD_LAZY_MAX
        if not lazy:
            row.head_lazy = 0
        self._lazy_ids[id(row)] = lazy
        return lazy

    def _span_reason(self, row: _Row, start: int, end: int) -> str:
        """Why the prompt span ``start:end`` ends where it does (grid / checkpoint / tail)."""
        n = len(row.req.ids)
        atom = self.atom(start)
        on_grid = end % atom == 0
        if end == n:
            return "tail" + ("+checkpoint" if end in row.ckpts else "")
        if end in row.ckpts:
            return "grid+checkpoint" if on_grid else "checkpoint-cut"
        return "grid" if on_grid else "other"

    def _ple_lookahead(self, pf, items: list, budget: int) -> None:
        """Start reading the PLE rows of the spans this prompt prefills next (a wrong guess costs
        a synchronous read of the missed rows, never a different value)."""
        last = items[-1]
        row, ids = last.row, last.row.req.ids
        if last.seg is None or row.done >= len(ids):
            pf.cancel()
            return
        start = row.done
        for _ in range(ple_prefetch.MAX_PENDING):
            if start >= len(ids):
                break
            end = self.layer.span_end(
                row, start, max(budget, self.chunk), atom=self.atom
            )
            pf.submit_span(ids, start, end)
            start = end

    def _log_spans(self, items: list, span_t: list, wait_s: float) -> None:
        """One INFO line per Flash-Next prompt span of a request that is not already cached: where it
        starts and ends, why it was cut, the host time spent building its graph and the step's eval
        wait (shared by the step's spans). Logging only."""
        for i, it in enumerate(items):
            if it.seg is None:
                continue
            logger.info(
                "fn_span path=driver ids=%d span=%d:%d len=%d cut=%s submit=%.1f wait=%.1f ms"
                " (step of %d span(s))",
                len(it.row.req.ids),
                it.start,
                it.end,
                it.end - it.start,
                self._span_reason(it.row, it.start, it.end),
                (span_t[i + 2] - span_t[i + 1]) * 1e3,
                wait_s * 1e3,
                len(items),
            )

    def _record_tail(self, items: list[_Item], hidden, captures) -> None:
        """Keep the final block a real forward produced for the next identical prompt."""
        for it in items:
            key = self._replay_key(it)
            if key is None or it.seg is None:
                continue
            assert it.row.cache is not None
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
                end = self.layer.span_end(
                    r,
                    start,
                    budget,
                    atom=self.atom,
                    scoring=r.req.prompt_logprobs is not None,
                )
                assert r.cache is not None
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
        tm = fn_ttft.Timer()
        tm.segs = len(items)
        span_t: list[float] = [time.perf_counter()]
        captures: list[mx.array] = []
        entry = None
        tail_key = None
        if len(items) == 1 and items[0].seg is not None:
            tail_key = self._replay_key(items[0])
            if tail_key is not None:
                assert items[0].row.cache is not None
                entry = replay.lookup(
                    tail_key, items[0].row.cache, items[0].start, items[0].end
                )
        hidden: mx.array | None = None
        if entry is not None:
            assert items[0].row.cache is not None
            replay.restore(entry, tail_key, items[0].row.cache)
            hidden = entry.hidden
            captures = list(entry.captures)
        else:
            ev = self._preempt_event(items)
            # Flash-Next spans read their PLE rows from SSD: the following spans' rows are read on a
            # background thread while this step runs on the GPU (None when the table is in memory)
            pf = ple_prefetch.attach(self.lm) if self.flash else None
            try:
                with preempt.scope(ev) if ev is not None else contextlib.nullcontext():
                    # Every segment runs the engine's own prefill forward (nax matmul, prefill
                    # kernels, chunked GDN under prefill_scope), as the lane does: a segment's
                    # bits are what it would be alone in a step, so one forward per segment
                    # keeps the arithmetic of the packed form. The prompt's KV is reserved once.
                    parts: list[mx.array] = []
                    per_seg: list[list[mx.array]] = []
                    for it in items:
                        span_t.append(
                            time.perf_counter()
                        )  # host time of the previous item
                        if it.seg is None:
                            continue
                        reserve_prompt_kv(it.row.cache, len(it.row.req.ids))
                        if pf is not None:
                            pf.begin(len(it.row.req.ids), it.start, it.end)
                        got: list[mx.array] = []
                        out = self.layer.prefill(
                            it.row,
                            it.seg,
                            it.start,
                            it.end,
                            capture_ids=self.dflash.capture_ids if self.dflash else (),
                            captured=got if self.dflash else None,
                        )
                        if out is not None:
                            parts.append(out)
                        if (
                            self.layer.caps.per_token_rows
                            and it.row.req.prompt_logprobs is not None
                        ):
                            from .prompt_logprobs import score_span

                            h = parts[-1]
                            for offset in range(0, it.count, 32):
                                score_span(
                                    it.row,
                                    it.start + offset,
                                    self._logits(h[offset : offset + 32]),
                                )
                        per_seg.append(got)
                    if pf is not None:
                        self._ple_lookahead(pf, items, budget)
                    span_t.append(time.perf_counter())
                    if not self.layer.caps.per_token_rows:
                        # rows of ``hidden`` are the logits of the prompts finished this step: an item
                        # owns one row if it ends its prompt, none otherwise
                        run = 0
                        for it in items:
                            ends = int(it.end == len(it.row.req.ids))
                            it.at, it.count = run, ends
                            run += ends
                    hidden = (
                        None
                        if not parts
                        else parts[0]
                        if len(parts) == 1
                        else mx.concatenate(parts, axis=0)
                    )
                    if self.dflash is not None and per_seg:
                        captures = [
                            mx.concatenate([g[layer] for g in per_seg], axis=0)
                            for layer in range(len(per_seg[0]))
                        ]
            except preempt.PreemptedError as stop:
                if pf is not None:
                    pf.cancel()
                return self._preempted(items, stop)
        tm.mark("submit")
        # hidden is None on a generic-layer step where no prompt ends (_draw then picks nothing)
        draws = self._draw(items, hidden)
        tm.mark("draw")
        # Evaluate every cache the step advanced, not only what feeds a token:
        # a prompt chunk that emits nothing would otherwise stay a lazy graph
        # on top of the previous chunk's, and a long prompt becomes one graph
        # holding every chunk's KV buffer version until its last chunk.
        advanced = {id(it.row): it.row for it in items}
        t_eval = time.perf_counter()
        mx.eval(
            *[d[1] for d in draws],
            *[a for d in draws if d[2] for a in d[2]],
            *[a for r in advanced.values() for a in cache_buffers(cast(list, r.cache))],
        )
        tm.mark("wait")
        if self.layer.caps.fn_trace and len(span_t) == len(items) + 2:
            self._log_spans(items, span_t, time.perf_counter() - t_eval)
        if entry is None:
            self._record_tail(items, hidden, captures)
        tm.mark("ckpt")
        events: list[Event] = []
        for it, tok, extra in draws:
            self._emit(it.row, [int(tok.tolist()[0])], extra, events)
        tm.mark("emit")
        if self.dflash is not None and captures:
            self.dflash.absorb(items, captures)
        tm.mark("capture")
        # prompt chunks of draftable rows feed their MTP head, in order
        absorb: dict[int, tuple] = {}
        for it in items:
            row = it.row
            if row.mtp_cache is None or row.finished:
                continue
            ids = row.req.ids
            nxt = ids[it.start + 1 : it.end]
            tail_tok = ids[it.end] if it.end < len(ids) else row.pending
            assert tail_tok is not None
            nxt.append(tail_tok)
            absorb_entry = absorb.setdefault(id(row), (row, [], [], []))
            absorb_entry[1].extend(nxt)
            # rows with an MTP head run the packed / Flash-Next layer: hidden holds every span row
            absorb_entry[2].append(cast(mx.array, hidden)[it.at : it.at + it.count])
            if row.req.position_ids is not None:
                absorb_entry[3].append(row.req.position_ids[:, it.start : it.end])
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
        tm.mark("head")
        for it in items:
            if it.end in it.row.ckpts and not it.row.finished:
                self._store_checkpoint(it.row, it.end)
        tm.mark("ckpt")
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
        # The MTP head's prompt prefill (one layer over every prompt row, ~3% of a Flash-Next
        # prefill) stays queued while its HC states fit HEAD_LAZY_MAX: the first token does not
        # wait for it. A row that joins evaluates it in _finish_join, after the token is out.
        arrays += [
            a
            for r in self.rows
            if r.mtp_cache and not self._head_lazy(r, advanced, hidden, items)
            for a in cache_buffers(r.mtp_cache)
        ]
        arrays += [
            h for k, h in self._pending_heads.items() if not self._lazy_ids.get(k)
        ]
        mx.eval(*arrays)
        tm.mark("tail")
        for r in advanced.values():
            if r.tm is None:
                r.tm = fn_ttft.Timer()
                r.tm.t0 = tm.t0
            r.tm.merge(tm)
            if (
                r.done == len(r.req.ids)
                and r.hit < len(r.req.ids)
                and self.layer.caps.fn_trace
            ):
                r.tm.log("driver", len(r.req.ids), r.hit)
                r.tm = None
        self.steps += 1
        return events

    ahead_steps = 0  # decode steps taken from a queued next step (pipelined lone rows)

    def _lone_blockers(self) -> list[str]:
        """Why the lone row cannot decode pipelined (empty: it can)."""
        if len(self.rows) != 1:
            return ["rows"]
        r = self.rows[0]
        req = r.req
        checks = {
            "dflash": self.dflash is not None,
            "head": self.head is not None,
            "layer": not self.layer.caps.lone_pipelining,
            "batch": len(self.batch.rows) != 1,
            "ngram": r.ngram is not None,
            "force": bool(r.force),
            "guide": req.guide is not None,
            "processors": bool(req.processors),
            "budget": req.budget is not None,
            "sampling": req.sampling is not None,
            "logprobs": bool(req.logprobs),
            "penalty": r.penalty is not None,
        }
        return [k for k, v in checks.items() if v]

    def _lone_ahead_ok(self) -> bool:
        """A lone plain greedy generic row decodes pipelined (the stock loop's ``async_eval`` of the
        next step before reading this one): nothing host-side can change the next token."""
        blockers = self._lone_blockers()
        if blockers and len(self.rows) == 1 and not self.rows[0].ahead_logged:
            self.rows[0].ahead_logged = True
            logger.info("round driver: lone row not pipelined: %s", ",".join(blockers))
        return not blockers

    def _drop_ahead(self) -> None:
        """Forget a queued next step of a row that leaves (stop, cancel): only its output token is dropped.
        The step's input was the row's last emitted token, so the cache then holds valid K/V for the whole
        context including that token: one position more than a normal finish, never trimmed (a window
        cache cannot trim) and stored as it is (the store accepts a cache holding every emitted token)."""
        for r in self.batch.rows:
            r.ahead = None

    def _settle_ahead(self) -> list[Event]:
        """Finish the queued next step of every row that has one (its token is the row's correct next token:
        the same M=1 arithmetic), so the batch can change shape with no position to trim."""
        events: list[Event] = []
        rows = [r for r in self.rows if r.ahead is not None]
        for row in rows:
            tok, row.ahead = row.ahead, None
            self.ahead_steps += 1
            used = self._emit(row, tok.tolist(), None, events)
            self.batch.commit(
                [used] if len(self.batch.rows) == 1 else [1] * len(self.batch.rows)
            )
        gone = [r for r in rows if r.finished]
        self.rows = [r for r in self.rows if not r.finished]
        self._release(gone)
        self.steps += 1
        return events

    def _lone_draw(self, row: _Row, tokens: mx.array) -> mx.array:
        hidden = self.batch.forward([1], tokens)
        return cast(
            mx.array,
            self._draw([_Item("d", row, None, 0, 0, 0, 1)], hidden)[0][1],
        )

    def _decode_step_lone(self) -> list[Event]:
        from .. import gpu_window

        gpu_window.tick(capture_kernels=False)
        gpu_window.mark("plain.graph")
        row = self.rows[0]
        tok = row.ahead
        row.ahead = None
        if tok is not None:
            self.ahead_steps += 1
        if tok is None:
            tok = self._lone_draw(row, mx.array([row.pending], dtype=mx.int32))
            mx.async_eval(tok, *self.batch.arrays())
        if row.generated + 2 <= row.req.max_tokens:
            row.ahead = self._lone_draw(row, tok.astype(mx.int32))
            mx.async_eval(row.ahead, *self.batch.arrays())
        gpu_window.mark("plain.readback")
        toks = tok.tolist()  # waits for the step
        events: list[Event] = []
        used = self._emit(row, toks, None, events)
        gpu_window.mark("plain.host_emit")
        self.batch.commit([used])
        if row.finished:
            self._drop_ahead()
        gone = [r for r in self.rows if r.finished]
        self.rows = [r for r in self.rows if not r.finished]
        self._release(gone)
        self.steps += 1
        return events

    def _decode_step(self) -> list[Event]:
        if self.dflash is not None:
            return cast(list[Event], self.dflash.step(self))
        if self._lone_ahead_ok():
            return self._decode_step_lone()
        if any(r.ahead is not None for r in self.rows):
            return self._settle_ahead()
        from .. import gpu_window
        from .stage_profile import RoundProfile

        prof = getattr(self, "_mtp_profile", None)
        if prof is None:
            self._mtp_profile = prof = RoundProfile()
        if not prof.on:
            gpu_window.tick(capture_kernels=False)
        prof.begin()
        gpu_window.mark("mtp.graph")
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
        lens, parts, used_drafts, wins = [], [], [], []
        for r in rows:
            forced = self._forced_window(r, MAX_DECODE_TOKENS) or self._ngram_window(r)
            if forced:
                r.draft_dev = mx.array(forced, dtype=mx.int32)
                r.nd = len(forced)
            use = r.nd if r.draft_dev is not None and not r.force else 0
            parts.append(mx.array([r.pending], dtype=mx.int32))
            if use:
                parts.append(r.draft_dev)
            wins.append(
                mx.concatenate(parts[-2:])
                if use
                and (r.penalty is not None or r.generic or r.req.guide is not None)
                else None
            )
            lens.append(1 + use)
            used_drafts.append(use)
        if prof.on:
            prof.mark("plan", *parts)
        want_hidden = (
            self.generic
            and len(rows) == 1
            and rows[0].ngram is not None
            and rows[0].ngram.needs_hidden
        )
        if self.generic:
            self.batch.want_hidden = want_hidden
        hidden = self.batch.forward(lens, mx.concatenate(parts))
        items, at = [], 0
        for r, n, win in zip(rows, lens, wins, strict=True):
            items.append(_Item("d", r, None, 0, 0, at, n, tokens=win))
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
        if prof.on:
            prof.mark("verify", *[d[1] for d in draws], early_out, early_seeds)
        gpu_window.mark("mtp.readback")
        window_tokens, draft_tokens = round_readback(
            [d[1] for d in draws], rows, used_drafts
        )  # the round's one token-ID readback
        step_s = time.perf_counter() - started
        gpu_window.mark("mtp.host_accept_emit")
        # forward cost: from when the GPU could start it (the chain before it
        # is not part of the forward) to now
        gpu_free = self._chain_t + self.chain_ms * self._chain_deep / 1e3
        ms = (time.perf_counter() - max(started, gpu_free)) * 1e3
        # a row's first decode steps carry one-time work (the head's queued prompt absorb, first
        # kernel shapes): recorded, they priced a one-token window above a seven-token one and the
        # allocator then drafted deep for the whole reply (fndepth-b7 prose 8K -40%)
        if ms > 0 and all(r.generated >= COST_WARM_STEPS for r in rows):
            self.cost.observe(at, ms)
        self._chain_deep = 0

        events: list[Event] = []
        used_all: list[int] = []
        for (it, _tok, extra), toks, draft in zip(
            draws, window_tokens, draft_tokens, strict=True
        ):
            row = it.row
            window = [row.pending, *draft]
            keep = 1
            while keep < len(window) and window[keep] == toks[keep - 1]:
                keep += 1
            if len(window) > 1:
                self._observe(row, len(window) - 1, keep - 1)
                if row.ngram is not None:
                    row.ngram.observe(len(window) - 1, keep - 1)
            # a stop, the length limit or the thinking budget can cut the
            # window short: the cache keeps what was emitted
            used = self._emit(row, toks[:keep], extra, events)
            if row.ngram is not None and len(rows) == 1:
                row.ngram.observe_step(len(window) > 1, used, step_s)
            if row.ngram is not None and row.ngram.needs_hidden:
                # the target's hidden state at the last kept position: the assistant's next proposal reads it
                kept = self.batch.hidden
                row.ngram.carry(
                    kept[0, max(used, 1) - 1]
                    if want_hidden and kept is not None
                    else None
                )
            used_all.append(used)
        gpu_window.mark("mtp.commit")
        self.batch.commit(used_all)
        gpu_window.mark("mtp.post_commit")
        gone = [r for r in rows if r.finished]
        self.rows = [r for r in self.rows if not r.finished]
        for b, r in head_rows:
            if r.finished:
                # the early absorb wrote every kept position of the last window into the head's KV:
                # the head is level with the target, so the reply's head cache can be stored
                r.hn += used_all[b]
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
                assert early_seeds is not None
                first = early_seeds[
                    mx.array([i for i, _, _ in alive], dtype=mx.int32),
                    mx.array(at_last, dtype=mx.int32),
                ]
                gpu_window.mark("mtp.next_draft")
                self._draft([r for _, _, r in alive], heads, first)
        if prof.on:
            prof.mark("commit", *self.batch.arrays())
            prof.end([id(r) for r in rows], None)
            for r in gone:
                prof.finish(id(r))
        gpu_window.mark("mtp.return")
        self.steps += 1
        return events

    def _forced_window(self, row: _Row, width: int) -> list[int]:
        """Plan a forced decode window without advancing the live guide.

        Feed the pending token and the forced prefix, then verify all target draws,
        including the bonus after that prefix. Planning stops before stop/guide-entry
        tokens, so a boundary can only be the final draw. Thinking-budget rows stay
        serial. Generic window rounding remains opt-in with the existing False default.
        """
        from .. import settings
        from ..constrained_spec import forced_tokens

        req = row.req
        if (
            not settings.get("YUNSHU_JUMP_FORWARD")
            or req.guide is None
            or row.force
            or req.budget is not None
            or (req.processors and not windowable(req.processors))
            or (req.sampling is not None and row.keyed is None)
        ):
            return []
        limit = min(width - 1, req.max_tokens - row.generated - 1)
        try:
            forced = forced_tokens(
                req.guide, limit, self.stop_tokens, before_boundary=True
            )
        except GrammarDeadEndError:
            # Lookahead is speculative. Only _draw's committed-position mask
            # may fault the live request; a future dead end retains serial emission.
            return []
        if not forced:
            return []
        # A formerly plain row now applies processors to every window position. Its
        # penalty already holds the serial committed counts; _emit commits only fed tokens.
        if generic_processors(req.processors, True, bool(req.logprobs)):
            row.generic = True
            row.penalty = next(
                (p for p in req.processors if hasattr(p, "window_row")), None
            )
        row.forced_windows += 1
        if row.forced_windows == 1:
            logger.info(
                "Round driver forced window engaged: %d tokens", len(forced) + 1
            )
        return forced

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

        def report(token: int, one) -> dict | None:
            if one is None:
                return None
            lp = {
                "token_id": token,
                "logprob": float(one[0].item()),
                "top_logprobs": [],
            }
            if len(one) > 1:
                lp["top_logprobs"] = [
                    {"token_id": int(t), "logprob": float(v)}
                    for t, v in zip(one[1][0].tolist(), one[2][0].tolist(), strict=True)
                ]
            return lp

        per_row = isinstance(extra, _PerRow)
        lp = report(committed[0], extra[0] if per_row else extra)
        row.drafts, row.draft_dev, row.nd = [], None, 0
        before = row.pending
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
            out.append(
                Event(
                    row.req.handle,
                    t,
                    lp,
                    finish,
                    prompt_logprobs=row.prompt_scores
                    if row.generated == 1 and row.req.prompt_logprobs is not None
                    else None,
                )
            )
            lp = (
                report(committed[used], extra[used])
                if per_row and used < len(committed) and used < len(extra)
                else None
            )
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
        if row.penalty is not None:
            # count what a serial run would have fed: the old pending token and every kept token
            # but the new pending one (``commit_tokens``)
            fed = ([before] if before is not None else []) + committed[: used - 1]
            if fed:
                row.penalty.commit_tokens(fed)
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
            room = min(self.max_draft, r.req.max_tokens - r.generated - 1)
            probs.append(chain(r.rates, room) if room > 0 else [])
        return allocate(fixed, probs, self.cost, LANE_ROWS, self.chain_ms, padded=True)


__all__ = ["CHUNK", "Event", "Request", "RoundDriver", "cache_buffers"]
