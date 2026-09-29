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
   and rows with logits processors run one token a step through their own
   sampler / processors (never drafting);
3. commits each window (KV lengths advance, rows that kept part of a window
   continue their GDN state from that position), runs stop / length checks;
4. keeps every draftable row's MTP head current with the kept positions, then
   drafts for the next step with the cost-aware allocation (``allocate``) over
   the measured step cost curve.

The driver never looks at other rows to decide a row's tokens: greedy output is
the same alone, in any batch, with any draft depth (row-invariant forward).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .allocate import CostCurve, allocate, chain
from .batch import MAX_WINDOW, DecodeBatch
from .forward import Segment, forward, logits

logger = logging.getLogger(__name__)

MAX_DECODE_TOKENS = MAX_WINDOW  # pending token + up to 7 drafts
CHUNK = 512  # default prefill chunk: fixed spans from the prompt start
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
    budget: Any = None  # mlx_vlm ThinkingBudgetCriteria (forces "\n</think>")
    handle: Any = None
    extra_hash: int = 0  # APC salt (model / processor identity)
    use_apc: bool = True  # the runner's capacity check for this prompt


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
    drafts: list = field(default_factory=list)
    rates: list = field(default_factory=list)
    key: Any = None
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
        stop_tokens=None,
        chunk: int | None = None,
        apc: Any = None,
    ):
        from .mtp import MTPHead

        # Fixed prefill span per prompt (YUNSHU_ROUND_PREFILL_CHUNK). While rows
        # decode a prefill step is one span, so this is also the longest a
        # decoding row waits behind a prompt.
        self.chunk = int(chunk or CHUNK)
        self.idle_budget = max(IDLE_BUDGET, self.chunk)

        self.model = model
        self.lm = model.language_model if hasattr(model, "language_model") else model
        self.stop_tokens = set(stop_tokens or ())
        self.head = (
            MTPHead(drafter, self.lm, lambda h: logits(self.lm, h))
            if drafter is not None
            else None
        )
        prev = mx.set_cache_limit(CACHE_LIMIT)
        if prev < CACHE_LIMIT:
            mx.set_cache_limit(prev)  # a tighter limit set by the host stays
        self.batch = DecodeBatch(self.lm)
        # APC prefix cache (mlx_vlm ``APCManager``): a prompt starts from the
        # longest stored checkpoint (target KV + GDN state, plus the MTP
        # head's KV for rows that draft) and stores checkpoints at chunk
        # boundaries while it prefills.
        self.apc = None
        if apc is not None:
            from mlx_vlm.apc_coordinator import APCCoordinator

            coordinator = APCCoordinator(apc, self.lm)
            if coordinator.enabled and coordinator.is_checkpoint:
                self.apc = coordinator
        self.rows: list[_Row] = []
        self.cost = CostCurve()
        self.steps = 0
        self.drafted = 0
        self.accepted = 0
        self._prefilled_last = False
        self.chain_ms = 0.0  # measured drafting cost per chain depth
        # per draft depth: drafted / landed (telemetry)
        self.depth_drafted = [0] * (MAX_DECODE_TOKENS - 1)
        self.depth_landed = [0] * (MAX_DECODE_TOKENS - 1)

    # ── rows ─────────────────────────────────────────────────────────────
    def add(self, req: Request) -> int:
        """Queue a row; returns the prompt tokens restored from the APC."""
        if len(req.ids) < 1:
            raise ValueError("round driver: empty prompt")
        draft = bool(req.draft and self.head is not None and req.sampling is None)
        row = _Row(
            req=req,
            cache=self.lm.make_cache(),
            mtp_cache=self.head.make_cache() if draft else None,
            rates=[ACCEPT_PRIOR] * (MAX_DECODE_TOKENS - 1),
            context=list(req.ids),
            drafting=draft,
        )
        seed = getattr(req.sampling, "seed", None)
        if seed is not None:
            row.key = mx.random.key(int(seed))
        self._restore(row)
        self.rows.append(row)
        return row.hit

    # ── APC prefix cache ─────────────────────────────────────────────────
    def _restore(self, row: _Row) -> None:
        """Start ``row`` from the longest stored checkpoint of its prompt and
        plan the checkpoints it stores on the way."""
        if self.apc is None or not row.req.use_apc:
            return
        ids = row.req.ids
        # Entries carry the head's KV only for rows that draft, and come from
        # this driver's numerics: keep them apart from other users of the
        # manager (the upstream runner's entries) and between layouts.
        mix = f"{int(row.req.extra_hash)}:round-driver:{int(row.mtp_cache is not None)}"
        row.salt = (
            int.from_bytes(hashlib.blake2b(mix.encode(), digest_size=8).digest(), "big")
            >> 1
        )
        try:
            hit = self.apc.lookup(
                ids,
                extra_hash=row.salt,
                safe_lookup_min=0,
                suffix_is_text_only=lambda _n: True,
                prefix_has_media=lambda _n: False,
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
                with self.apc.manager.lock:
                    self.apc.manager.stats.restored_tokens += prefix
            self.apc.prepare_prefill(
                [len(ids)], prefill_step_size=self.chunk, prefix_lengths=[row.hit]
            )
            floor = self.apc.manager.exact_cache_min_tokens
            row.ckpts = [
                c
                for c in self.apc.checkpoint_lengths(ids, set())
                if row.hit < c < len(ids) and c >= floor
            ]
        except Exception:
            logger.warning("APC lookup failed; prefilling from scratch", exc_info=True)
            if row.hit == 0:
                row.cache = self.lm.make_cache()
                if row.mtp_cache is not None:
                    row.mtp_cache = self.head.make_cache()
            row.ckpts = []

    def _span_end(self, row: _Row, start: int) -> int:
        """End of the prompt span that starts at ``start``: the next multiple
        of ``chunk``, cut at the row's next APC checkpoint and at the prompt's
        end. The plan depends on the prompt alone (grid plus its
        checkpoints), so a row restored at a checkpoint on the grid continues
        with the spans a cold prefill would have run."""
        end = min((start // self.chunk + 1) * self.chunk, len(row.req.ids))
        for c in row.ckpts:
            if start < c < end:
                end = c
                break
        return end

    def _store_checkpoint(self, row: _Row, end: int) -> None:
        caches = list(row.cache)
        if row.mtp_cache is not None:
            caches += row.mtp_cache
        try:
            self.apc.store_checkpoint(row.req.ids[:end], caches, extra_hash=row.salt)
        except Exception:
            logger.warning("APC checkpoint store failed", exc_info=True)

    def remove(self, handle: Any) -> None:
        gone = [r for r in self.rows if r.req.handle is handle]
        self.rows = [r for r in self.rows if r.req.handle is not handle]
        self._release(gone)

    def _release(self, gone: list[_Row]) -> None:
        """Drop rows' slots in the decode batch and the head's."""
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
        from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, apply_xtc

        p = row.req.sampling
        if p is None:
            return mx.argmax(logprobs, axis=-1)
        x = logprobs
        if 0 < p.top_p < 1.0:
            x = apply_top_p(x, p.top_p)
        if p.min_p:
            x = apply_min_p(x, p.min_p)
        if p.xtc_probability > 0.0:
            x = apply_xtc(
                x, p.xtc_probability, p.xtc_threshold, list(p.xtc_special_tokens or [])
            )
        if p.top_k > 0:
            x = apply_top_k(x, p.top_k)
        x = x * (1 / p.temperature)
        if row.key is not None:
            row.key, sub = mx.random.split(row.key)
            return mx.random.categorical(x, key=sub)
        return mx.random.categorical(x)

    def _draw(self, items: list[_Item], hidden: mx.array):
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
        lg = logits(self.lm, hidden[mx.array(pick, dtype=mx.int32)])
        greedy = mx.argmax(lg, axis=-1)
        draws = []
        for it, off, n in outs:
            row = it.row
            req = row.req
            if row.force:
                # the thinking budget's token at this position, not a sample
                draws.append((it, mx.array(row.force[:1], dtype=mx.int32), None))
                row.force = row.force[1:]
                continue
            if req.sampling is None and not req.processors and not req.logprobs:
                draws.append((it, greedy[off : off + n], None))
                continue
            x = self._processed(row, lg[off : off + 1], first=it.kind == "p")
            lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
            tok = self._sample(row, lp)
            extra = None
            if req.logprobs:
                extra = [lp[0, tok[0]]]
                if req.top_logprobs > 0:
                    idx = mx.argsort(lp, axis=-1)[..., -req.top_logprobs :][..., ::-1]
                    extra += [idx, mx.take_along_axis(lp, idx, axis=-1)]
            draws.append((it, tok, extra))
        return draws

    # ── one step ─────────────────────────────────────────────────────────
    def step(self) -> list[Event]:
        if not self.rows:
            return []
        if self._will_decode():
            self._prefilled_last = False
            return self._decode_step()
        self._prefilled_last = True
        return self._prefill_step(
            [r for r in self.rows if r.pending is None], bool(self.batch.rows)
        )

    def _will_decode(self) -> bool:
        """Whether the next step is a decode step: rows decode and either no
        prompt waits or the last step was a prefill."""
        if not self.batch.rows:
            return False
        return self._prefilled_last or all(r.pending is not None for r in self.rows)

    def _prefill_step(self, waiting: list[_Row], decoding: bool) -> list[Event]:
        items: list[_Item] = []
        budget = self.chunk if decoding else self.idle_budget
        at = 0
        for r in waiting:
            ids = r.req.ids
            while budget > 0 and r.done < len(ids):
                start = r.done
                end = self._span_end(r, start)
                chunk = mx.array(ids[start:end], dtype=mx.int32)
                items.append(
                    _Item("p", r, Segment(r.cache, chunk), start, end, at, end - start)
                )
                r.done = end
                at += end - start
                budget -= end - start
                if end in r.ckpts:
                    break  # the checkpoint is stored after this span
        if not items:
            return []
        hidden = forward(self.lm, [it.seg for it in items])
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
        events: list[Event] = []
        for it, tok, extra in draws:
            self._emit(it.row, [int(tok.tolist()[0])], extra, events)
        # prompt chunks of draftable rows feed their MTP head, in order
        absorb: dict[int, tuple] = {}
        for it in items:
            row = it.row
            if row.mtp_cache is None or row.finished:
                continue
            ids = row.req.ids
            nxt = ids[it.start + 1 : it.end]
            nxt.append(ids[it.end] if it.end < len(ids) else row.pending)
            entry = absorb.setdefault(id(row), (row, [], []))
            entry[1].extend(nxt)
            entry[2].append(hidden[it.at : it.at + it.count])
        live = list(absorb.values())
        heads = {}
        if live:
            outs = self.head.absorb_prompt(
                [(r, t, mx.concatenate(h, axis=0)) for r, t, h in live]
            )
            heads = {id(r): o for (r, _, _), o in zip(live, outs, strict=True)}
        for it in items:
            if it.end in it.row.ckpts and not it.row.finished:
                self._store_checkpoint(it.row, it.end)
        self.rows = [r for r in self.rows if not r.finished]
        joined = [
            r
            for r in advanced.values()
            if r.pending is not None and not r.finished and r.done == len(r.req.ids)
        ]
        for r in joined:
            r.n = len(r.req.ids)
        self.batch.join(joined)
        arrays = [a for r in self.rows if r.cache for a in cache_buffers(r.cache)]
        arrays += [
            a for r in self.rows if r.mtp_cache for a in cache_buffers(r.mtp_cache)
        ]
        ready = [r for r in joined if r.drafting]
        if ready:
            self.head.join(ready)
            arrays += self.head.slots.arrays()
            self._draft(ready, mx.stack([heads[id(r)] for r in ready]))
        mx.eval(*arrays, *self.batch.arrays())
        self.steps += 1
        return events

    def _decode_step(self) -> list[Event]:
        started = time.perf_counter()
        rows = list(self.batch.rows)
        windows = [[r.pending, *([] if r.force else r.drafts)] for r in rows]
        hidden = self.batch.forward(windows)
        items, at = [], 0
        for r, w in zip(rows, windows, strict=True):
            items.append(_Item("d", r, None, 0, 0, at, len(w)))
            at += len(w)
        draws = self._draw(items, hidden)
        mx.eval(
            *[d[1] for d in draws],
            *[a for d in draws if d[2] for a in d[2]],
            *self.batch.arrays(),
        )
        self.cost.observe(at, (time.perf_counter() - started) * 1e3)

        events: list[Event] = []
        used_all: list[int] = []
        toks_all: list[list[int]] = []
        for (it, tok, extra), window in zip(draws, windows, strict=True):
            row = it.row
            toks = [int(t) for t in tok.tolist()]
            keep = 1
            while keep < len(window) and window[keep] == toks[keep - 1]:
                keep += 1
            if len(window) > 1:
                self._observe(row, len(window) - 1, keep - 1)
            # a stop, the length limit or the thinking budget can cut the
            # window short: the cache keeps what was emitted
            used = self._emit(row, toks[:keep], extra, events)
            used_all.append(used)
            toks_all.append(toks[:used])
        self.batch.commit(used_all)
        gone = [r for r in rows if r.finished]
        self.rows = [r for r in self.rows if not r.finished]
        self._release(gone)
        if self.head is not None:
            alive = [
                (b, r)
                for b, r in enumerate(rows)
                if r.hslot is not None and not r.finished
            ]
            if alive:
                heads = self.head.absorb(
                    [r for _, r in alive],
                    [toks_all[b] for b, _ in alive],
                    hidden,
                    [[items[b].at + j for j in range(used_all[b])] for b, _ in alive],
                )
                self._draft([r for _, r in alive], heads)
                mx.eval(*self.head.slots.arrays())
        self.steps += 1
        return events

    def _draft(self, rows: list[_Row], heads: mx.array) -> None:
        depths = self._depths(rows)
        drafted_at = time.perf_counter()
        drafts = self.head.draft(rows, heads, depths)
        deepest = max(depths, default=0)
        if deepest:
            ms = (time.perf_counter() - drafted_at) * 1e3 / deepest
            self.chain_ms += ACCEPT_EMA * (ms - self.chain_ms)
        for r, d in zip(rows, drafts, strict=True):
            r.drafts = d
            self.drafted += len(d)

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
        row.drafts = []
        used = 0
        for t in committed:
            used += 1
            row.generated += 1
            row.context.append(t)
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
        row.pending = committed[used - 1]
        return used

    # ── drafting policy ──────────────────────────────────────────────────
    def _observe(self, row: _Row, drafted: int, landed: int) -> None:
        self.accepted += landed
        st = getattr(row.req.handle, "stats", None)
        if st is not None:
            st.spec_drafted += drafted
            st.spec_accepted += landed
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
