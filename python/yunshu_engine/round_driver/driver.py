"""Round driver: rows advance in packed forwards, prefill and decode interleaved.

See docs/guides/ROUND_DRIVER.md. Each step is one packed forward of one kind:

- a **decode step**: every decoding row's window (pending token + drafts);
- a **prefill step**: fixed ``chunk``-token spans of waiting prompts —
  several prompts in one forward — up to ``idle_budget`` tokens when no row
  decodes, one span while rows decode.

While rows decode and prompts wait, prefill and decode steps alternate
one-to-one, so the chunk size sets the trade between the waiting prompt's
TTFT and the decoding rows' rate (27B, 4 rows + a 16K prompt: decode rows and
a prefill chunk packed into one forward measured no better than alternating
separate forwards at the same chunk size, docs/research/runs/
2026-09-29-fused-prefill/, so steps stay single-kind). A step:

1. plans its segments;
2. runs one packed forward (``forward.forward``) and one LM-head call over the
   rows that need logits;
3. samples: greedy rows take the argmax of every window position and keep the
   drafts up to the first mismatch plus the target's token there; sampled rows
   and rows with logits processors run one token a step through their own
   sampler / processors (never drafting);
4. commits each window (``Segment.commit``: KV trimmed and GDN state rolled
   back to the kept length), runs stop / length checks;
5. keeps every draftable row's MTP head current with the kept positions (and
   the prompt chunks just prefilled), then drafts for the next step with the
   cost-aware allocation (``allocate``) over the measured step cost curve.

The driver never looks at other rows to decide a row's tokens: greedy output is
the same alone, in any batch, with any draft depth (row-invariant forward).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .allocate import CostCurve, allocate, chain
from .forward import MAX_DECODE_TOKENS, Segment, forward, logits

logger = logging.getLogger(__name__)

CHUNK = 512  # default prefill chunk: fixed spans from the prompt start
IDLE_BUDGET = (
    2048  # prefill tokens per prefill step when no row decodes (at least one chunk)
)
ACCEPT_PRIOR = 0.7  # per-depth draft acceptance before a row has history
ACCEPT_EMA = 0.15


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
    cache: list
    mtp_cache: list | None
    done: int = 0  # prompt tokens prefilled
    pending: int | None = None  # committed token not yet in the cache
    generated: int = 0
    drafts: list = field(default_factory=list)
    rates: list = field(default_factory=list)
    key: Any = None
    context: list = field(default_factory=list)
    mtp_temp: int = 0
    force: list = field(default_factory=list)  # the budget's next token
    finished: bool = False

    @property
    def draftable(self) -> bool:
        return self.mtp_cache is not None


@dataclass
class _Item:
    kind: str  # "d": decode window, "p": prefill chunk
    row: _Row
    seg: Segment
    start: int = 0
    end: int = 0
    at: int = 0  # first packed row


class RoundDriver:
    """Rows of one Qwen3.5-family target (projections converted to lane)."""

    def __init__(
        self,
        model: Any,
        *,
        drafter: Any = None,
        stop_tokens=None,
        chunk: int | None = None,
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
    def add(self, req: Request) -> None:
        if len(req.ids) < 1:
            raise ValueError("round driver: empty prompt")
        draft = bool(req.draft and self.head is not None and req.sampling is None)
        row = _Row(
            req=req,
            cache=self.lm.make_cache(),
            mtp_cache=self.head.make_cache() if draft else None,
            rates=[ACCEPT_PRIOR] * (MAX_DECODE_TOKENS - 1),
            context=list(req.ids),
        )
        seed = getattr(req.sampling, "seed", None)
        if seed is not None:
            row.key = mx.random.key(int(seed))
        self.rows.append(row)

    def remove(self, handle: Any) -> None:
        self.rows = [r for r in self.rows if r.req.handle is not handle]

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

    # ── one step ─────────────────────────────────────────────────────────
    def _plan(self) -> list[_Item]:
        items: list[_Item] = []
        decoding = [r for r in self.rows if r.pending is not None]
        waiting = [r for r in self.rows if r.pending is None]
        if decoding and (not waiting or self._prefilled_last):
            self._prefilled_last = False
            for r in decoding:
                drafts = [] if r.force else r.drafts
                window = mx.array([r.pending, *drafts], dtype=mx.int32)
                items.append(_Item("d", r, Segment(r.cache, window, decode=True)))
            return self._place(items)
        self._prefilled_last = True
        budget = self.chunk if decoding else self.idle_budget
        for r in waiting:
            ids = r.req.ids
            while budget > 0 and r.done < len(ids):
                start, end = r.done, min(r.done + self.chunk, len(ids))
                chunk = mx.array(ids[start:end], dtype=mx.int32)
                items.append(_Item("p", r, Segment(r.cache, chunk, False), start, end))
                r.done = end
                budget -= end - start
        return self._place(items)

    @staticmethod
    def _place(items: list[_Item]) -> list[_Item]:
        at = 0
        for it in items:
            it.at = at
            at += it.seg.length
        return items

    def step(self) -> list[Event]:
        if not self.rows:
            return []
        started = time.perf_counter()
        items = self._plan()
        if not items:
            return []
        hidden = forward(self.lm, [it.seg for it in items])
        # logits: every decode window position; the last position of a prompt
        # finished this step
        pick, outs = [], []
        for it in items:
            if it.kind == "d":
                outs.append((it, len(pick), it.seg.length))
                pick.extend(range(it.at, it.at + it.seg.length))
            elif it.end == len(it.row.req.ids):
                outs.append((it, len(pick), 1))
                pick.append(it.at + it.seg.length - 1)
        lg = logits(self.lm, hidden[mx.array(pick, dtype=mx.int32)]) if pick else None
        greedy = mx.argmax(lg, axis=-1) if lg is not None else None
        draws = []  # (item, token ids array, logprob arrays or None)
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
        # Evaluate every cache the step advanced, not only what feeds a token:
        # a prompt chunk that emits nothing would otherwise stay a lazy graph
        # on top of the previous chunk's, and a long prompt becomes one graph
        # holding every chunk's KV buffer version until its last chunk.
        advanced = {id(it.row): it.row for it in items if it.kind == "p"}
        mx.eval(
            *[d[1] for d in draws],
            *[a for d in draws if d[2] for a in d[2]],
            *[a for r in advanced.values() for a in cache_buffers(r.cache)],
        )
        self.cost.observe(
            sum(it.seg.length for it in items), (time.perf_counter() - started) * 1e3
        )

        events: list[Event] = []
        absorb: dict[int, tuple] = {}  # row id -> (row, next tokens, [hidden])
        for it, tok, extra in draws:
            row = it.row
            toks = [int(t) for t in tok.tolist()]
            if it.kind == "d":
                window = [int(t) for t in it.seg.tokens.tolist()]
                keep = 1
                while keep < len(window) and window[keep] == toks[keep - 1]:
                    keep += 1
                if len(window) > 1:
                    self._observe(row, len(window) - 1, keep - 1)
                # a stop, the length limit or the thinking budget can cut the
                # window short: the cache keeps what was emitted
                used = self._emit(row, toks[:keep], extra, events)
                it.seg.commit(used)
                if row.draftable:
                    absorb[id(row)] = (row, toks[:used], [hidden[it.at : it.at + used]])
            else:
                self._emit(row, toks[:1], extra, events)
        # prompt chunks of draftable rows feed their MTP head, in order
        for it in items:
            row = it.row
            if it.kind != "p" or not row.draftable or row.finished:
                continue
            ids = row.req.ids
            nxt = ids[it.start + 1 : it.end]
            nxt.append(ids[it.end] if it.end < len(ids) else row.pending)
            h = hidden[it.at : it.at + it.seg.length]
            entry = absorb.setdefault(id(row), (row, [], []))
            entry[1].extend(nxt)
            entry[2].append(h)
        self.rows = [r for r in self.rows if not r.finished]
        live = [e for e in absorb.values() if not e[0].finished]
        if self.head is not None and live:
            heads = self.head.absorb(
                [(r, t, mx.concatenate(h, axis=0)) for r, t, h in live]
            )
            ready = [
                (r, o)
                for (r, _, _), o in zip(live, heads, strict=True)
                if r.pending is not None
            ]
            depths = self._depths([r for r, _ in ready])
            drafted_at = time.perf_counter()
            drafts = self.head.draft(
                [(r, o, d) for (r, o), d in zip(ready, depths, strict=True)]
            )
            deepest = max(depths, default=0)
            if deepest:
                ms = (time.perf_counter() - drafted_at) * 1e3 / deepest
                self.chain_ms += ACCEPT_EMA * (ms - self.chain_ms)
            for (r, _), d in zip(ready, drafts, strict=True):
                r.drafts = d
                self.drafted += len(d)
            # the head caches of rows still prefilling (no drafts yet)
            mx.eval(*[a for r, _, _ in live for a in cache_buffers(r.mtp_cache)])
        self.steps += 1
        return events

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
        for j in range(drafted):
            self.depth_drafted[j] += 1
            self.depth_landed[j] += int(j < landed)
        for j in range(min(drafted, landed + 1)):
            hit = 1.0 if j < landed else 0.0
            row.rates[j] += ACCEPT_EMA * (hit - row.rates[j])

    def _depths(self, ready: list[_Row]) -> list[int]:
        """Next step's drafts per ready row (cost-aware allocation)."""
        if not ready:
            return []
        fixed = sum(1 for r in self.rows if r.pending is not None)
        probs = []
        for r in ready:
            room = min(MAX_DECODE_TOKENS - 1, r.req.max_tokens - r.generated - 1)
            probs.append(chain(r.rates, room) if room > 0 else [])
        return allocate(
            fixed, probs, self.cost, fixed + sum(len(p) for p in probs), self.chain_ms
        )


__all__ = ["CHUNK", "Event", "Request", "RoundDriver", "cache_buffers"]
