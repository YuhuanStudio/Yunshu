# Upstream (inspired): Blaizzy/mlx-vlm (MIT) speculative/drafters/dflash2, models/qwen4_exp/language.py @ 92b31ad3
"""DFlash2 drafting for Flash-Next rows on the round driver.

The drafter side is the one the Qwen3.5 DFlash batch uses (``dflash.DFlashBatch``: per-row context rings
fed with the target's captured layer features, one batched drafter forward for all rows, the trained
block and sliding window), so a row's context, APC window and prompt capture are inherited. The target
side differs: Flash-Next's QSA verifier is an exact chain (a tree would need per-node sparse selection),
so a row's window is the pending token and a chain of drafts, verified through the packed Flash-Next
step (``flashnext.chain_forward_capture``: the fused HC / expert kernels once over the pack, QSA / GDN /
PLE per row on the row's own caches) and committed through upstream's speculative cache transactions.
A drafted chain takes the drafter's best candidate at each depth; the target accepts the longest prefix
equal to its own tokens, so drafts only decide how much a step commits.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import mlx.core as mx

from .. import dflash_ring, dflash_tree, keyed_sampling
from ..constrained_spec import masks_every_row
from ..spec_schedule import CHAIN_PRIOR, NodeBudget
from ..text_logprobs import top_logprobs_device
from . import flashnext
from .allocate import fair_nodes
from .dflash import DFlashBatch, _plain_guided

if TYPE_CHECKING:
    from .driver import Event

logger = logging.getLogger(__name__)

PROPOSALS = 15  # most drafter positions per round (the drafter's trained block minus one is used)
CHAIN_DEPTH = 7  # deepest chain a cost-aware budget may pick (the old lane's ``install_chain_budget``)


def walk_on_device(window: mx.array, target: mx.array):
    """The accepted chain prefix of a window on the device: ``(used, bonus)`` where ``used`` positions are
    kept (the pending token plus the drafts equal to the target's own tokens) and ``bonus`` is the
    target's token after the last kept one. ``window`` [W] and ``target`` [W] int32."""
    width = int(window.shape[0])
    if width == 1:
        return mx.array([1], dtype=mx.int32), target[:1]
    match = (window[1:] == target[:-1]).astype(mx.int32)
    accepted = mx.cumprod(match).sum()
    return (accepted + 1).astype(mx.int32).reshape(1), mx.take(
        target, accepted
    ).reshape(1)


class FlashNextDFlash(DFlashBatch):
    """The decoding rows of a Flash-Next target with a DFlash2 drafter."""

    def __init__(self, model, draft, kv_format: str = "bf16"):
        if draft is None:
            raise ValueError(
                "FlashNextDFlash needs a drafter (spec off is FlashNextBatch)"
            )
        if kv_format != "bf16":
            raise ValueError(
                f"YUNSHU_KV_PRECISION={kv_format} is not available for Flash-Next on the round driver"
            )
        super().__init__(model, draft, kv_format)
        self.lm = getattr(model, "language_model", model)

    @property
    def positions(self) -> int:
        """Masked drafter positions of a round: its trained block minus the root."""
        return max(1, min(PROPOSALS, int(self.draft.config.block_size) - 1))

    def _chain(self, dh, anchor, row) -> mx.array:
        """The row's draft chain ``[positions]`` from the drafter's hidden rows ``dh`` [1, P, H]: the best
        candidate at each depth. A greedy row reads the argmax of the (reduced-vocabulary) head directly,
        without the candidate selector a tree needs; a keyed row takes the coupled candidates' best."""
        if row.keyed is None:
            vocab = getattr(self.draft, "_draft_vocab", None)
            subset = (
                vocab.logits_subset(dh)
                if vocab is not None
                and self.draft.config.final_logit_softcapping is None
                else None
            )
            if subset is not None:
                logits, ids = subset
                return (
                    mx.take(ids, mx.argmax(logits, axis=-1))
                    .astype(mx.int32)
                    .reshape(-1)
                )
            return mx.argmax(self.draft._logits(dh)[0], axis=-1).astype(mx.int32)
        lat = dflash_tree.lattice_from_hidden(
            self.draft,
            dh,
            anchor,
            keyed=row.keyed,
            committed=row.generated,
            penalty=row.penalty,
        )
        return cast(mx.array, lat.cands[:, 0].astype(mx.int32))

    def _budget(self, row) -> NodeBudget:
        found = self.budgets.get(id(row))
        if found is None:
            found = self.budgets[id(row)] = NodeBudget(
                min(CHAIN_DEPTH, self.positions),
                prior=self.mode_priors.prior(
                    getattr(row, "keyed", None) is not None, CHAIN_PRIOR
                ),
            )
        return found

    # ── membership ───────────────────────────────────────────────────────
    def join(self, rows):
        for row in rows:
            hidden = self.hidden.pop(id(row), None)
            ring = dflash_ring.Ring(self.draft, self.keep + 64, self.keep)
            count = 0 if hidden is None else int(hidden.shape[1])
            # window-relative drafter positions, like the Qwen3.5 batch's ring
            ring.pos = mx.array([0], dtype=mx.int32)
            if count:
                ring.append(self.draft, hidden, mx.array([count], dtype=mx.int32))
            self.rings[id(row)] = ring
            row.slot = id(row)
            self.rows.append(row)

    # ── one step ─────────────────────────────────────────────────────────
    def step(self, driver) -> list[Event]:
        rows = list(self.rows)
        events: list[Event] = []
        for start in range(0, len(rows), flashnext.PACK_ROWS):
            events.extend(
                self._step_rows(driver, rows[start : start + flashnext.PACK_ROWS])
            )
        return events

    def _queue_next(self, rows, windows, results, draws, nodes) -> set:
        """A lone greedy drafting row (no guide / budget / force / report): take the accepted prefix on the
        device, append its features to the drafter ring and queue the next round's chain, all before the
        host reads the verify back (the Qwen3.5 batch's next-round pipelining). Returns the rows whose ring
        was appended here."""
        if len(rows) != 1:
            return set()
        r, w, res = rows[0], windows[0], results[0]
        target, _lp, extra = draws[0]
        if (
            not r.drafting
            or r.force
            or r.keyed is not None
            or r.req.guide is not None
            or r.req.budget is not None
            or not nodes[0]
            or extra is not None
            or r.req.logprobs
        ):
            return set()
        used, bonus = walk_on_device(w.reshape(-1), target.reshape(-1).astype(mx.int32))
        ring = self.rings[id(r)]
        ring.append(self.draft, mx.concatenate(res[1], axis=-1), used)
        hs = dflash_ring.draft_hidden(self.draft, ring, bonus, self.positions)
        chain = self._chain(hs, bonus, r)
        mx.async_eval(chain)
        self.prebuilt[id(r)] = chain
        return {id(r)}

    def _step_rows(self, driver, rows) -> list[Event]:
        from ..tree_verify import TreeShape
        from .driver import _Item, _PerRow

        t_start = time.perf_counter()
        rooms = []
        forced = {}
        for b, r in enumerate(rows):
            run = driver._forced_window(r, flashnext.PACK_ROWS)
            if run:
                forced[b] = run
                rooms.append(len(run))
                continue
            room = max(0, r.req.max_tokens - r.generated - 1)
            rooms.append(
                0
                if not r.drafting or r.force or _plain_guided(r) or not room
                else min(self._budget(r).choose(room), room, self.positions)
            )
        nodes = fair_nodes(rooms, budget=flashnext.PACK_ROWS, turn=self.turn)
        self.turn += 1
        live = [b for b, n in enumerate(nodes) if n and b not in forced]
        chains: dict[int, mx.array] = {}
        ready = {b: self.prebuilt.pop(id(rows[b]), None) for b in live}
        self.prebuilt.clear()  # a row without a drafted round drops its queued chain
        fresh = [b for b in live if ready[b] is None]
        if fresh:
            pend = mx.array([rows[b].pending for b in fresh], dtype=mx.int32)
            hs = dflash_ring.draft_hidden_batch(
                self.draft,
                [self.rings[id(rows[b])] for b in fresh],
                pend,
                self.positions,
            )
            for j, b in enumerate(fresh):
                ready[b] = self._chain(hs[j : j + 1], pend[j : j + 1], rows[b])
        for b in live:
            chains[b] = ready[b][: nodes[b]]
        for b, run in forced.items():
            chains[b] = mx.array(run[: nodes[b]], dtype=mx.int32)
        windows = []
        for b, r in enumerate(rows):
            pend = mx.array([r.pending], dtype=mx.int32)
            windows.append(
                mx.concatenate([pend, chains[b]])[None] if b in chains else pend[None]
            )
        results = flashnext.chain_forward_capture(
            self.lm,
            windows,
            [r.cache for r in rows],
            self.capture_ids,
            [flashnext.media_of(r) for r in rows],
        )
        committed_rows: set[int] = set()
        events: list[Event] = []
        try:
            packed = flashnext.logits(
                self.lm, mx.concatenate([res[0][0] for res in results], axis=0)
            )
            draws, at = [], 0
            shapes = [TreeShape(list(range(-1, int(w.shape[1]) - 1))) for w in windows]
            for b, (r, w, res, shape) in enumerate(
                zip(rows, windows, results, shapes, strict=True)
            ):
                n = int(w.shape[1])
                row_logits = packed[at : at + n]
                at += n
                if b in forced or not r.drafting or r.force or _plain_guided(r):
                    it = _Item("d", r, at=0, count=n, tokens=w)
                    _, targets, extra = driver._draw(
                        [it], res[0][0], target_logits=row_logits
                    )[0]
                    draws.append((targets, None, extra))
                    continue
                drawn = row_logits[None]
                if r.req.guide is not None and masks_every_row(r.req.guide):
                    from ..constrained_spec import plan_tree
                    from ..tool_call_grammar import apply_bitmask

                    masks = plan_tree(
                        r.req.guide,
                        w.reshape(-1).tolist(),
                        shape.parents_array().tolist(),
                    )
                    drawn = apply_bitmask(drawn[0], masks)[None]
                if r.generic:
                    from .driver import window_processed

                    drawn = window_processed(
                        r.req.processors,
                        drawn[0],
                        r.context,
                        r.generated,
                        w.reshape(-1).tolist(),
                        shape.parents_array().tolist(),
                    )[None]
                    penalty = None
                else:
                    for p in r.req.processors:
                        if p is not r.penalty:
                            drawn = p.pure_at(r.generated, drawn)
                    penalty = r.penalty
                targets, lp = keyed_sampling.tree_target_rows(
                    drawn,
                    shape,
                    r.generated,
                    r.keyed,
                    SimpleNamespace(logprobs=r.req.logprobs),
                    penalty,
                    w,
                )
                draws.append((targets, lp, None))
            mx.async_eval(*[d[0] for d in draws])
            walked = self._queue_next(rows, windows, results, draws, nodes)
            for b, (r, w, res, (target, lp, extra)) in enumerate(
                zip(rows, windows, results, draws, strict=True)
            ):
                toks = [int(t) for t in cast(list[int], w.reshape(-1).tolist())]
                targets = [int(t) for t in target.reshape(-1).tolist()]
                path = [0]
                while (
                    path[-1] + 1 < len(toks) and toks[path[-1] + 1] == targets[path[-1]]
                ):
                    path.append(path[-1] + 1)
                committed = [toks[p] for p in path[1:]] + [targets[path[-1]]]
                used = 0
                for p, token in zip(path, committed, strict=True):
                    emit_extra = extra[p] if isinstance(extra, _PerRow) else extra
                    if r.req.logprobs and lp is not None:
                        emit_extra = [lp[p, token]]
                        if r.req.top_logprobs:
                            emit_extra += list(
                                top_logprobs_device(lp[p : p + 1], r.req.top_logprobs)
                            )
                    used += driver._emit(r, [int(token)], emit_extra, events)
                    if r.finished or r.force or (b not in forced and _plain_guided(r)):
                        break
                res[2].commit(used)
                committed_rows.add(id(r))
                r.n += used
                drafted = len(toks) - 1
                st = self.stats.setdefault(
                    id(r),
                    dict(rounds=0, nodes=0, used=0, copy=0, copy_used=0, hist=[0] * 17),
                )
                st["rounds"] += 1
                st["hist"][min(16, max(0, used - 1))] += 1
                st["nodes"] += drafted
                st["used"] += used
                if drafted:
                    driver.drafted += drafted
                    driver.accepted += max(0, used - 1)
                    stats = getattr(r.req.handle, "stats", None)
                    if stats is not None:
                        from ..spec_metrics import observe

                        stats.spec_drafted += drafted
                        stats.spec_accepted += max(0, used - 1)
                        stats.spec_rounds += 1
                        observe(stats, max(0, used - 1), drafted)
                vocab = getattr(self.draft, "_draft_vocab", None)
                if vocab is not None:
                    vocab.learn(committed[:used])
                if drafted:
                    self._budget(r).observe(
                        drafted,
                        range(min(max(0, used - 1), drafted)),
                        (time.perf_counter() - t_start) * 1000,
                        first=r.generated - used == 1,
                    )
                if id(r) in walked:
                    if not r.finished and used != len(path):
                        raise RuntimeError(
                            "round driver: a pipelined Flash-Next DFlash row was cut short"
                        )
                    if r.finished:
                        self.prebuilt.pop(id(r), None)
                elif not r.finished:
                    feats = mx.concatenate(res[1], axis=-1)[:, :used]
                    self.rings[id(r)].append(
                        self.draft, feats, mx.array([used], dtype=mx.int32)
                    )
        except BaseException:
            for r, res in zip(rows, results, strict=True):
                if id(r) not in committed_rows:
                    res[2].abort()
            raise
        gone = [r for r in rows if r.finished]
        for r in gone:
            self.stats.pop(id(r), None)
        driver.rows = [r for r in driver.rows if not r.finished]
        driver._release(gone)
        mx.async_eval(*self.arrays(), *self.prebuilt.values())
        driver.steps += 1
        return events
