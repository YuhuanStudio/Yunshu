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
from types import SimpleNamespace
from typing import TYPE_CHECKING

import mlx.core as mx

from .. import dflash_ring, dflash_tree, keyed_sampling
from ..constrained_spec import masks_every_row
from ..text_logprobs import top_logprobs_device
from . import flashnext
from .allocate import fair_nodes
from .dflash import DFlashBatch, _plain_guided

if TYPE_CHECKING:
    from .driver import Event

logger = logging.getLogger(__name__)

PROPOSALS = 15  # drafter positions per round (its trained block minus the root)


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

    def _step_rows(self, driver, rows) -> list[Event]:
        from ..tree_verify import TreeShape
        from .driver import _Item

        rooms = []
        for r in rows:
            room = max(0, r.req.max_tokens - r.generated - 1)
            rooms.append(
                0
                if not r.drafting or r.force or _plain_guided(r)
                else min(PROPOSALS, room)
            )
        nodes = fair_nodes(rooms, budget=flashnext.PACK_ROWS, turn=self.turn)
        self.turn += 1
        live = [b for b, n in enumerate(nodes) if n]
        chains: dict[int, mx.array] = {}
        if live:
            hs = dflash_ring.draft_hidden_batch(
                self.draft,
                [self.rings[id(rows[b])] for b in live],
                mx.array([rows[b].pending for b in live], dtype=mx.int32),
                PROPOSALS,
            )
            for j, b in enumerate(live):
                r = rows[b]
                lat = dflash_tree.lattice_from_hidden(
                    self.draft,
                    hs[j : j + 1],
                    mx.array([r.pending], dtype=mx.int32),
                    keyed=r.keyed,
                    committed=r.generated,
                    penalty=r.penalty,
                )
                chains[b] = lat.cands[: nodes[b], 0].astype(mx.int32)
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
            for r, w, res, shape in zip(rows, windows, results, shapes, strict=True):
                n = int(w.shape[1])
                row_logits = packed[at : at + n]
                at += n
                if not r.drafting or r.force or _plain_guided(r):
                    it = _Item("d", r, at=0, count=1)
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
            for r, w, res, (target, lp, extra) in zip(
                rows, windows, results, draws, strict=True
            ):
                toks = [int(t) for t in w.reshape(-1).tolist()]
                targets = [int(t) for t in target.reshape(-1).tolist()]
                path = [0]
                while (
                    path[-1] + 1 < len(toks) and toks[path[-1] + 1] == targets[path[-1]]
                ):
                    path.append(path[-1] + 1)
                committed = [toks[p] for p in path[1:]] + [targets[path[-1]]]
                used = 0
                for p, token in zip(path, committed, strict=True):
                    emit_extra = extra
                    if r.req.logprobs and lp is not None:
                        emit_extra = [lp[p, token]]
                        if r.req.top_logprobs:
                            emit_extra += list(
                                top_logprobs_device(lp[p : p + 1], r.req.top_logprobs)
                            )
                    used += driver._emit(r, [int(token)], emit_extra, events)
                    if r.finished or r.force or _plain_guided(r):
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
                if not r.finished:
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
        mx.async_eval(*self.arrays())
        driver.steps += 1
        return events
