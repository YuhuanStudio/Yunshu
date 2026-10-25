"""DFlash transactions for RoundDriver: per-row context, one packed verify.

The driver's prefill scheduler, request lifecycle and APC remain the owners.
This adapter retains each row's lane caches so tree_commit can replay exactly
its accepted path. It does not run a server or a separate generation loop.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import mlx.core as mx

from .. import dflash_ring, dflash_tree, keyed_sampling, tree_verify
from ..dflash_context import context_window
from ..dflash_fast import Verifier, prepare
from ..dflash_plan import FastShape, reorder, search_tree
from .allocate import fair_nodes
from .forward import logits
from .verify import Window, forward

if TYPE_CHECKING:
    from .driver import Event


logger = logging.getLogger(__name__)


class DFlashBatch:
    def __init__(self, model, draft):
        from mlx_vlm.models.qwen3_5 import language as q35

        self.model = model
        self.lm = getattr(model, "language_model", model)
        draft.reset(model)
        self.draft = prepare(draft).view_of(draft)
        keep = context_window(draft)
        if keep is None:
            raise ValueError("packed DFlash requires a bounded sliding context")
        self.keep = keep
        self.capture_ids = list(draft.config.target_layer_ids)
        self.verifier = Verifier(q35._EXACT_SPECULATIVE_VERIFIER)
        self.rows = []
        self.hidden = {}
        self.rings = {}
        self.turn = 0
        logger.info(
            "Round driver DFlash packed: shared 32-node budget, context=%d", self.keep
        )

    def absorb(self, items, captures):
        """Retain only the drafter's trailing context during target prefill."""
        packed = mx.concatenate(captures, axis=-1)
        for it in items:
            key = id(it.row)
            if it.row.finished:
                self.hidden.pop(key, None)
                continue
            h = packed[it.at : it.at + it.count][None]
            old = self.hidden.get(key)
            if old is not None:
                h = mx.concatenate([old, h], axis=1)
            self.hidden[key] = h[:, -self.keep :]

    def join(self, rows):
        for row in rows:
            hidden = self.hidden.pop(id(row), None)
            ring = dflash_ring.Ring(self.draft, self.keep + 64, self.keep)
            count = 0 if hidden is None else int(hidden.shape[1])
            ring.pos = mx.array([row.n - count], dtype=mx.int32)
            if count:
                ring.append(self.draft, hidden, mx.array([count], dtype=mx.int32))
            self.rings[id(row)] = ring
            row.slot = id(row)
            self.rows.append(row)
            tree_verify.reserve_decode(row.cache, row.req.max_tokens + 32)

    def leave(self, rows):
        gone = {id(r) for r in rows}
        self.rows = [r for r in self.rows if id(r) not in gone]
        for row in rows:
            row.slot = None
            self.rings.pop(id(row), None)
            self.hidden.pop(id(row), None)

    def arrays(self):
        from .driver import cache_buffers

        arrays = [a for r in self.rows for a in cache_buffers(r.cache)]
        arrays += list(self.hidden.values())
        for ring in self.rings.values():
            arrays += [ring.pos, ring.slot_pos, *ring.keys, *ring.values]
        return arrays

    def step(self, driver) -> list[Event]:
        from ..kernels import lane_linear

        previous = lane_linear.sum_reuse_enabled()
        lane_linear.set_sum_reuse(True)
        rows = list(self.rows)
        events: list[Event] = []
        try:
            # More than 32 live requests cannot fit even their roots in one
            # certified forward. Every row still advances in this scheduler
            # step, in bounded packs; no request loses its turn.
            for start in range(0, len(rows), tree_verify.FAST_MAX_ROWS):
                events.extend(
                    self._step_rows(
                        driver, rows[start : start + tree_verify.FAST_MAX_ROWS]
                    )
                )
            return events
        finally:
            lane_linear.set_sum_reuse(previous)
            tree_verify.vq.clear_group_sums()

    def _step_rows(self, driver, rows) -> list[Event]:
        from .driver import _Item

        rooms = [
            min(15, max(0, r.req.max_tokens - r.generated - 1))
            if r.drafting and not r.force
            else 0
            for r in rows
        ]
        nodes = fair_nodes(rooms, turn=self.turn)
        self.turn += 1
        live = [b for b, n in enumerate(nodes) if n]
        proposals = {}
        if live:
            hs = dflash_ring.draft_hidden_batch(
                self.draft,
                [self.rings[id(rows[b])] for b in live],
                mx.array([rows[b].pending for b in live], dtype=mx.int32),
                15,
            )
            for j, b in enumerate(live):
                r = rows[b]
                lat = dflash_tree.lattice_from_hidden(
                    self.draft,
                    hs[j : j + 1],
                    mx.array([r.pending], dtype=mx.int32),
                    keyed=r.keyed,
                    committed=r.generated,
                )
                toks, parents = search_tree(lat, nodes[b])
                toks, parents, ranks = reorder(toks, parents)
                proposals[b] = (
                    mx.concatenate([mx.array([r.pending], dtype=mx.int32), toks])[None],
                    FastShape(parents, 15, original_ranks=ranks),
                )
        windows = []
        for b, r in enumerate(rows):
            tokens, shape = proposals.get(
                b,
                (mx.array([[r.pending]], dtype=mx.int32), tree_verify.TreeShape([-1])),
            )
            windows.append(Window(tokens, shape, r.cache))
        results = forward(self.lm, windows, self.capture_ids, verifier=self.verifier)
        draws = []
        committed_rows = set()
        try:
            packed_logits = logits(
                self.lm, mx.concatenate([res.hidden[0] for res in results])
            )
            at = 0
            for r, w, res in zip(rows, windows, results, strict=True):
                row_logits = packed_logits[at : at + w.shape.width]
                at += w.shape.width
                if not r.drafting or r.force:
                    it = _Item("d", r, at=0, count=1)
                    _, targets, extra = driver._draw(
                        [it], res.hidden[0], target_logits=row_logits
                    )[0]
                    draws.append((targets, None, extra))
                else:
                    targets, lp = keyed_sampling.tree_target_rows(
                        row_logits[None],
                        w.shape,
                        r.generated,
                        r.keyed,
                        SimpleNamespace(logprobs=r.req.logprobs),
                    )
                    draws.append((targets, lp, None))
            mx.async_eval(*[d[0] for d in draws])
            events: list[Event] = []
            for r, w, res, (target, lp, extra), n in zip(
                rows, windows, results, draws, nodes, strict=True
            ):
                toks = w.tokens.reshape(-1).tolist()
                parents = w.shape.parents_array().tolist()
                targets = target.reshape(-1).tolist()
                path = dflash_tree.walk(toks, parents, targets)
                committed = [toks[p] for p in path[1:]] + [targets[path[-1]]]
                used = 0
                for p, token in zip(path, committed, strict=True):
                    emit_extra = extra
                    if r.req.logprobs and lp is not None:
                        emit_extra = [lp[p, token]]
                        if r.req.top_logprobs:
                            idx = mx.argsort(lp[p : p + 1], axis=-1)[
                                :, -r.req.top_logprobs :
                            ][:, ::-1]
                            emit_extra += [
                                idx,
                                mx.take_along_axis(lp[p : p + 1], idx, axis=-1),
                            ]
                    used += driver._emit(r, [int(token)], emit_extra, events)
                    if r.finished or r.force:
                        break
                kept = path[:used]
                tree_verify.tree_commit(self.lm, r.cache, res, kept)
                committed_rows.add(id(r))
                r.n += used
                if n:
                    # Tree ranks are not chain depths; only aggregate counters
                    # use these statistics (the MTP acceptance EMA stays separate).
                    driver.drafted += n
                    driver.accepted += max(0, used - 1)
                    st = getattr(r.req.handle, "stats", None)
                    if st is not None:
                        from ..spec_metrics import observe

                        st.spec_drafted += n
                        st.spec_accepted += max(0, used - 1)
                        st.spec_rounds += 1
                        observe(st, max(0, used - 1), n)
                if not r.finished:
                    hidden = mx.concatenate(res.captured, axis=-1)[
                        :, mx.array(kept, dtype=mx.int32)
                    ]
                    self.rings[id(r)].append(
                        self.draft, hidden, mx.array([used], dtype=mx.int32)
                    )
        except BaseException:
            for r, w, res in zip(rows, windows, results, strict=True):
                if id(r) not in committed_rows:
                    tree_verify.tree_abort(w.cache, res)
            raise
        gone = [r for r in rows if r.finished]
        driver.rows = [r for r in driver.rows if not r.finished]
        driver._release(gone)
        mx.eval(*self.arrays())
        driver.steps += 1
        return events
