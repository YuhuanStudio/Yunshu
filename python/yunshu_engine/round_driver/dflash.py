"""DFlash transactions for RoundDriver: per-row context, one packed verify.

The driver's prefill scheduler, request lifecycle and APC remain the owners.
This adapter retains each row's lane caches so tree_commit can replay exactly
its accepted path. It does not run a server or a separate generation loop.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from .. import (
    dflash_apc_window,
    dflash_ring,
    dflash_tree,
    keyed_sampling,
    settings,
    tree_verify,
)
from ..constrained_spec import masks_every_row
from ..copy_drafter import CopyDrafter
from ..dflash_context import context_window
from ..dflash_fast import Verifier, pack_readback, prepare, unpack_readback
from ..dflash_plan import FastShape, remap_landed, reorder, search_tree
from ..text_logprobs import top_logprobs_device
from .allocate import fair_nodes
from .forward import logits
from .verify import DRIVER_MAX_ROWS, Window, forward

if TYPE_CHECKING:
    from .driver import Event


logger = logging.getLogger(__name__)


def _plain_guided(r) -> bool:
    """A guided row that decodes one masked token a round: a tool-call guide inside a call.
    A response-format guide masks every row, so its row drafts a masked tree instead."""
    guide = r.req.guide
    return guide is not None and guide.constrained and not masks_every_row(guide)


class DFlashBatch:
    def __init__(self, model, draft, kv_format: str = "bf16"):
        self.kv_format = kv_format
        self._diag = dict(rounds=0, rows=0, s=0.0, nodes=0, used=0)
        from mlx_vlm.models.qwen3_5 import language as q35

        self.model = model
        self.lm = getattr(model, "language_model", model)
        # draft None: speculation off. Rows decode through the very same packed verify
        # forward (a one-token window) but never draft, so spec on == spec off.
        self.can_draft = draft is not None
        self.draft: Any = None
        if draft is None:
            self.draft, self.keep, self.capture_ids = None, 0, []
        else:
            draft.reset(model)
            self.draft = prepare(draft).view_of(draft)
            keep = context_window(draft)
            if keep is None:
                raise ValueError("packed DFlash requires a bounded sliding context")
            self.keep = keep
            self.capture_ids = list(draft.config.target_layer_ids)
        self.verifier = Verifier(q35._EXACT_SPECULATIVE_VERIFIER)
        self.rows: list[Any] = []
        self.stats: dict[int, Any] = {}
        self.hidden: dict[int, Any] = {}
        # YUNSHU_NEXT_TURN_PREFILL=decode: id(row) -> DecodeTail (the drafter window ending at the
        # row's cache), and the GPU-walked rounds' padded captured rows
        self.tails: dict[int, Any] = {}
        self.walk_rows: dict[int, Any] = {}
        self.rings: dict[int, Any] = {}
        self.copiers: dict[int, Any] = {}
        # A lone drafting row's next lattice, queued behind its verify before the host readback
        # (``_prebuild``): id(row) -> lattice
        self.prebuilt: dict[int, Any] = {}
        # A lone row's cost-aware node budget (the lane's NodeBudget: 0 / 7 / 15 nodes by measured
        # landing and round cost): id(row) -> NodeBudget
        self.budgets: dict[int, Any] = {}
        self.turn = 0
        logger.info(
            "Round driver DFlash packed: shared 32-node budget, context=%d", self.keep
        )

    def absorb(self, items, captures):
        """Retain only the drafter's trailing context during target prefill."""
        if not self.can_draft:
            return
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
        # Evaluate the windows now: the prefill step evaluates only the caches, and a window left
        # lazy chained every chunk's captures (5 layers x chunk x hidden, 100 MiB per 2048 rows on
        # the 27B) until the first token: +5 GiB at 128K over the lane, which evaluates them.
        mx.async_eval(
            *[self.hidden[id(it.row)] for it in items if id(it.row) in self.hidden]
        )

    def window(self, row):
        """The drafter's trailing prompt context as one capture list per target layer, or
        None. Stored beside an APC checkpoint so a hit restores the drafter's context
        (without it a follow-up turn drafts with only the uncached suffix's positions)."""
        hidden = self.hidden.get(id(row))
        if hidden is None or not self.capture_ids:
            return None
        return list(mx.split(hidden, len(self.capture_ids), axis=-1))

    def seed(self, row, layers):
        """Restore the window a checkpoint stored (see ``window``)."""
        self.hidden[id(row)] = mx.concatenate(list(layers), axis=-1)[:, -self.keep :]

    @staticmethod
    def _quantize_cache(cache: list) -> None:
        """int8 KV: a joining row's bf16 prefill attention caches become int8 row caches (codes +
        scales), quantized once from the same bf16 values whether the prefix was restored from the
        APC or computed (hit == miss); GDN state is untouched."""
        from mlx_vlm.models.cache import KVCache

        from ..kernels.int8_kv import Int8KVCache

        for i, c in enumerate(cache):
            if type(c) is KVCache:
                cache[i] = Int8KVCache.from_kv(c)

    def join(self, rows):
        for row in rows:
            if self.kv_format == "int8":
                self._quantize_cache(row.cache)
            if not self.can_draft:
                row.slot = id(row)
                self.rows.append(row)
                tree_verify.reserve_decode(row.cache, row.req.max_tokens + 32)
                continue
            hidden = self.hidden.pop(id(row), None)
            if (
                hidden is not None
                and settings.get("YUNSHU_NEXT_TURN_PREFILL") == "decode"
            ):
                self.tails[id(row)] = dflash_apc_window.DecodeTail(
                    hidden, row.n, self.keep, len(self.capture_ids)
                )
            ring = dflash_ring.Ring(self.draft, self.keep + 64, self.keep)
            count = 0 if hidden is None else int(hidden.shape[1])
            # Window-relative drafter positions, like the lane's ring: RoPE only needs
            # differences, and the drafter never saw 8K+ absolute offsets in training.
            ring.pos = mx.array([0], dtype=mx.int32)
            if count:
                ring.append(self.draft, hidden, mx.array([count], dtype=mx.int32))
            self.rings[id(row)] = ring
            from ..mtp_lane import copy_rows_for_model

            copy_rows = copy_rows_for_model(self.lm)
            if copy_rows >= DRIVER_MAX_ROWS:
                # a lone row's copy round runs to the fast tree's width (the lane's
                # rule); a pack clamps each step to DRIVER_MAX_ROWS (_step_rows_inner)
                copy_rows = tree_verify.FAST_MAX_ROWS
            if copy_rows >= 3:
                copier = CopyDrafter(max_draft=copy_rows - 1)
                copier.extend(row.context)
                self.copiers[id(row)] = copier
            row.slot = id(row)
            self.rows.append(row)
            tree_verify.reserve_decode(row.cache, row.req.max_tokens + 32)

    def finish_tail(self, row) -> None:
        """A finished row's drafter window, held for ``store_decode_window`` (None: no window)."""
        dflash_apc_window.finish_decode(self.tails.get(id(row)))

    def _tail_add(self, r, res, kept, used: int) -> None:
        tail = self.tails.get(id(r))
        if tail is None:
            return
        if (
            kept is None
        ):  # a GPU-walked row: the padded path rows, the first ``used`` valid
            rows = self.walk_rows.pop(id(r))[:, :used]
        else:
            rows = mx.concatenate(res.captured, axis=-1)[
                :, mx.array(kept, dtype=mx.int32)
            ]
        tail.add(rows, used)

    def leave(self, rows):
        gone = {id(r) for r in rows}
        self.rows = [r for r in self.rows if id(r) not in gone]
        for row in rows:
            row.slot = None
            self.rings.pop(id(row), None)
            self.copiers.pop(id(row), None)
            self.hidden.pop(id(row), None)
            self.tails.pop(id(row), None)
            self.walk_rows.pop(id(row), None)
            self.prebuilt.pop(id(row), None)
            self.budgets.pop(id(row), None)

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
            for start in range(0, len(rows), DRIVER_MAX_ROWS):
                events.extend(
                    self._step_rows(driver, rows[start : start + DRIVER_MAX_ROWS])
                )
            return events
        finally:
            lane_linear.set_sum_reuse(previous)
            tree_verify.vq.clear_group_sums()

    def _step_rows(self, driver, rows) -> list[Event]:
        t_round = time.perf_counter()
        out = self._step_rows_inner(driver, rows)
        self._diag_round(len(rows), time.perf_counter() - t_round)
        return out

    def _diag_round(self, width: int, seconds: float) -> None:
        d = self._diag
        d["rounds"] += 1
        d["rows"] += width
        d["s"] += seconds
        if d["rounds"] % 100 == 0:
            logger.info(
                "Round driver DFlash window of 100 rounds (total %d): rows/round %.2f, %.1f ms/round, "
                "nodes/row %.2f, tokens/row-round %.2f",
                self.turn,
                d["rows"] / d["rounds"],
                1000 * d["s"] / d["rounds"],
                d["nodes"] / max(1, d["rows"]),
                d["used"] / max(1, d["rows"]),
            )
            self._diag = dict(rounds=0, rows=0, s=0.0, nodes=0, used=0)

    def _step_rows_inner(self, driver, rows) -> list[Event]:
        from .driver import _Item

        t_start = time.perf_counter()

        copies = {}
        rooms = []
        # one live row verifies up to FAST_MAX_ROWS rows; a pack shares DRIVER_MAX_ROWS
        max_rows = tree_verify.FAST_MAX_ROWS if len(rows) == 1 else DRIVER_MAX_ROWS
        for b, r in enumerate(rows):
            room = max(0, r.req.max_tokens - r.generated - 1)
            if not r.drafting or r.force or _plain_guided(r):
                rooms.append(0)
                continue
            copier = self.copiers.get(id(r))
            copied = (
                copier.draft(min(copier.max_draft, room, max_rows - 1))
                if copier is not None
                else []
            )
            if (
                copied
                and copier is not None
                and (r.generated == 1 or copier.last_match < copier.cfg.confident_match)
            ):
                copied = copied[:15]
            if copied:
                copies[b] = copied
            rooms.append(len(copied) if copied else min(15, room))
        nodes = fair_nodes(rooms, budget=max_rows, turn=self.turn)
        self.turn += 1
        budget = None
        if len(rows) == 1 and rooms[0] and 0 not in copies:
            # a lone row sizes its tree as the lane does; more rows share the 32-node budget
            from ..spec_schedule import NodeBudget

            r = rows[0]
            budget = self.budgets.get(id(r))
            if budget is None:
                budget = self.budgets[id(r)] = NodeBudget(
                    15, prior=dflash_tree.PRIOR, widths=(7, 15)
                )
            nodes = [
                min(nodes[0], budget.choose(max(0, r.req.max_tokens - r.generated - 1)))
            ]
        live = [b for b, n in enumerate(nodes) if n and b not in copies]
        proposals = {}
        ready = {b: self.prebuilt.pop(id(rows[b]), None) for b in live}
        self.prebuilt.clear()  # a copy round or a row without room drops its lattice
        fresh = [b for b in live if ready[b] is None]
        if fresh:
            hs = dflash_ring.draft_hidden_batch(
                self.draft,
                [self.rings[id(rows[b])] for b in fresh],
                mx.array([rows[b].pending for b in fresh], dtype=mx.int32),
                15,
            )
            for j, b in enumerate(fresh):
                r = rows[b]
                ready[b] = dflash_tree.lattice_from_hidden(
                    self.draft,
                    hs[j : j + 1],
                    mx.array([r.pending], dtype=mx.int32),
                    keyed=r.keyed,
                    committed=r.generated,
                    penalty=r.penalty,
                )
        if live:
            for b in live:
                r = rows[b]
                lat = ready[b]
                toks, parents = search_tree(lat, nodes[b])
                toks, parents, ranks = reorder(toks, parents)
                proposals[b] = (
                    mx.concatenate([mx.array([r.pending], dtype=mx.int32), toks])[None],
                    FastShape(parents, 15, original_ranks=ranks),
                )
        for b, copied in copies.items():
            count = nodes[b]
            if count:
                copied = copied[:count]
                proposals[b] = (
                    mx.array([[rows[b].pending, *copied]], dtype=mx.int32),
                    FastShape(
                        mx.arange(-1, count, dtype=mx.int32), count, is_chain=True
                    ),
                )
        windows = []
        for b, r in enumerate(rows):
            tokens, shape = proposals.get(
                b,
                (mx.array([[r.pending]], dtype=mx.int32), tree_verify.TreeShape([-1])),
            )
            windows.append(Window(tokens, shape, r.cache, r.req.rope_delta))
        results = forward(self.lm, windows, self.capture_ids, verifier=self.verifier)
        draws = []
        committed_rows = set()
        vocab = getattr(self.draft, "_draft_vocab", None) if self.can_draft else None
        try:
            packed_logits = logits(
                self.lm, mx.concatenate([res.hidden[0] for res in results])
            )
            at = 0
            for r, w, res in zip(rows, windows, results, strict=True):
                row_logits = packed_logits[at : at + w.shape.width]
                at += w.shape.width
                if not r.drafting or r.force or _plain_guided(r):
                    it = _Item("d", r, at=0, count=1)
                    _, targets, extra = driver._draw(
                        [it], res.hidden[0], target_logits=row_logits
                    )[0]
                    draws.append((targets, None, extra))
                else:
                    drawn = row_logits[None]
                    if r.req.guide is not None and masks_every_row(r.req.guide):
                        from ..constrained_spec import plan_tree
                        from ..tool_call_grammar import apply_bitmask

                        # every position's mask depends on the tokens before it on its path
                        masks = plan_tree(
                            r.req.guide,
                            w.tokens.reshape(-1).tolist(),
                            w.shape.parents_array().tolist(),
                        )
                        drawn = apply_bitmask(drawn[0], masks)[None]
                    if r.generic:
                        # processors one window row at a time, in list order and before the
                        # guide, as the serial step draws it (Driver.window_processed)
                        from .driver import window_processed

                        toks = w.tokens.reshape(-1).tolist()
                        parents = w.shape.parents_array().tolist()
                        drawn = window_processed(
                            r.req.processors,
                            drawn[0],
                            r.context,
                            r.generated,
                            toks,
                            parents,
                        )[None]
                        penalty = None
                    else:
                        for p in r.req.processors:
                            # static bans (and a penalty: per window row below)
                            if p is not r.penalty:
                                drawn = p.pure_at(r.generated, drawn)
                        penalty = r.penalty
                    targets, lp = keyed_sampling.tree_target_rows(
                        drawn,
                        w.shape,
                        r.generated,
                        r.keyed,
                        SimpleNamespace(logprobs=r.req.logprobs),
                        penalty,
                        w.tokens,
                    )
                    draws.append((targets, lp, None))
            mx.async_eval(*[d[0] for d in draws])
            gpu_walked = self._prebuild(rows, windows, results, draws, nodes, copies)
            events: list[Event] = []
            for b, (r, w, res, (target, lp, extra), n) in enumerate(
                zip(rows, windows, results, draws, nodes, strict=True)
            ):
                ranks_dev = getattr(w.shape, "original_ranks", None)
                parents_dev = w.shape.parents_array()
                # one transfer for the round (the lane's pack_readback)
                targets, toks, parents, ranks = unpack_readback(
                    pack_readback(target, w.tokens, parents_dev, ranks_dev).tolist(),
                    int(target.size),
                    int(w.tokens.size),
                    int(parents_dev.size),
                    ranks_dev is not None,
                )
                path = dflash_tree.walk(toks, parents, targets)
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
                kept = path[:used]
                if budget is not None and b == 0:
                    landed = [row - 1 for row in path[1:]]
                    if ranks is not None:
                        landed = [x for x in remap_landed(landed, ranks) if x < n]
                    budget.observe(
                        n,
                        landed,
                        (time.perf_counter() - t_start) * 1000,
                        first=r.generated - used == 1,
                    )
                st = self.stats.setdefault(
                    id(r),
                    dict(rounds=0, nodes=0, used=0, copy=0, copy_used=0, hist=[0] * 17),
                )
                st["rounds"] += 1
                st["hist"][min(16, max(0, used - 1))] += 1
                st["nodes"] += n
                st["used"] += used
                if b in copies:
                    st["copy"] += 1
                    st["copy_used"] += used
                tree_verify.tree_commit(self.lm, r.cache, res, kept)
                committed_rows.add(id(r))
                r.n += used
                copier = self.copiers.get(id(r))
                if copier is not None:
                    copier.extend(committed[:used])
                    if b in copies and n:
                        copier.observe_copy(n, max(0, used - 1))
                    else:
                        copier.observe_model(used)
                self._diag["nodes"] += n
                self._diag["used"] += used
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
                if vocab is not None:
                    vocab.learn(committed[:used])
                self._tail_add(r, res, None if b in gpu_walked else kept, used)
                if b in gpu_walked:
                    if not r.finished and used != len(path):
                        # only a stop / length cut ends a walked row's path early (rows with a
                        # budget or a guide are never walked on the GPU)
                        raise RuntimeError(
                            "round driver: a GPU-walked DFlash row was cut short"
                        )
                elif not r.finished and self.can_draft:
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
        for r in gone:
            st = self.stats.pop(id(r), None)
            if st and st["rounds"]:
                logger.info(
                    "Round driver DFlash row: %d rounds, %.2f tokens/round, %.1f nodes/round, "
                    "%d copy rounds (%.2f tokens/copy round); accepted-length histogram %s",
                    st["rounds"],
                    st["used"] / st["rounds"],
                    st["nodes"] / st["rounds"],
                    st["copy"],
                    st["copy_used"] / max(1, st["copy"]),
                    st["hist"],
                )
        for r in gone:
            self.prebuilt.pop(id(r), None)
        driver.rows = [r for r in driver.rows if not r.finished]
        driver._release(gone)
        # queued, not waited for: the next step builds its graph while the GPU finishes this one
        # (and the prebuilt lattice), as the lane's round loop does
        mx.async_eval(
            *self.arrays(),
            *[
                x
                for lat in self.prebuilt.values()
                for x in (lat.cands, lat.unary, lat.anchor)
            ],
        )
        driver.steps += 1
        return events

    def _prebuild(self, rows, windows, results, draws, nodes, copies) -> set:
        """A lone drafting row (model tree round, no guide / budget / force): take its accepted path
        on the GPU (``walk_gpu``), append it to the drafter ring and queue the next round's lattice,
        all before the host reads the verify back -- the lane's next-round pipelining. Returns the
        rows (indices) whose ring was appended here."""
        from ..dflash_plan import walk_gpu

        if not self.can_draft or len(rows) != 1:
            return set()
        b, r = 0, rows[0]
        target, _lp, extra = draws[0]
        if (
            not r.drafting
            or r.force
            or r.req.guide is not None
            or r.req.budget is not None
            or b in copies
            or not nodes[b]
            or extra is not None
        ):
            return set()
        w, res = windows[0], results[0]
        path_g, count_g, bonus_g = walk_gpu(
            w.tokens.reshape(-1), w.shape.parents_array(), target.reshape(-1)
        )
        ring = self.rings[id(r)]
        rows_g = mx.take(mx.concatenate(res.captured, axis=-1), path_g, axis=1)
        ring.append(self.draft, rows_g, count_g)
        if id(r) in self.tails:
            self.walk_rows[id(r)] = rows_g
        lat = dflash_tree.lattice_from_hidden(
            self.draft,
            dflash_ring.draft_hidden(self.draft, ring, bonus_g, 15),
            bonus_g,
            keyed=r.keyed,
            committed=r.generated + count_g,
            penalty=r.penalty,
        )
        mx.async_eval(lat.cands, lat.unary, lat.anchor)
        self.prebuilt[id(r)] = lat
        return {b}
