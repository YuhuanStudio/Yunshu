# Upstream (extended): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/dflash.py @ v0.7.4
"""Copy islands for the existing single-request greedy DFlash verifier.

The adapter changes proposals only. The upstream loop still owns target sampling,
transactions, rollback and publication. Pending target taps from skipped draft
passes are consumed once when model drafting resumes; no draft-cache work or new
readback is needed inside a copy island.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx

from .copy_drafter import CopyDrafter
from .dflash_context import context_window


class CopyDraft:
    """Request-local proxy; counters stay on the real, engine-owned drafter."""

    def __init__(self, draft: Any, context: list[int], bonus: int, rows: int):
        object.__setattr__(self, "_draft", draft)
        object.__setattr__(self, "_copy", CopyDrafter(max_draft=rows - 1))
        object.__setattr__(self, "_pending_hidden", None)
        object.__setattr__(self, "_next_copy", [])
        object.__setattr__(self, "_previous", None)
        object.__setattr__(self, "_seen_rounds", 0)
        object.__setattr__(self, "_copy_counted", False)
        object.__setattr__(self, "_pending_skipped", 0)
        object.__setattr__(self, "_window", context_window(draft))
        self._copy.extend(context)
        self._copy.extend([bonus])

    def __getattr__(self, name):
        return getattr(self._draft, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._draft, name, value)

    def choose(self, block: int, remaining: int) -> int:
        self.observe()
        self._next_copy = self._copy.draft(min(self._copy.max_draft, remaining - 1))
        width = len(self._next_copy) + 1 if self._next_copy else min(block, remaining)
        self._previous = (bool(self._next_copy), width - 1)
        self._copy_counted = False
        return width

    def observe(self):
        rounds = len(self._draft.accept_lens)
        if rounds <= self._seen_rounds or self._previous is None:
            return
        accepted = int(self._draft.accept_lens[-1])
        copied, proposed = self._previous
        if copied:
            self._copy.observe_copy(proposed, accepted)
        else:
            self._copy.observe_model(accepted + 1)
        self._seen_rounds = rounds

    def draft_block_greedy(
        self, bonus, hidden, cache, block, sampler, token_dtype=mx.int32
    ):
        if self._next_copy:
            self._pending_hidden = (
                hidden
                if self._pending_hidden is None
                else mx.concatenate([self._pending_hidden, hidden], axis=1)
            )
            if (
                self._window is not None
                and self._pending_hidden.shape[1] > self._window
            ):
                skipped = int(self._pending_hidden.shape[1]) - self._window
                self._pending_skipped += skipped
                self._pending_hidden = self._pending_hidden[:, skipped:]
            return mx.array([self._next_copy], dtype=token_dtype)
        if self._pending_hidden is not None:
            hidden = mx.concatenate([self._pending_hidden, hidden], axis=1)
            self._pending_hidden = None
            for entry in cache:
                entry.offset += self._pending_skipped
            self._pending_skipped = 0
        return self._draft.draft_block_greedy(
            bonus, hidden, cache, block, sampler, token_dtype
        )

    def emitted(self, token: int):
        self._copy.extend([token])
        if self._previous is not None and self._previous[0]:
            if not self._copy_counted:
                self._draft.copy_total_rounds = (
                    getattr(self._draft, "copy_total_rounds", 0) + 1
                )
                self._copy_counted = True
            self._draft.copy_total_tokens = (
                getattr(self._draft, "copy_total_tokens", 0) + 1
            )

    def close(self):
        self.observe()
        self._pending_hidden = None
        self._next_copy = []
        self._pending_skipped = 0


def install() -> bool:
    """Install once; unsupported and sampled requests retain their original loop."""
    from mlx_vlm.speculative import dflash, utils

    from .spec_schedule import install_chain_budget

    install_chain_budget()
    current = utils._dflash_rounds
    if getattr(current, "_yunshu_copy", False):
        return True
    if getattr(current, "_yunshu_tree", False):
        return False  # The explicitly selected tree loop owns its proposals.
    choose = dflash._dflash_next_block_size

    def next_block(draft, block, remaining, initial=None):
        if isinstance(draft, CopyDraft):
            return draft.choose(block, remaining)
        return choose(draft, block, remaining, initial)

    def rounds(model, draft, cache, hidden, **kw):
        from mlx_vlm.speculative.common import _dflash_block_total

        from . import mtp_lane, tree_verify

        lm = getattr(model, "language_model", model)
        context = mtp_lane._STATE["context"]
        rows = mtp_lane.copy_rows_for_model(lm)
        if (
            not kw.get("greedy_sampling", True)
            or context is None
            or rows < 3
            or mtp_lane._STATE["guide"] is not None
            or not hasattr(draft, "candidate_selector")
            or callable(getattr(draft, "prepare_target_hidden", None))
            or context_window(draft) is None
            or not tree_verify.supported(lm)
            or not tree_verify.lane_projections(lm)
            or not tree_verify.lane_ready(lm, cache)
            or mtp_lane.verify_max_rows(True, lm) < 16
            or _dflash_block_total(draft, kw.get("draft_block_size"))
            > mtp_lane.verify_max_rows(tree_verify.lane_projections(lm), lm)
        ):
            yield from current(model, draft, cache, hidden, **kw)
            return
        proxy = CopyDraft(draft, context, int(kw["first_bonus"]), rows)
        iterator = current(model, proxy, cache, hidden, **kw)
        try:
            for token, state in iterator:
                proxy.emitted(int(token))
                yield token, state
        finally:
            try:
                iterator.close()
            finally:
                proxy.close()

    # Keep the original depth controller installed across engine reloads.
    cast(Any, next_block)._yunshu_budget = True
    cast(Any, rounds)._yunshu_copy = True
    utils._dflash_rounds = rounds
    dflash._dflash_next_block_size = next_block
    return True


def configure(
    language_model: Any, rows: int, *, invariant: bool, lane_projections: bool
) -> tuple[int, bool]:
    """Apply the existing setting to this target and install only the measured lane."""
    from . import mtp_lane

    limit = mtp_lane.verify_max_rows(lane_projections, language_model)
    effective = mtp_lane.set_copy_rows(rows, limit)
    enabled = bool(
        effective >= 3 and invariant and lane_projections and limit >= 16 and install()
    )
    return effective, enabled
