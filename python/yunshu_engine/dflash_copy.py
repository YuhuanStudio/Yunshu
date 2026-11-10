# Upstream (inspired): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/dflash.py @ v0.7.4
"""Copy islands for the existing single-request greedy DFlash verifier.

The adapter changes proposals only. The upstream loop still owns target sampling,
transactions, rollback and publication. Pending target taps from skipped draft
passes are consumed once when model drafting resumes; no draft-cache work or new
readback is needed inside a copy island.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .copy_drafter import CopyDrafter
from .dflash_context import context_window

logger = logging.getLogger(__name__)


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
        return self._propose(
            self._draft.draft_block_greedy,
            bonus,
            hidden,
            cache,
            block,
            sampler,
            token_dtype,
        )

    def draft_block(self, bonus, hidden, cache, block, sampler, token_dtype=mx.int32):
        return self._propose(
            self._draft.draft_block, bonus, hidden, cache, block, sampler, token_dtype
        )

    def _propose(self, method, bonus, hidden, cache, block, sampler, token_dtype):
        if self._next_copy:
            if hidden is None:  # a promoted row's first round: nothing captured yet
                return mx.array([self._next_copy], dtype=token_dtype)
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
        return method(bonus, hidden, cache, block, sampler, token_dtype)

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


class ResumedCopyDraft(CopyDraft):
    """Continue a committed fast-tree transaction without resetting draft KV."""

    def __init__(self, draft, copy, cache):
        # The full history and copy benefit controller already belong to this
        # request. Re-indexing a truncated context would discard copy islands.
        super().__init__(draft, [], 0, copy.max_draft + 1)
        self._copy = copy
        self._resume_cache = cache
        self._seen_rounds = len(draft.accept_lens)

    def reset(self, model):
        return self._resume_cache


def resume_chain(original, model, draft, cache, hidden, *, copy, draft_cache, **kw):
    """Transfer ownership after tree_commit; the bonus is already published."""
    proxy = ResumedCopyDraft(draft, copy, draft_cache)
    iterator = original(model, proxy, cache, hidden, **kw)
    try:
        for token, state in iterator:
            proxy.emitted(int(token))
            yield token, state
    finally:
        try:
            iterator.close()
        finally:
            proxy.close()


def configure(
    language_model: Any, rows: int, *, invariant: bool, lane_projections: bool
) -> tuple[int, bool]:
    """Apply the copy-rows setting to this target; True when the measured lane is certified."""
    from . import mtp_lane

    limit = mtp_lane.verify_max_rows(lane_projections, language_model)
    effective = mtp_lane.set_copy_rows(rows, limit)
    enabled = bool(effective >= 3 and invariant and lane_projections and limit >= 16)
    return effective, enabled
