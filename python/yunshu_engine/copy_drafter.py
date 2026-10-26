# Upstream (derived): ashhart/TensorFold (MIT) src/tensorfold/engine/lane_engine.py SuffixLookupProposer @ 34bae79a
# Upstream (inspired): raullenchai/Rapid-MLX prompt-lookup copy drafts (index over the full prompt)
"""Prompt-copy / lookup drafting for the speculative lane.

Agent and code-editing traffic quotes its own context: a tool result is echoed
into an edit, a file is rewritten with a few lines changed, a second turn
continues the first verbatim. A draft that *copies* the continuation of the
longest earlier occurrence of the current tail then lands many tokens per
verify where the model-driven draft lands two to five.

The drafter only proposes. Every proposed token is verified by the same
batch-invariant verify the model draft goes through (greedy: argmax compare;
sampled: the keyed sampler compare), so the committed stream is exactly the
non-speculative stream whatever is proposed; a wrong copy costs a round, never
a token.

``CopyDrafter`` owns three things:

- the index: n-gram -> end positions over the FULL request context (the whole
  prompt, not only the tail after a prefix-cache hit, plus every committed
  token), extended incrementally;
- the proposal: longest backward-matching earlier occurrence, ``<= max_draft``
  tokens (the verify window minus one, a parameter);
- the round choice: copy only when the match is long enough and a windowed
  benefit estimate (tokens a copy round commits vs tokens a model round
  commits) says it pays, with exponential backoff after misses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .copy_cost import CopyCosts


_TOKEN_BITS = 21
_TOKEN_LIMIT = 1 << _TOKEN_BITS
_HEAD = (
    8  # backward steps compared for every candidate before only the survivors continue
)


@dataclass
class CopyConfig:
    ngram: int = 3  # tokens hashed into the index key
    # backward-matching tokens needed to propose. 4 replayed as 47 -> 42 rounds on a 32K code
    # reply but measured 179 -> 166 tok/s (more 32-row copy rounds, each ~71 ms against a
    # ~47 ms model round, no fewer rounds): the replay counted rounds, not their cost.
    min_match: int = 6
    confident_match: int = 24  # a match this long always proposes, full width
    first_width: int = 4  # drafts offered on a short match / after a miss
    max_extension: int = 64  # backward comparison cap
    # most recent occurrences compared per proposal. 24 missed the long earlier match on 3
    # of 45 rounds of a 32K code reply (tf110 1021c replay: 88 tokens copied short)
    max_candidates: int = 256
    benefit_ratio: float = 1.0  # copy tokens/round must reach this x model's
    backoff_cap: int = 16  # longest silence, in rounds
    ema: float = 0.5


class CopyDrafter:
    name = "copy"

    def __init__(
        self,
        config: CopyConfig | None = None,
        *,
        max_draft: int = 7,
        costs: CopyCosts | None = None,
    ) -> None:
        self.cfg = config or CopyConfig()
        self.costs = costs
        if self.cfg.min_match < self.cfg.ngram:
            raise ValueError("min_match must be at least ngram")
        self.max_draft = max(1, int(max_draft))
        self.ctx: list[int] = []
        self._index: dict[tuple[int, ...], list[int]] = {}
        self._indexed = 0
        # vectorized index: packed n-gram keys (21 bits per token) per end position
        n = self.cfg.ngram
        self._vec = n >= 1 and n * _TOKEN_BITS <= 63
        self._tok = np.zeros(1024, dtype=np.int64)
        self._keys = np.zeros(1024, dtype=np.uint64)
        self._vsynced = 0  # ctx tokens already copied into the numpy buffers
        self._silent = 0
        self._misses = 0
        self._wide = False
        self._copy_gain = float(self.max_draft)  # optimistic until judged
        self._model_tpr = 3.0
        self.last_match = 0
        # the longest-match search depends only on the context, so a second lookup of the same
        # tail (a graft lookup after the round's draft, a different width) reuses it:
        # (context length, extend count) -> (best end, best length)
        self._memo_key: tuple[int, int] | None = None
        self._memo: tuple[int, int] = (-1, 0)
        self._extends = 0
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.committed = 0  # tokens committed by copy rounds

    # -- index ---------------------------------------------------------
    def extend(self, tokens: Sequence[int]) -> None:
        """Append tokens (the prompt first, then each committed token)."""
        self.ctx.extend(int(t) for t in tokens)
        self._extends += 1

    def _sync_vec(self) -> bool:
        """Grow the numpy token/key buffers; False if the dict path must be used."""
        n, ctx = self.cfg.ngram, self.ctx
        size, old = len(ctx), self._vsynced
        if size > old:
            new = np.asarray(ctx[old:size], dtype=np.int64)
            if int(new.min()) < 0 or int(new.max()) >= _TOKEN_LIMIT:
                self._vec = False
                return False
            if size + 1 > len(self._tok):
                cap = len(self._tok)
                while cap < size + 1:
                    cap *= 2
                for name in ("_tok", "_keys"):
                    arr = getattr(self, name)
                    grown = np.zeros(cap, dtype=arr.dtype)
                    grown[: len(arr)] = arr
                    setattr(self, name, grown)
            tok = self._tok
            tok[old:size] = new
            lo = max(old + 1, n)  # first end whose key gained a new token
            if size >= lo:
                acc = np.zeros(size - lo + 1, dtype=np.uint64)
                for j in range(n):
                    part = tok[lo - n + j : size - n + j + 1].astype(np.uint64)
                    acc |= part << np.uint64(_TOKEN_BITS * (n - 1 - j))
                self._keys[lo : size + 1] = acc
            self._vsynced = size
        return True

    def _sync_index(self) -> None:
        if self._vec and self._sync_vec():
            return
        n = self.cfg.ngram
        ctx = self.ctx
        # the tail key must not index itself: positions are "end of a key whose
        # continuation exists", so the last ngram of ctx waits for its next token
        for end in range(max(self._indexed, n), len(ctx)):
            self._index.setdefault(tuple(ctx[end - n : end]), []).append(end)
        self._indexed = max(self._indexed, len(ctx))

    # -- proposal ------------------------------------------------------
    def lookup(self, max_draft: int | None = None) -> list[int]:
        """Longest-match continuation of the context tail (no gating)."""
        cfg, ctx = self.cfg, self.ctx
        width = self.max_draft if max_draft is None else max_draft
        n, size = cfg.ngram, len(ctx)
        self.last_match = 0
        if width <= 0 or size < n + 1:
            return []
        key = (size, self._extends)
        if self._memo_key != key:
            self._memo = self._search(size)
            self._memo_key = key
        best_end, best_len = self._memo
        self.last_match = best_len
        return ctx[best_end : best_end + width] if best_end >= 0 else []

    def _search(self, size: int) -> tuple[int, int]:
        """(end, length) of the longest earlier match of the context tail, (-1, 0) if none."""
        cfg, ctx, n = self.cfg, self.ctx, self.cfg.ngram
        self._sync_index()
        if self._vec:
            ends = np.flatnonzero(self._keys[n:size] == self._keys[size]) + n
            ends = ends[ends < size][-cfg.max_candidates :]
            if not len(ends):
                return -1, 0
            return self._longest(ends, size)
        cands = self._index.get(tuple(ctx[size - n :])) or []
        if not cands:
            return -1, 0
        best_end, best_len = -1, 0
        for end in reversed(cands[-cfg.max_candidates :]):
            if end >= size:
                continue
            limit = min(cfg.max_extension, end)
            k = 0
            while k < limit and ctx[end - 1 - k] == ctx[size - 1 - k]:
                k += 1
            if k > best_len:
                best_len, best_end = k, end
                if k >= cfg.max_extension:
                    break
        return best_end, best_len

    def _longest(self, ends: np.ndarray, size: int) -> tuple[int, int]:
        """(end, length) of the longest backward match among ``ends``, the most recent on a
        tie: the scalar loop's choice, for every candidate at once.

        Most candidates share only the n-gram key, so the first ``_HEAD`` steps are compared
        for all of them and only the full-length survivors go on to the rest of the reach."""
        reach = self.cfg.max_extension
        head = min(reach, _HEAD)
        lengths = self._match_lengths(ends, size, 0, head)
        if reach > head:
            alive = np.flatnonzero(lengths == head)
            if len(alive):
                lengths[alive] += self._match_lengths(ends[alive], size, head, reach)
        pick = len(ends) - 1 - int(np.argmax(lengths[::-1]))
        best = int(lengths[pick])
        return (int(ends[pick]), best) if best > 0 else (-1, 0)

    def _match_lengths(
        self, ends: np.ndarray, size: int, lo: int, hi: int
    ) -> np.ndarray:
        """Leading run of equal tokens per end over backward steps ``lo..hi - 1``."""
        tok = self._tok
        steps = np.arange(lo, hi)
        back = ends[:, None] - 1 - steps[None, :]  # [C, hi - lo]
        tail = size - 1 - steps
        valid = (back >= 0) & (tail >= 0)[None, :]
        same = valid & (tok[np.maximum(back, 0)] == tok[np.maximum(tail, 0)][None, :])
        lengths: np.ndarray = np.cumprod(same, axis=1).sum(axis=1)
        return lengths

    # -- round choice --------------------------------------------------
    def draft(self, max_draft: int | None = None) -> list[int]:
        """The copy draft for this round, or ``[]`` to use the model's draft."""
        cfg = self.cfg
        if self._silent > 0:
            self._silent -= 1
            return []
        cap = self.max_draft if max_draft is None else min(self.max_draft, max_draft)
        prop = self.lookup(cap)
        if len(prop) < 2 or self.last_match < cfg.min_match:
            return []
        confident = self.last_match >= cfg.confident_match
        if (
            self.costs is None
            and not confident
            and self._copy_gain < self._model_tpr * cfg.benefit_ratio
        ):
            return []
        if not (confident or self._wide):
            prop = prop[: cfg.first_width]
        if self.costs is not None:
            prop = prop[
                : self.costs.choose(len(prop), self._model_tpr, confident=confident)
            ]
        return prop

    def observe_copy(
        self, proposed: int, accepted: int, elapsed_ms: float | None = None
    ) -> None:
        cfg = self.cfg
        self.rounds += 1
        self.proposed += proposed
        self.accepted += accepted
        self.committed += accepted + 1
        if self.costs is not None and elapsed_ms is not None:
            self.costs.observe_copy(
                proposed,
                accepted,
                elapsed_ms,
                confident=self.last_match >= cfg.confident_match,
            )
        self._copy_gain += cfg.ema * ((accepted + 1) - self._copy_gain)
        self._wide = accepted >= proposed  # a fully verified copy earns width
        if accepted == 0:
            self._misses += 1
            self._silent = min(2 ** (self._misses - 1), cfg.backoff_cap)
        elif accepted >= 2:
            self._misses = 0

    def observe_model(self, committed: int, elapsed_ms: float | None = None) -> None:
        self._model_tpr += self.cfg.ema * (committed - self._model_tpr)
        if self.costs is not None and elapsed_ms is not None:
            self.costs.observe_model(committed, elapsed_ms)

    def telemetry(self) -> dict:
        return {
            "rounds": self.rounds,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "committed": self.committed,
        }
