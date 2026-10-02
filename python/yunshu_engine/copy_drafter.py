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


@dataclass
class CopyConfig:
    ngram: int = 3  # tokens hashed into the index key
    min_match: int = 6  # backward-matching tokens needed to propose
    confident_match: int = 24  # a match this long always proposes, full width
    first_width: int = 4  # drafts offered on a short match / after a miss
    max_extension: int = 64  # backward comparison cap
    max_candidates: int = 24  # most recent occurrences compared per proposal
    benefit_ratio: float = 1.0  # copy tokens/round must reach this x model's
    backoff_cap: int = 16  # longest silence, in rounds
    ema: float = 0.5


class CopyDrafter:
    name = "copy"

    def __init__(self, config: CopyConfig | None = None, *, max_draft: int = 7) -> None:
        self.cfg = config or CopyConfig()
        if self.cfg.min_match < self.cfg.ngram:
            raise ValueError("min_match must be at least ngram")
        self.max_draft = max(1, int(max_draft))
        self.ctx: list[int] = []
        self._index: dict[tuple[int, ...], list[int]] = {}
        self._indexed = 0
        self._silent = 0
        self._misses = 0
        self._wide = False
        self._copy_gain = float(self.max_draft)  # optimistic until judged
        self._model_tpr = 3.0
        self.last_match = 0
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.committed = 0  # tokens committed by copy rounds

    # -- index ---------------------------------------------------------
    def extend(self, tokens: Sequence[int]) -> None:
        """Append tokens (the prompt first, then each committed token)."""
        self.ctx.extend(int(t) for t in tokens)

    def _sync_index(self) -> None:
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
        self._sync_index()
        cands = self._index.get(tuple(ctx[size - n :]))
        if not cands:
            return []
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
        self.last_match = best_len
        if best_end < 0:
            return []
        return ctx[best_end : best_end + width]

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
        if not confident and self._copy_gain < self._model_tpr * cfg.benefit_ratio:
            return []
        if not (confident or self._wide):
            prop = prop[: cfg.first_width]
        return prop

    def observe_copy(self, proposed: int, accepted: int) -> None:
        cfg = self.cfg
        self.rounds += 1
        self.proposed += proposed
        self.accepted += accepted
        self.committed += accepted + 1
        self._copy_gain += cfg.ema * ((accepted + 1) - self._copy_gain)
        self._wide = accepted >= proposed  # a fully verified copy earns width
        if accepted == 0:
            self._misses += 1
            self._silent = min(2 ** (self._misses - 1), cfg.backoff_cap)
        elif accepted >= 2:
            self._misses = 0

    def observe_model(self, committed: int) -> None:
        self._model_tpr += self.cfg.ema * (committed - self._model_tpr)

    def telemetry(self) -> dict:
        return {
            "rounds": self.rounds,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "committed": self.committed,
        }
