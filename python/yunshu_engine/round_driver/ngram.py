"""Model-free drafter of the round driver: n-gram / suffix prompt lookup on the generic layer.

The proposers (``ngram_proposer.NgramProposer``, ``suffix_proposer.SuffixProposer``) are the ones the
text engine uses, unchanged. Here they run on the host after each emit: the row's context (prompt +
emitted tokens, ending with the pending token) proposes the tokens that followed its longest earlier
match, the driver feeds ``[pending, *drafts]`` as one window and keeps the prefix whose top-1 equals
the draft (``RoundDriver._decode_step``, the same verification as every other window).

Guarantee (user decision 2026-11-07, option A): on the generic layer a window is one upstream T>1
forward, so the guarantee is the old verifier's: every kept token is the model's own top-1 over that
window; it is NOT bitwise equal to T=1 decode on stock kernels (a near-tie may round differently).
That is why n-gram speculation there is a documented opt-in (per-request ``spec_decode`` or
``YUNSHU_NGRAM_DEFAULT``), greedy rows only, and why its replies are cached under their own APC
namespace. Packed Qwen3.5 / Flash-Next rows never take this drafter: their windows are
batch-invariant tree verifies (DFlash copy drafts, MTP) and stay on that path.
"""

from __future__ import annotations

MAX_DRAFT = 5  # drafts per window (the text engine's k)
MAX_COOL = 8
WARM_STEPS = 3  # plain steps first: they measure what one token costs on this row
SLOW_COOL = (
    24  # steps a row decodes plain after its windows measured slower than plain steps
)
EMA = 0.3
MAX_MODEL_LEN = (
    1 << 24
)  # the proposers' own 32K default would silence drafts on a longer prompt  # steps a row sits out after repeated misses


def make_proposer(kind: str):
    """A fresh per-row proposer of the configured family (``YUNSHU_SPEC_PROPOSER``)."""
    if kind == "suffix":
        from ..suffix_proposer import SuffixConfig, SuffixProposer

        return SuffixProposer(
            SuffixConfig(
                max_draft=MAX_DRAFT, max_trie_depth=80, max_model_len=MAX_MODEL_LEN
            )
        )
    from ..ngram_proposer import NgramConfig, NgramProposer

    return NgramProposer(
        NgramConfig(max_n=5, k=MAX_DRAFT, mode="lps", max_model_len=MAX_MODEL_LEN)
    )


def _ema(old: float, new: float) -> float:
    return new if not old else old + EMA * (new - old)


class NgramDrafter:
    """One row's proposer plus its back-off. A window with no accepted draft costs a wider verify for
    one token, so a row that keeps missing proposes less often (resets on any hit). Windows also cost
    more than a plain step (more expert weights, a T>1 attention over a long cache): the row measures
    both and decodes plain for a while when tokens per second through windows fall below plain."""

    name = "n-gram"  # log label of the row
    mode = "ngram"  # spec_mode of the row
    needs_hidden = False  # True: the driver hands this drafter the target's hidden state (``carry``)
    needs_target_cache = False

    def __init__(self, kind: str = "ngram", *, proposer=None) -> None:
        self.kind = kind
        self.k = MAX_DRAFT
        self.proposer = proposer if proposer is not None else make_proposer(kind)
        self.misses = 0
        self.cool = 0
        self.windows = self.drafted = self.landed = 0
        self.steps = 0
        self.plain_s = 0.0  # EMA seconds of a plain step
        self.win_s = 0.0  # EMA seconds of a window step
        self.win_tok = 0.0  # EMA tokens a window step kept

    def propose(self, context: list[int], room: int) -> list[int]:
        """Up to ``room`` draft tokens after ``context[-1]``; empty while cooling down."""
        if room <= 0 or self.steps < WARM_STEPS:
            return []
        if self.cool > 0:
            self.cool -= 1
            return []
        if self.slow():
            self.cool = SLOW_COOL
            # forget the windows' timings: after the cool-down one fresh window is measured again, so a
            # single early miss cannot switch the row's drafting off for good
            self.win_s = self.win_tok = 0.0
            return []
        return [int(t) for t in self._propose(context, room)[:room]]

    def _propose(self, context: list[int], room: int) -> list[int]:
        return list(self.proposer.propose(context))

    def slow(self) -> bool:
        """Windows measured slower per token than plain steps (both measured on this row)."""
        if not (self.plain_s and self.win_s and self.win_tok):
            return False
        return self.win_tok / self.win_s < 1.0 / self.plain_s

    def observe_step(self, windowed: bool, tokens: int, seconds: float) -> None:
        """One lone-row decode step took ``seconds`` and kept ``tokens``."""
        self.steps += 1
        if windowed:
            self.win_s = _ema(self.win_s, seconds)
            self.win_tok = _ema(self.win_tok, float(tokens))
        elif self.steps > 1:  # the first step after a prompt carries one-off costs
            self.plain_s = _ema(self.plain_s, seconds)

    def observe(self, drafted: int, landed: int) -> None:
        if not drafted:
            return
        self.windows += 1
        self.drafted += drafted
        self.landed += landed
        if landed:
            self.misses = 0
            return
        self.misses += 1
        self.cool = min((1 << self.misses) - 1, MAX_COOL)
