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


class NgramDrafter:
    """One row's proposer plus its back-off: a window with no accepted draft costs a wider verify
    for one token, so a row that keeps missing proposes less often (resets on any hit)."""

    def __init__(self, kind: str = "ngram") -> None:
        self.kind = kind
        self.k = MAX_DRAFT
        self.proposer = make_proposer(kind)
        self.misses = 0
        self.cool = 0

    def propose(self, context: list[int], room: int) -> list[int]:
        """Up to ``room`` draft tokens after ``context[-1]``; empty while cooling down."""
        if room <= 0:
            return []
        if self.cool > 0:
            self.cool -= 1
            return []
        return [int(t) for t in self.proposer.propose(context)[:room]]

    def observe(self, drafted: int, landed: int) -> None:
        if not drafted:
            return
        if landed:
            self.misses = 0
            return
        self.misses += 1
        self.cool = min((1 << self.misses) - 1, MAX_COOL)
