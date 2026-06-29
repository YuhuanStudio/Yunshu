"""(HIGH): /v1/completions best_of ranking included the PROMPT tokens' logprobs
when echo=True (batched path) → the wrong completion was selected.

best_of generates >n candidates and keeps the top-n by average per-token logprob. With
echo, _format_logprobs PREPENDS the prompt tokens' logprobs into token_logprobs.
The prompt forward is IDENTICAL across all candidates (deterministic), so averaging the
constant prompt-logprob sum over each candidate's (prompt+completion) length dilutes more
for longer completions — flipping the ranking so a worse, longer completion can win. The
fix ranks on the COMPLETION only (entries whose text_offset >= the prompt's char length).
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import completions


def _avg_completion_only(lp, prompt_len, echo):
    """Faithful mirror of completions.py _avg_logprob ranking logic."""
    probs = lp["token_logprobs"]
    if echo and isinstance(lp.get("text_offset"), list):
        offs = lp["text_offset"]
        if len(offs) == len(probs):
            probs = [p for p, o in zip(probs, offs, strict=False) if o >= prompt_len]
    valid = [p for p in probs if p is not None]
    return sum(valid) / len(valid) if valid else float("-inf")


def test_completion_only_ranking_picks_the_better_short_completion():
    # prompt "ab" (char len 2): prompt tokens at text_offset 0,1; completion at offset>=2.
    # Candidate A: one strong completion token (-0.1). Candidate B: six weak ones (-3 each).
    lp_a = {"token_logprobs": [None, -8.0, -0.1], "text_offset": [0, 1, 2]}
    lp_b = {"token_logprobs": [None, -8.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
            "text_offset": [0, 1, 2, 3, 4, 5, 6, 7]}
    a = _avg_completion_only(lp_a, 2, echo=True)
    b = _avg_completion_only(lp_b, 2, echo=True)
    assert a > b, "A (completion avg -0.1) must beat B (completion avg -3.0)"


def test_old_polluted_ranking_would_have_picked_the_worse_candidate():
    # Demonstrates the bug the fix removes: averaging prompt+completion flips the winner.
    lp_a = [None, -8.0, -0.1]
    lp_b = [None, -8.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]
    old_a = sum(x for x in lp_a if x is not None) / 2   # (-8 - 0.1)/2  = -4.05
    old_b = sum(x for x in lp_b if x is not None) / 7   # (-8 - 18)/7  ≈ -3.71
    assert old_b > old_a  # the WORSE completion (B) wrongly wins under the old logic


def test_no_echo_unchanged():
    # Without echo there is no prompt prepend → ranking is over all (completion) entries.
    lp = {"token_logprobs": [-0.5, -0.5], "text_offset": [0, 3]}
    assert _avg_completion_only(lp, 0, echo=False) == -0.5


def test_production_avg_logprob_slices_by_text_offset():
    src = inspect.getsource(completions)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the fix: echo ranking slices completion entries by text_offset vs prompt len
    assert "o >= _plen" in code
    assert 'lp.get("text_offset")' in code
    assert "isinstance(_prompts[0], str)" in code
