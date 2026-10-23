"""Round-trip token consistency for the generation capture (``YUNSHU_DEBUG_STREAM_CAPTURE``).

An agent client re-sends the conversation each turn, so the next prompt contains what the engine just
generated, re-rendered through the chat template. The engine can reuse its own decode-time state for
that span only where the re-rendered tokens equal the generated ones. This tracker keeps the last few
``prompt + generated`` token sequences in memory and, for each new prompt, reports the longest common
prefix against the best earlier sequence, so the share of generated tokens that a later request
reproduces is measurable without writing prompts anywhere. Numbers and token ids only.
"""

from __future__ import annotations

from collections import deque


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class RoundTripTracker:
    def __init__(self, keep: int = 8) -> None:
        self._seqs: deque[tuple[int, list[int], int]] = deque(maxlen=keep)
        self._n = 0

    def observe_prompt(self, prompt: list[int]) -> dict | None:
        """Match ``prompt`` against the remembered sequences; the best match as a record, else None.

        ``reusable`` is how many of the earlier response's generated tokens the new prompt reproduces in
        place (a prefix match that runs through the earlier prompt into its generation).
        """
        best = None
        for idx, seq, plen in self._seqs:
            lcp = common_prefix(seq, prompt)
            # only a continuation of that earlier prompt counts as the same conversation
            if lcp < plen:
                continue
            if best is None or lcp > best[0]:
                best = (lcp, idx, seq, plen)
        if best is None:
            return None
        lcp, idx, seq, plen = best
        gen_len = len(seq) - plen
        rec = {
            "prev_index": idx,
            "prev_prompt_len": plen,
            "prev_gen_len": gen_len,
            "matched_gen": min(lcp - plen, gen_len),
            "whole_gen_reproduced": lcp >= len(seq),
        }
        if lcp < len(seq) and lcp < len(prompt):
            # first divergence: a few ids from each side, enough to see what the template changed
            rec["diverge_at_gen"] = lcp - plen
            rec["prev_ids"] = seq[lcp : lcp + 6]
            rec["new_ids"] = prompt[lcp : lcp + 6]
        return rec

    def add(self, prompt: list[int], generated: list[int]) -> int:
        self._n += 1
        self._seqs.append((self._n, list(prompt) + list(generated), len(prompt)))
        return self._n
