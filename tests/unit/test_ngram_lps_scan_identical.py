"""The LPS proposer scan (list.index jumps) must equal the verbatim old range scan."""

import random

from yunshu_engine.ngram_proposer import _find_longest_ngram_and_propose


def _old(token_ids, min_n, max_n, max_model_len, k):
    total = len(token_ids)
    if total < min_n + 1:
        return []
    k = min(k, max_model_len - total)
    if k <= 0:
        return []
    _SCAN_WINDOW = 4096
    for n in range(min(max_n, total - 1), min_n - 1, -1):
        suffix = token_ids[total - n :]
        start = max(0, total - n - _SCAN_WINDOW)
        for i in range(start, total - n):
            if token_ids[i : i + n] == suffix:
                cont_start = i + n
                cont_end = min(cont_start + k, total)
                return token_ids[cont_start:cont_end]
    return []


def test_random_sequences_identical():
    rng = random.Random(1234)
    for _ in range(3000):
        vocab = rng.choice([2, 3, 5, 20, 1000])
        length = rng.choice([0, 1, 2, 3, 8, 50, 300, 4200, 5000])
        toks = [rng.randrange(vocab) for _ in range(length)]
        min_n = rng.choice([1, 2, 3])
        max_n = rng.choice([3, 5])
        k = rng.choice([1, 3, 5])
        mml = rng.choice([length + 2, 32768, length])
        assert _find_longest_ngram_and_propose(toks, min_n, max_n, mml, k) == _old(
            toks, min_n, max_n, mml, k
        )


def test_window_edge_identical():
    toks = [7, 8, 9] + [0] * 4100 + [7, 8, 9]
    assert _find_longest_ngram_and_propose(toks, 1, 5, 99999, 4) == _old(toks, 1, 5, 99999, 4)
