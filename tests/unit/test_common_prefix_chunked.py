"""common_prefix (chunked list compare) must equal the verbatim per-item loop."""

import random

from yunshu_engine.roundtrip_capture import common_prefix


def _old(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def test_identical_to_item_loop():
    rng = random.Random(5)
    for _ in range(2000):
        n = rng.choice([0, 1, 255, 256, 257, 511, 512, 513, 1000, 3000])
        a = [rng.randrange(50000) for _ in range(n)]
        b = list(a)
        mode = rng.choice(["same", "flip", "short", "long"])
        if mode == "flip" and n:
            b[rng.randrange(n)] += 1
        elif mode == "short":
            b = b[: rng.randrange(n + 1)]
        elif mode == "long":
            b = b + [1, 2, 3]
        assert common_prefix(a, b) == _old(a, b)
        assert common_prefix(b, a) == _old(b, a)
