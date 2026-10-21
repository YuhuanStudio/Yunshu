"""dflash_plan.walk_gpu equals the host walk (dflash_tree.walk) on random trees and chains."""

import random

import mlx.core as mx
import pytest

from yunshu_engine import dflash_tree as dt
from yunshu_engine.dflash_plan import walk_gpu


def _case(rng, width, vocab, chain):
    parents = [-1] + [r - 1 if chain else rng.randrange(r) for r in range(1, width)]
    window = [rng.randrange(vocab) for _ in range(width)]
    target = [rng.randrange(vocab) for _ in range(width)]
    # plant accepted children so paths are long
    for r in range(1, width):
        if rng.random() < 0.6:
            target[parents[r]] = window[r]
    return window, parents, target


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("width", [1, 2, 16, 17, 32])
def test_walk_gpu_matches_host(width, chain):
    rng = random.Random(width * 2 + chain)
    for _ in range(40):
        window, parents, target = _case(rng, width, 5, chain)
        want = dt.walk(window, parents, target)
        path, count, bonus = walk_gpu(
            mx.array(window), mx.array(parents), mx.array(target)
        )
        n = int(count.item())
        assert path.tolist()[:n] == want
        assert path.tolist()[n:] == [0] * (width - n)
        assert int(bonus.item()) == target[want[-1]]
