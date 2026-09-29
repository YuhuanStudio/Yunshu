"""DFlash2 GPU best-first draft tree and the acceptance walk."""

from __future__ import annotations

import heapq

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine import dflash_tree  # noqa: E402


def _reference_tree(lat, nodes, children, edge, tau):
    """Best-first search with a heap (TensorFold's algorithm), on the CPU."""
    cands, unary, hproj = (
        np.array(lat.cands),
        np.array(lat.unary),
        np.array(lat.hproj),
    )
    succ, pred, anchor = np.array(lat.succ), np.array(lat.pred), np.array(lat.anchor)

    def scores(depth, pr):
        s = (unary[depth] + edge * (succ[depth] @ (pr * hproj[depth]))) / tau
        s = s - s.max()
        return s - np.log(np.exp(s).sum())

    tokens, parents, heap = [], [], []
    root = scores(0, anchor)
    for i in np.argsort(-root)[:children]:
        heapq.heappush(heap, (-float(root[i]), -1, 0, int(i)))
    while heap and len(tokens) < nodes:
        neg, parent, depth, i = heapq.heappop(heap)
        tokens.append(int(cands[depth][i]))
        parents.append(parent)
        me = len(tokens) - 1
        if depth + 1 < cands.shape[0]:
            ls = scores(depth + 1, pred[depth][i])
            for j in np.argsort(-ls)[:children]:
                heapq.heappush(heap, (neg - float(ls[j]), me, depth + 1, int(j)))
    return tokens, [-1] + [0 if p < 0 else p + 1 for p in parents]


def _random_lattice(seed, d=7, k=16, r=12):
    rng = np.random.default_rng(seed)
    return dflash_tree.GpuLattice(
        mx.array(rng.permutation(5000)[: d * k].reshape(d, k).astype(np.uint32)),
        mx.array(rng.normal(size=(d, k)).astype(np.float32) * 3),
        mx.array(rng.normal(size=(d, r)).astype(np.float32)),
        mx.array(rng.normal(size=(d, k, r)).astype(np.float32)),
        mx.array(rng.normal(size=(d, k, r)).astype(np.float32)),
        mx.array(rng.normal(size=(r,)).astype(np.float32)),
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("nodes", [1, 3, 7])
def test_search_tree_matches_heap_best_first(seed, nodes):
    lat = _random_lattice(seed)
    tokens, parents = dflash_tree.search_tree(lat, nodes)
    ref_tokens, ref_parents = _reference_tree(
        lat, nodes, dflash_tree.CHILDREN, dflash_tree.EDGE, dflash_tree.TAU
    )
    assert tokens.tolist() == ref_tokens
    assert parents.tolist() == ref_parents


def test_search_tree_prefixes_are_trees():
    lat = _random_lattice(3)
    tokens, parents = dflash_tree.search_tree(lat, 7)
    rows = parents.tolist()
    assert rows[0] == -1 and all(0 <= rows[i] < i for i in range(1, len(rows)))
    small_tokens, small_parents = dflash_tree.search_tree(lat, 4)
    assert small_tokens.tolist() == tokens.tolist()[:4]
    assert small_parents.tolist() == rows[:5]


def test_walk_follows_matching_children():
    window = [10, 11, 12, 13, 14, 15]
    parents = [-1, 0, 0, 1, 1, 3]
    # target after row 0 picks 12; after row 2 (12) picks 99 -> path [0, 2]
    assert dflash_tree.walk(window, parents, [12, 0, 99, 0, 0, 0]) == [0, 2]
    # target picks 11, then 14, nothing follows row 4
    assert dflash_tree.walk(window, parents, [11, 14, 0, 0, 7, 0]) == [0, 1, 4]
    # deep path
    assert dflash_tree.walk(window, parents, [11, 13, 0, 15, 0, 5]) == [0, 1, 3, 5]
