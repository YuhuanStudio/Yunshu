"""Static draft-tree topologies and the DFlash2 GPU tree builder."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine import dflash_tree  # noqa: E402
from yunshu_engine.spec_topology import Topology  # noqa: E402


def test_topology_structure():
    top = Topology([(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (2,)])
    assert top.size == 6
    assert top.parents == [-1, -1, 0, 0, 2, -1]
    assert top.window_parents == [-1, 0, 0, 1, 1, 3, 0]
    assert top.max_depth == 3
    assert top.shape().parents == tuple(top.window_parents)
    assert top.prefix(3).paths == [(0,), (1,), (0, 0)]
    assert top.prefix(99) is top
    levels = top.levels()
    assert [n for n, _, _ in levels] == [[0, 1, 5], [2, 3], [4]]
    # level-2 nodes hang off node 0, level 3's off level 2's first node
    assert levels[1][1] == [0, 0] and levels[2][1] == [0]


def test_topology_rejects_open_paths():
    with pytest.raises(ValueError):
        Topology([(0, 0)])
    with pytest.raises(ValueError):
        Topology([(0,), (0, 0, 0)])


def test_permutation_restores_node_order():
    top = Topology([(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (2,)])
    level_order = [i for nodes, _, _ in top.levels() for i in nodes]
    assert [level_order[p] for p in top.permutation()] == list(range(top.size))


def _reference_tokens(lat, topo, edge, tau):
    cands, unary, hproj = (
        np.array(lat.cands),
        np.array(lat.unary),
        np.array(lat.hproj),
    )
    succ, pred, anchor = np.array(lat.succ), np.array(lat.pred), np.array(lat.anchor)
    tokens = []
    for path in topo.paths:
        cidx, pr = None, anchor
        for depth, rank in enumerate(path):
            if depth:
                pr = pred[depth - 1][cidx]
            score = (unary[depth] + edge * (succ[depth] @ (pr * hproj[depth]))) / tau
            cidx = int(np.argsort(-score)[rank])
        tokens.append(int(cands[len(path) - 1][cidx]))
    return tokens


def test_build_tokens_matches_reference_scoring():
    rng = np.random.default_rng(0)
    d, k, r = 7, 16, 12
    lat = dflash_tree.GpuLattice(
        mx.array(rng.permutation(5000)[: d * k].reshape(d, k).astype(np.uint32)),
        mx.array(rng.normal(size=(d, k)).astype(np.float32) * 3),
        mx.array(rng.normal(size=(d, r)).astype(np.float32)),
        mx.array(rng.normal(size=(d, k, r)).astype(np.float32)),
        mx.array(rng.normal(size=(d, k, r)).astype(np.float32)),
        mx.array(rng.normal(size=(r,)).astype(np.float32)),
    )
    top = Topology(
        [(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (2,), (0, 0, 0, 0), (1, 0), (1, 0, 1)]
    )
    got = dflash_tree.build_tokens(lat, top).astype(mx.int32).tolist()
    assert got == _reference_tokens(lat, top, dflash_tree.EDGE, dflash_tree.TAU)


def test_walk_follows_matching_children():
    window = [10, 11, 12, 13, 14, 15]
    parents = [-1, 0, 0, 1, 1, 3]
    # target after row 0 picks 12; after row 2 (12) picks 99 -> path [0, 2]
    assert dflash_tree.walk(window, parents, [12, 0, 99, 0, 0, 0]) == [0, 2]
    # target picks 11, then 14, nothing follows row 4
    assert dflash_tree.walk(window, parents, [11, 14, 0, 0, 7, 0]) == [0, 1, 4]
    # deep path
    assert dflash_tree.walk(window, parents, [11, 13, 0, 15, 0, 5]) == [0, 1, 3, 5]
