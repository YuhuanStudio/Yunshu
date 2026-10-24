"""Synthetic identity gates for position-keyed speculative windows."""

from itertools import product
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.constrained_spec import normalize_rows
from yunshu_engine.dflash_tree import walk
from yunshu_engine.keyed_sampling import (
    KeyedSampler,
    coupled_scores,
    tree_supported,
    tree_target_rows,
)
from yunshu_engine.tree_verify import TreeShape


def params(**kw):
    values = dict(temperature=0.7, top_p=1.0, top_k=0, min_p=0.0, xtc_probability=0.0)
    values.update(kw)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "temperature,top_p,top_k,min_p",
    list(product((0.7, 1.0), (1.0, 0.95), (0, 20), (0.0, 0.1))),
)
@pytest.mark.parametrize(
    "fast,parents",
    [
        (False, [-1, 0, 0, 1, 2, 3]),
        (True, [-1, 0, 0, 1, 2, 3]),
        (True, [-1, 0, 1, 2, 3, 4]),
    ],
)
def test_tree_and_copy_verify_match_serial(
    temperature, top_p, top_k, min_p, fast, parents
):
    p = params(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
    shape = TreeShape(parents)
    depths = shape.depths
    if fast:
        from yunshu_engine.dflash_plan import FastShape

        shape = FastShape(
            mx.array(parents), max(depths), is_chain=parents == [-1, 0, 1, 2, 3, 4]
        )
    logits = mx.array(
        np.random.default_rng(117).normal(size=(1, len(parents), 32)).astype(np.float32)
    )
    probs = normalize_rows(logits)
    for seed in range(12):
        sampler = KeyedSampler(p, seed)
        actual, returned_probs = tree_target_rows(logits, shape, 19, sampler)
        expected = [
            int(sampler.sample_positions(probs[i : i + 1], [19 + depth])[0])
            for i, depth in enumerate(depths)
        ]
        assert actual.tolist() == expected
        assert np.array_equal(np.array(returned_probs), np.array(probs))
        assert sampler._next == 0


def test_siblings_share_noise_even_after_storage_reordering():
    shape = TreeShape([-1, 0, 0, 2, 1])
    row = mx.array(np.random.default_rng(19).normal(size=(1, 1, 64)).astype(np.float32))
    logits = mx.broadcast_to(row, (1, 5, 64))
    sampler = KeyedSampler(params(), 29)
    actual, _ = tree_target_rows(logits, shape, 7, sampler)
    assert int(actual[1]) == int(actual[2])
    assert int(actual[3]) == int(actual[4])


@pytest.mark.parametrize("seed", range(16))
def test_coupling_only_changes_acceptance(seed):
    sampler = KeyedSampler(params(top_p=0.95, top_k=20), seed)
    logits = mx.array(
        np.random.default_rng(7).normal(size=(1, 7, 32)).astype(np.float32)
    )
    # Every proposal window describes the same serial rows. Coupled proposals agree
    # exactly when draft == target, while uncoupled proposals usually stop early.
    probs = normalize_rows(logits)
    coupled = mx.argmax(
        coupled_scores(probs, sampler, mx.arange(7) + 1), axis=-1
    ).tolist()
    plain = mx.argmax(probs, axis=-1).tolist()
    expected = sampler.sample_positions(probs, list(range(1, 8))).tolist()
    lengths = []
    for proposals in (plain, coupled):
        window = [0, *proposals[:-1]]
        target, _ = tree_target_rows(
            logits, TreeShape([-1, 0, 1, 2, 3, 4, 5]), 1, sampler
        )
        path = walk(window, [-1, 0, 1, 2, 3, 4, 5], target.tolist())
        emitted = [window[i] for i in path[1:]] + [int(target[path[-1]])]
        assert emitted == expected[: len(emitted)]
        lengths.append(len(emitted))
    assert lengths[1] == 7
    assert lengths[1] >= lengths[0]
    scores = coupled_scores(probs, sampler, mx.arange(7) + 1)
    ranked = mx.argsort(-scores, axis=-1).tolist()
    assert [r[0] for r in ranked] == expected
    assert all(len(set(r)) == len(r) for r in ranked)


def test_only_keyed_supported_sampled_requests_are_admitted():
    assert tree_supported(KeyedSampler(params(), 3), False)
    assert not tree_supported(KeyedSampler(params(xtc_probability=0.5), 3), False)
    assert not tree_supported(KeyedSampler(params(temperature=0), 3), False)
    assert not tree_supported(lambda x: x, False)
    assert tree_supported(lambda x: x, True)


@pytest.mark.parametrize("fast", [False, True])
def test_coupled_lattice_ranks_full_vocabulary_before_shortlisting(fast):
    from yunshu_engine.dflash_tree import lattice_from_hidden, search_tree

    if fast:
        from yunshu_engine.dflash_plan import search_tree

    logits = mx.array(
        np.random.default_rng(71).normal(size=(1, 8, 64)).astype(np.float32)
    )
    sel = SimpleNamespace(
        top_k=4,
        hidden_projection=lambda h: mx.ones((1, 7, 2)),
        successor_codebook=lambda ids: mx.ones((*ids.shape, 2)),
        predecessor_codebook=lambda ids: mx.ones((*ids.shape, 2)),
    )

    def reduced(_):
        pytest.fail("sampled candidates must search full vocabulary")

    draft = SimpleNamespace(
        candidate_selector=sel,
        _draft_vocab=SimpleNamespace(logits_subset=reduced),
        config=SimpleNamespace(final_logit_softcapping=None),
        _logits=lambda _: logits[:, :7],
    )
    sampler = KeyedSampler(params(top_p=0.95, top_k=20), 79)
    lat = lattice_from_hidden(
        draft,
        mx.zeros((1, 7, 4)),
        mx.array([0]),
        keyed=sampler,
        committed=mx.array([9]),
    )
    expected = sampler.sample_positions(
        normalize_rows(logits), list(range(9, 17))
    ).tolist()
    winners = (
        mx.take_along_axis(lat.cands, mx.argmax(lat.unary, axis=-1)[:, None], axis=-1)
        .reshape(-1)
        .tolist()
    )
    assert winners == expected[:7]
    tokens, parents = search_tree(lat, 7)
    window, par = [0, *tokens.tolist()], parents.tolist()
    shape = TreeShape(par)
    per_node = logits[:, mx.array(shape.depths)]
    target, _ = tree_target_rows(per_node, shape, 9, sampler)
    path = walk(window, par, target.tolist())
    emitted = [window[i] for i in path[1:]] + [int(target[path[-1]])]
    assert emitted == expected[: len(emitted)]
