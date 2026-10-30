"""The fast tree applies the generated-token penalty per row exactly as the serial window_row does."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import keyed_sampling as ks
from yunshu_engine.constrained_spec import SpecRequest, normalize_rows, tree_penalty
from yunshu_engine.keyed_sampling import KeyedSampler, tree_target_rows
from yunshu_engine.text_logit_processors import GeneratedTokenPenalty
from yunshu_engine.tree_verify import TreeShape
from yunshu_engine.vlm_batch_runner import RowParams, TokenMaskProcessor

V = 300
PARENTS = [-1, 0, 0, 1, 1, 2, 3, 3, 6]


def _setup(dtype, fp=0.4, pp=1.5, seed=3):
    rng = np.random.default_rng(seed)
    logits = mx.array(
        rng.normal(scale=3.0, size=(1, len(PARENTS), V)).astype(np.float32)
    ).astype(dtype)
    # repeated ids on one path exercise the count (frequency) term
    window = mx.array([[7, 11, 5, 11, 7, 5, 11, 11, 20]], dtype=mx.int32)
    penalty = GeneratedTokenPenalty(fp, pp)
    penalty.window_row([], logits[:, 0, :])  # materialize the counts vector
    penalty.commit_tokens([5, 7, 7, 40, 41])
    return logits, window, penalty


def _serial(logits, window, penalty, shape):
    rows = []
    toks = [int(t) for t in window.reshape(-1).tolist()]
    for r, path in enumerate(shape.paths):
        rows.append(penalty.window_row([toks[i] for i in path], logits[:, r, :]))
    return mx.stack(rows, axis=1)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("fp,pp", [(0.0, 1.5), (0.4, 0.0), (0.4, 1.5)])
def test_greedy_rows_equal_serial_window_rows(dtype, fp, pp):
    logits, window, penalty = _setup(dtype, fp, pp)
    shape = TreeShape(PARENTS)
    target, probs = tree_target_rows(logits, shape, 10, None, None, penalty, window)
    want = _serial(logits, window, penalty, shape)
    want_probs = normalize_rows(want)
    assert mx.array_equal(probs, want_probs).item()
    assert mx.array_equal(target, mx.argmax(want_probs, axis=-1)).item()


def test_keyed_rows_equal_serial_window_rows():
    logits, window, penalty = _setup(mx.bfloat16)
    shape = TreeShape(PARENTS)
    params = RowParams(temperature=0.7, top_p=0.95, top_k=0, min_p=0.0)
    sampler = KeyedSampler(params, 1234)
    target, _ = tree_target_rows(logits, shape, 10, sampler, None, penalty, window)
    want = normalize_rows(_serial(logits, window, penalty, shape))
    ref = sampler.sample_positions(want, shape.depth_array() + 10)
    assert mx.array_equal(target, ref).item()


def test_penalty_changes_the_distribution_and_counts_are_not_mutated():
    logits, window, penalty = _setup(mx.float32)
    before = penalty._cnt
    tree_target_rows(logits, TreeShape(PARENTS), 0, None, None, penalty, window)
    assert mx.array_equal(penalty._cnt, before).item()
    _, plain = tree_target_rows(logits, TreeShape(PARENTS), 0, None, None)
    _, pen = tree_target_rows(
        logits, TreeShape(PARENTS), 0, None, None, penalty, window
    )
    assert not mx.array_equal(plain, pen).item()


def test_ancestors_only_not_siblings():
    # row 2 (token 5) is a sibling of row 1 (token 11): row 1 must not count row 2's token
    logits, window, penalty = _setup(mx.float32, fp=0.0, pp=2.0)
    shape = TreeShape(PARENTS)
    counts = ks.tree_penalty_counts(penalty, shape, window, V)
    toks = [int(t) for t in window.reshape(-1).tolist()]
    base = np.array(penalty._cnt)
    for r, path in enumerate(shape.paths):
        want = base.copy()
        for i in path:
            want[toks[i]] += 1
        assert np.array_equal(np.array(counts[r]), want), r


def test_tree_penalty_gate():
    pen = GeneratedTokenPenalty(0.0, 1.5)

    def request(procs, logprobs=False, guide=None):
        r = SpecRequest(logprobs)
        r.processors = procs
        r.guide = guide
        return r

    assert tree_penalty(request([pen])) is pen
    assert tree_penalty(request([])) is None
    assert tree_penalty(None) is None
    assert tree_penalty(request([pen], logprobs=True)) is None
    assert tree_penalty(request([pen], guide=object())) is None
    mask = TokenMaskProcessor(suppress=[1], eos_ids=[2], min_tokens=0, ignore_eos=False)
    assert tree_penalty(request([pen, mask])) is None
    assert tree_penalty(request([mask])) is None
