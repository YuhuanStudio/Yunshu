"""The fused noise kernel must reproduce the op-by-op graph bit for bit."""

import mlx.core as mx
import numpy as np

from yunshu_engine import keyed_sampling as ks


def _bases(seeds):
    return mx.array(np.array([ks.seed_base(s) for s in seeds], dtype=np.uint64))


def test_noise_kernel_is_bit_identical_to_the_graph():
    pos = mx.array([0, 1, 7, 12345, 2**31 + 5, 3], dtype=mx.uint32)
    bases = _bases([1, 2, 3, 2**63 + 9, 17, 17])
    vocab = 248320  # not a multiple of the 256-wide threadgroup
    a = ks.gumbel_rows(bases, pos, vocab)
    b = ks._gumbel_rows_graph(bases, pos, vocab)
    assert a.shape == b.shape == (6, vocab)
    assert np.array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))


def test_noised_rows_matches_where_add():
    rng = np.random.default_rng(3)
    vocab = 5003
    row = rng.normal(size=(4, vocab)).astype(np.float32) * 5
    row[:, ::7] = -np.inf
    row = mx.array(row)
    pos = mx.array([5, 6, 7, 8], dtype=mx.int32)
    bases = _bases([9, 9, 10, 11])
    got = ks.noised_rows(row, bases, pos)
    noise = ks._gumbel_rows_graph(bases, pos, vocab)
    want = mx.where(row == -mx.inf, -mx.inf, row + noise)
    assert np.array_equal(np.array(got).view(np.uint32), np.array(want).view(np.uint32))
    assert np.array_equal(
        np.array(mx.argmax(got, axis=-1)), np.array(mx.argmax(want, axis=-1))
    )


def _bits(a):
    return np.array(a).view(np.uint32)


def test_noised_at_matches_graph():
    rng = np.random.default_rng(5)
    vals = rng.normal(size=(3, 1024)).astype(np.float32)
    vals[:, ::5] = -np.inf
    ids = rng.integers(0, 248320, size=(3, 1024)).astype(np.int32)
    vals, ids = mx.array(vals), mx.array(ids)
    pos = mx.array([4, 99, 2**31 + 1], dtype=mx.uint32)
    bases = _bases([1, 2, 2**63 + 1])
    want = mx.where(
        vals == -mx.inf,
        -mx.inf,
        vals + ks._gumbel_graph_at(bases, pos, ids.astype(mx.uint64)),
    )
    assert np.array_equal(_bits(ks.noised_at(vals, ids, bases, pos)), _bits(want))


def test_driver_draw_paths_equal_the_graph(monkeypatch):
    """sample_rows (dense and candidate path), coupled_scores and coupled_candidates give the
    same tokens / scores with the fused kernels as with the op graph."""
    from types import SimpleNamespace

    vocab = 248320
    rng = np.random.default_rng(11)
    x = mx.array(rng.normal(scale=4.0, size=(5, vocab)).astype(np.float32))
    lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
    pos = mx.array([0, 3, 9, 100, 101], dtype=mx.int32)
    bases = _bases([7, 7, 8, 9, 9])
    cases = [
        SimpleNamespace(top_k=0, top_p=1.0, min_p=0.0, temperature=1.0),  # dense
        SimpleNamespace(top_k=20, top_p=0.95, min_p=0.0, temperature=0.7),  # candidates
    ]
    sampler = SimpleNamespace(seed=42, position_base=3, params=cases[1])

    def run():
        out = [np.array(ks.sample_rows(lp, p, bases, pos)) for p in cases]
        out.append(_bits(ks.coupled_scores(lp, sampler, pos)))
        cand = ks.coupled_candidates(lp, sampler, pos)
        out.append(None if cand is None else (np.array(cand[0]), _bits(cand[1])))
        return out

    fused = run()
    monkeypatch.setattr(ks, "_fused_ok", lambda: False)
    graph = run()
    for a, b in zip(fused, graph, strict=True):
        if isinstance(a, tuple):
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        else:
            assert np.array_equal(a, b)
