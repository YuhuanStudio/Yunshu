from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from scripts.research import probe_wide_verify as probe  # noqa: E402


def test_single_node_reference_decodes_without_mutating_the_tree_prefix(monkeypatch):
    prefix = [[7]]

    def decode(tokens, *, cache, return_hidden):
        assert cache is not prefix and return_hidden
        cache[0][0] = 99
        return SimpleNamespace(hidden_states=["plain root"])

    def wrong_oracle(*args):
        pytest.fail("single-row speculative verify is not canonical decode")

    monkeypatch.setattr(probe, "chain_verify", wrong_oracle)
    hidden, abort = probe.path_reference(decode, SimpleNamespace(shape=(1, 1)), prefix)
    assert hidden == "plain root"
    abort()
    assert prefix == [[7]]


def test_longer_paths_keep_the_transactional_chain_oracle(monkeypatch):
    calls = []
    result = SimpleNamespace(hidden="chain", abort=lambda: calls.append("aborted"))
    monkeypatch.setattr(probe, "chain_verify", lambda *args: result)
    hidden, abort = probe.path_reference(None, SimpleNamespace(shape=(1, 2)), [])
    assert hidden == "chain"
    abort()
    assert calls == ["aborted"]


@pytest.mark.parametrize("kind", ["chain", "heap", "spine"])
def test_fast_invariance_uses_the_runtime_shape(kind):
    from yunshu_engine import tree_verify as tv

    shape = probe.make_shape(tv, kind, 8, fast=True)
    assert shape.fast_glue
    assert shape.width == 8
    assert shape.is_chain == (kind == "chain")


def test_invariance_dispatches_fast_shapes(monkeypatch):
    import mlx.core as mx

    from yunshu_engine import tree_verify as tv

    shapes = []
    monkeypatch.setattr(probe, "prefill", lambda *_: [])

    def chain(lm, tokens, cache):
        width = tokens.shape[1]
        return SimpleNamespace(
            hidden=mx.zeros((1, width, 2)),
            target_tokens=mx.zeros((1, width), dtype=mx.int32),
            abort=lambda: None,
        )

    def tree(lm, tokens, shape, cache):
        shapes.append(shape)
        return SimpleNamespace(hidden=mx.zeros((1, tokens.shape[1], 2)))

    monkeypatch.setattr(probe, "chain_verify", chain)
    monkeypatch.setattr(tv, "tree_forward", tree)
    monkeypatch.setattr(tv, "tree_abort", lambda *_: None)

    def model(tokens, **kwargs):
        return SimpleNamespace(
            hidden_states=[mx.zeros((1, 1, 2))], logits=mx.zeros((1, 1, 3))
        )

    args = SimpleNamespace(rows=[8], shapes=["chain", "heap"], fast=True, context=40)
    assert probe.invariance(args, model, list(range(100)))
    # the fast_chain check (FastShape chain against one-row decodes), then the chain and heap trees
    assert len(shapes) == 3
    assert shapes[0].is_chain
    assert all(getattr(shape, "fast_glue", False) for shape in shapes)


def test_fast_probe_reorders_branched_rows_with_their_tokens():
    from yunshu_engine import tree_verify as tv
    from yunshu_engine.dflash_plan import live_bound

    shape = probe.make_shape(tv, "heap", 16, fast=True)
    permutation = shape.original_ranks.tolist()
    assert sorted(permutation) == list(range(16))
    assert permutation != list(range(16))
    original = probe.shape_parents("heap", 16)
    parents = shape.parents_array().tolist()
    for row in range(1, 16):
        assert original[permutation[row]] == permutation[parents[row]]
    reads, writes = shape.state_slots()
    assert max(reads.tolist() + writes.tolist()) < live_bound(16)
