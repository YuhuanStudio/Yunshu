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
