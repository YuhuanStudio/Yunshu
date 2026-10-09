from types import SimpleNamespace

import pytest

from yunshu_engine import dflash_apc_window as w


@pytest.fixture(autouse=True)
def _fresh():
    w.clear()
    w.set_budget(1000)
    w._STATE["current"] = None
    yield
    w.clear()
    w._STATE["current"] = None


def _batch(prefix, ids, chunks=None, kwargs=True):
    sp = SimpleNamespace(
        kwargs={"x": 1} if kwargs else {}, chunks=chunks or [], _yunshu_keep=4
    )
    return SimpleNamespace(
        uids=[1],
        _speculative_prefill=sp,
        _apc_meta=[{"prefix_len": prefix, "full_input_ids": ids, "extra_hash": 7}],
    )


def test_hit_seeds_stored_window_and_miss_does_not():
    w.put(w.key_for(7, [1, 2, 3]), [SimpleNamespace(shape=(1, 4, 8)), "L1"], 100)
    hit = _batch(3, [1, 2, 3, 4, 5])
    assert w.seed(hit) is True and hit._speculative_prefill.chunks[0][1] == "L1"
    other = _batch(3, [9, 2, 3, 4, 5])
    assert w.seed(other) is False and other._speculative_prefill.chunks == []
    cold = _batch(0, [1, 2, 3])
    assert w.seed(cold) is False


def test_key_binds_namespace_and_length():
    assert w.key_for(1, [1, 2]) != w.key_for(2, [1, 2])
    assert w.key_for(1, [1, 2]) != w.key_for(1, [1, 2, 3])


def test_budget_evicts_oldest_and_counts_bytes():
    for i in range(5):
        w.put(w.key_for(0, [i]), [i], 400)
    assert w.resident_bytes() <= 1000 and len(w._STORE) == 2
    assert w.get(w.key_for(0, [0])) is None and w.get(w.key_for(0, [4])) == [4]
    assert w.put(w.key_for(0, [99]), [0], 5000) is False


def test_capture_stores_window_at_checkpoint(monkeypatch):
    monkeypatch.setattr(
        w, "window_from_chunks", lambda chunks, keep: ([f"k{keep}"], 10)
    )
    assert w.capture([1, 2], 7) is False  # no prefill step running
    w._STATE["current"] = _batch(0, [1, 2], chunks=[["a"]])
    assert w.capture([1, 2], 7) is True
    assert w.get(w.key_for(7, [1, 2])) == ["k4"]
    w._STATE["current"] = _batch(0, [1, 2], chunks=[["a"]], kwargs=False)
    assert w.capture([1, 3], 7) is False


def test_window_capture_never_blocks_the_prefill_step(monkeypatch):
    import sys
    import types

    calls = []

    class A:
        def __init__(self, n):
            self.shape, self.nbytes = (1, n, 4), n * 8

        def __getitem__(self, _):
            return A(2)

    stub = types.ModuleType("mlx.core")
    stub.concatenate = lambda parts, axis: A(sum(p.shape[1] for p in parts))
    stub.async_eval = lambda x: calls.append("async")
    stub.eval = lambda x: calls.append("eval")
    stub.contiguous = lambda x: calls.append("contiguous") or x
    monkeypatch.setitem(sys.modules, "mlx.core", stub)
    layers, nbytes = w.window_from_chunks([[A(3)], [A(4)]], keep=2)
    assert calls == ["async"] and nbytes == 16
