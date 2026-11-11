"""MTP round readback packs variable windows without reading the head's seed table."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine.round_driver import driver


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def test_round_readback_preserves_variable_windows_and_reads_once(monkeypatch):
    calls = []

    def read(a):
        calls.append(a)
        return a.tolist()

    rows = [
        SimpleNamespace(draft_dev=mx.array([7, 8])),
        SimpleNamespace(draft_dev=None),
        SimpleNamespace(draft_dev=mx.array([99, 100, 101])),
    ]
    targets = [mx.array([4, 5, 6]), mx.array([21]), mx.array([31, 32])]
    got, drafts = driver.round_readback(targets, rows, [2, 0, 1], read=read)
    assert got == [[4, 5, 6], [21], [31, 32]]
    assert drafts == [[7, 8], [], [99]]
    assert len(calls) == 1


def test_round_readback_ignores_stale_drafts_when_row_is_forced():
    rows = [SimpleNamespace(draft_dev=mx.array([999]))]
    assert driver.round_readback([mx.array([3])], rows, [0]) == ([[3]], [[]])


def test_decode_transaction_uses_one_packed_read_for_two_rows(monkeypatch):
    from .test_round_driver_generic import TinyLM, make, request, run

    original = driver.round_readback
    calls = []

    def packed(targets, rows, used_drafts):
        reads = []
        result = original(
            targets, rows, used_drafts, read=lambda a: (reads.append(a), a.tolist())[1]
        )
        assert len(reads) == 1
        calls.append(len(rows))
        return result

    monkeypatch.setattr(driver, "round_readback", packed)
    lm = TinyLM()
    out = run(
        make(lm),
        [
            request(lm, [1, 2, 3], n=5, handle="a"),
            request(lm, [4, 5, 6], n=5, handle="b"),
        ],
    )
    assert len(out["a"]) == len(out["b"]) == 5
    assert 2 in calls
