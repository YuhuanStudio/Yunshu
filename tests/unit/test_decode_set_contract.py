"""Contract every round-driver decode set runs (M2): the base class ``DecodeSet`` and the shared methods.

The four decode sets (``DecodeBatch``, ``FlashNextBatch``, ``GenericBatch``, ``DFlashBatch``) derive from
``DecodeSet`` and expose ``join`` / ``leave`` / ``arrays`` / ``cut_out`` and a ``rows`` list. ``DFlashBatch``
decodes through ``step(driver)`` instead of ``forward`` (the drafter owns the step). The own-cache sets
(``FlashNextBatch``, ``GenericBatch``) keep their rows' caches on the row: their join / leave / cut_out /
arrays are the base's and are checked here on fake rows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine.round_driver import flashnext, generic  # noqa: E402
from yunshu_engine.round_driver.batch import DecodeBatch  # noqa: E402
from yunshu_engine.round_driver.decode_set import DecodeSet  # noqa: E402
from yunshu_engine.round_driver.dflash import DFlashBatch  # noqa: E402

DECODE_SETS = (DecodeBatch, flashnext.FlashNextBatch, generic.GenericBatch, DFlashBatch)


@pytest.mark.parametrize("cls", DECODE_SETS, ids=lambda c: c.__name__)
def test_every_decode_set_runs_the_base_contract(cls):
    assert issubclass(cls, DecodeSet)
    for name in ("join", "leave", "arrays", "cut_out"):
        assert callable(getattr(cls, name)), (cls.__name__, name)
    # the decode entry: ``forward`` for the layer sets, ``step`` for the drafter-owned DFlash set
    entry = "step" if cls is DFlashBatch else "forward"
    assert callable(getattr(cls, entry)), (cls.__name__, entry)


@pytest.mark.parametrize(
    "cls", [flashnext.FlashNextBatch, generic.GenericBatch], ids=lambda c: c.__name__
)
def test_own_cache_sets_share_join_leave_and_keep_order(cls):
    s = cls.__new__(cls)
    s.rows = []
    rows = [SimpleNamespace(cache=[], slot=None, n=i) for i in range(3)]
    s.join(rows)
    assert s.rows == rows and all(r.slot == id(r) for r in rows)
    s.leave([rows[1]])
    assert s.rows == [rows[0], rows[2]]
    assert rows[1].slot is None and rows[0].slot == id(rows[0])


@pytest.mark.parametrize(
    "cls", [flashnext.FlashNextBatch, generic.GenericBatch], ids=lambda c: c.__name__
)
def test_own_cache_sets_cut_out_is_a_no_op(cls):
    s = cls.__new__(cls)
    s.rows = []
    row = SimpleNamespace(cache=[], slot=None, n=2)
    s.join([row])
    cache_before = row.cache
    s.cut_out(row)
    assert row.cache is cache_before and s.rows == [row]


def test_arrays_is_the_loop_over_the_family_hook():
    class Probe(DecodeSet):
        def __init__(self):
            self.rows = [SimpleNamespace(tag=1), SimpleNamespace(tag=2)]

        def _row_arrays(self, row):
            return [row.tag, row.tag * 10]

    assert Probe().arrays() == [1, 10, 2, 20]


def test_base_join_and_leave_and_hook_must_be_provided():
    class Bare(DecodeSet):
        def __init__(self):
            self.rows = []

    b = Bare()
    row = SimpleNamespace(slot=None)
    b.join([row])
    assert b.rows == [row] and row.slot == id(row)
    b.leave([row])
    assert b.rows == [] and row.slot is None
    with pytest.raises(NotImplementedError):
        Bare()._row_arrays(row)
