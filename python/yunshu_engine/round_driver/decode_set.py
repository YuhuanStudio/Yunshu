"""Decode sets of the round driver (M2): the base class of the batches that hold decoding rows.

A decode set owns the ``rows`` of one step (their order is the batch order) and the per-row state the
set keeps between steps. The shared bookkeeping lives here: admitting a row (``_admit``), dropping rows
(``_drop``), the default ``join`` / ``leave`` / ``cut_out`` and the ``arrays`` loop. A family's own
per-row cache function (``_row_arrays``) and its step (``forward``, ``step``) stay in its class.
Nothing here names a model family; the block library (``blocks/``) builds on the same base.
"""

from __future__ import annotations

from typing import Any


class DecodeSet:
    """Base of the decode sets (``DecodeBatch``, ``FlashNextBatch``, ``GenericBatch``, ``DFlashBatch``)."""

    rows: list

    def _admit(self, row: Any) -> None:
        """Add ``row`` to the decode order; the row's slot is its own id (its caches stay on the row)."""
        row.slot = id(row)
        self.rows.append(row)

    def _drop(self, gone: list) -> None:
        """Remove ``gone`` from the decode order and clear their slots."""
        ids = {id(r) for r in gone}
        for r in gone:
            r.slot = None
        self.rows = [r for r in self.rows if id(r) not in ids]

    def _row_arrays(self, row: Any) -> list:
        """The arrays one row keeps between steps (the family's own cache function)."""
        raise NotImplementedError

    def join(self, rows: list) -> None:
        """Move prefilled rows into the decode order (rows keep their own caches by default)."""
        for row in rows:
            self._admit(row)

    def leave(self, gone: list) -> None:
        """Drop rows that finished or were cancelled."""
        self._drop(gone)

    def cut_out(self, row: Any) -> None:
        """Give ``row`` its own cache back before it leaves. Rows that keep their own caches have nothing
        to give back; a set that holds rows in shared buffers overrides this."""

    def arrays(self) -> list:
        """Everything the set keeps between steps (evaluated each step so no lazy graph carries over)."""
        return [a for r in self.rows for a in self._row_arrays(r)]
