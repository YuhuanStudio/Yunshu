"""The numpy prompt-id helpers return exactly what the per-token Python loops returned."""

from __future__ import annotations

import hashlib
import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest

from yunshu_engine import apc_manager, dflash_apc_window, hit_stages
from yunshu_engine.apc_manager import APCManager  # noqa: F401  (import check)
from yunshu_engine.prompt_ids import (
    head_boundary_index,
    ids_array,
)

H0, H1, END = 151644, 77091, 151645
BT = frozenset({198, 271, 13, 382})


def _ids(n: int, seed: int, dense: bool) -> list[int]:
    rng = random.Random(seed)
    pool = [H0, H1, END, 872, 198, 271, 13, 382]
    if dense:
        return [rng.choice(pool + [rng.randrange(1000, 150000)]) for _ in range(n)]
    out = [rng.randrange(1000, 150000) for _ in range(n)]
    for i in range(0, n, 700):  # chat-like: user turn, assistant turn with newlines
        for j, t in enumerate([H0, 872, 198, END, H0, H1, 198, 271, END]):
            if i + j < n:
                out[i + j] = t
    for i in range(0, n, 37):
        out[i] = rng.choice([198, 271, 13])
    return out


def _mgr(header, end, turn, alignment=64):
    m = object.__new__(
        apc_manager.ApcManager if hasattr(apc_manager, "ApcManager") else _find()
    )
    m.prefill_assistant_header = header
    m.prefill_message_end = end
    m.prefill_turn_boundary = turn
    m.prefill_boundary_tokens = BT
    m.prefill_alignment = alignment
    m.head_marker = (H0, 872)
    return m


def _find():
    for v in vars(apc_manager).values():
        if isinstance(v, type) and hasattr(v, "_semantic_boundaries_loop"):
            return v
    raise AssertionError("no manager class")


@pytest.mark.parametrize("n", [0, 1, 2, 5, 64, 1000, 8192, 131611])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "header,end,turn",
    [
        ((H0, H1), END, True),
        ((H0, H1), END, False),
        ((H0, H1), None, True),
        (None, END, True),
    ],
)
def test_semantic_boundaries_match_loop(n, dense, header, end, turn):
    if dense and n > 8192:
        n = 20000
    m = _mgr(header, end, turn)
    ids = _ids(n, n + 7, dense)
    assert m.semantic_boundaries(ids) == m._semantic_boundaries_loop(ids)
    assert m.semantic_boundaries(tuple(ids)) == m._semantic_boundaries_loop(ids)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 100, 131611])
@pytest.mark.parametrize("dense", [False, True])
def test_head_boundary_matches_loop(n, dense):
    m = _mgr((H0, H1), END, True)
    ids = _ids(n, 3 + n, dense)
    assert m.head_boundary(ids) == m._head_boundary_loop(ids)
    if n > 3:
        ids = list(ids)
        ids[0], ids[1] = H0, 872  # starts with the marker: 0
        assert m.head_boundary(ids) == m._head_boundary_loop(ids) == 0


def test_head_boundary_index_unaligned():
    arr = np.array([5, 6, 151644, 872, 1], dtype=np.int32)
    assert head_boundary_index(arr, H0, 872) == 2


def test_sequence_hash_view_equals_tuple():
    ids = _ids(131611, 5, False)
    arr = ids_array(ids)
    for n in (1, 64, 960, 131072, 131611):
        assert apc_manager._sequence_hash(arr[:n], 7, 16) == apc_manager._sequence_hash(
            tuple(ids[:n]), 7, 16
        )


def test_key_for_view_equals_list():
    ids = _ids(33305, 9, False)
    arr = ids_array(ids)
    old = (
        3,
        32704,
        hashlib.blake2b(
            np.asarray(list(ids[:32704]), dtype=np.int64).tobytes(), digest_size=16
        ).digest(),
    )
    assert dflash_apc_window.key_for(3, arr[:32704]) == old
    assert (
        dflash_apc_window.key_for(3, ids[:32704]) == old
    )  # list input, slot untouched
    assert ids_array(ids) is arr


def test_ids_array_slot_and_rejects():
    a = [1, 2, 3]
    assert ids_array(a) is ids_array(a)
    assert ids_array([2**40]) is None
    assert ids_array(iter([1])) is None


def test_mostly_high_matches_python():
    from yunshu_engine.draft_vocab import DraftVocab

    ns = SimpleNamespace(keep=1000, vocab=50000)
    ids = _ids(30000, 11, False)
    py = sum(1 for t in ids if 1000 <= t < 50000) * 2 >= len(ids)
    assert DraftVocab._mostly_high(ns, ids_array(ids)) == py
    assert DraftVocab._mostly_high(ns, ids) == py


def test_stage_line(caplog):
    hit_stages.begin()
    hit_stages.add("standing", 0.002)
    with hit_stages.stage("hash"):
        pass
    hit_stages.add("add", 0.01)
    hit_stages.mark("add_end")
    with caplog.at_level(logging.INFO, logger="yunshu_engine.hit_stages"):
        hit_stages.finish(100, 74, 0.0001)
    (rec,) = caplog.records
    assert rec.getMessage().startswith("turn stages: ids=100 hit=74 | standing=2.0")
    assert not hit_stages.active()
    hit_stages.finish(1, 1, 0.0)  # no record: silent
    assert len(caplog.records) == 1
