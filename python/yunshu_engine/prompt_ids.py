"""Per-request prompt-id helpers that cost O(n) in numpy, not in Python.

The cache-hit path walked every prompt id in Python several times a request (boundary scans,
hashing, the draft-vocabulary context): ~1.4-2 us/token, 0.2 s at 131K ids. The functions here
give the same results from one int32 array built once per prompt object.
"""

from __future__ import annotations

from typing import cast

import numpy as np

_SLOT: list = [None, None]  # [the id sequence object, its int32 array]


def ids_array(token_ids, remember: bool = True) -> np.ndarray | None:
    """``token_ids`` (list / tuple / ndarray) as a 1-D int32 array, or None when it is not
    one (mx arrays, generators, out-of-range values). One slot: the same list object (a
    request's ``job.ids``, never mutated) converts once."""
    if isinstance(token_ids, np.ndarray):
        return (
            token_ids if token_ids.dtype == np.int32 and token_ids.ndim == 1 else None
        )
    if not isinstance(token_ids, (list, tuple)):
        return None
    if _SLOT[0] is token_ids and len(_SLOT[1]) == len(token_ids):
        arr = _SLOT[1]
        # a request's ids are never mutated; the ends are checked as a cheap tripwire
        if not len(arr) or (arr[0] == token_ids[0] and arr[-1] == token_ids[-1]):
            return cast(np.ndarray, arr)
    try:
        arr = np.asarray(token_ids, dtype=np.int32)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.ndim != 1:
        return None
    if remember:
        _SLOT[0], _SLOT[1] = token_ids, arr
    return arr


def head_boundary_index(arr: np.ndarray, a: int, b: int) -> int:
    """First i in [1, n-2] with ids[i] == a and ids[i+1] == b; 0 when ids start with (a, b)
    or there is none (the unaligned index; the caller aligns it)."""
    n = len(arr)
    if n >= 2 and arr[0] == a and arr[1] == b:
        return 0
    if n < 3:
        return 0
    hit = np.flatnonzero((arr[1 : n - 1] == a) & (arr[2:n] == b))
    return int(hit[0]) + 1 if len(hit) else 0


def semantic_boundary_set(
    arr: np.ndarray,
    header: tuple[int, int] | None,
    message_end: int | None,
    turn_boundary: bool,
    boundary_tokens,
) -> set[int]:
    """The unaligned boundary set of ``ApcManager.semantic_boundaries``, found by visiting only
    the header / message-end positions (the state machine changes there and one token after a
    header) and intersecting the assistant spans with the boundary-token positions."""
    n = len(arr)
    bt = np.fromiter(boundary_tokens, dtype=np.int64) if boundary_tokens else None
    out: set[int] = set()
    if bt is None or not len(bt):
        bt_pos = np.empty(0, dtype=np.int64)
    else:
        bt_pos = np.flatnonzero(np.isin(arr, bt))
    if header is None:
        return {int(i) + 1 for i in bt_pos}
    h0, h1 = int(header[0]), int(header[1])
    special = arr == h0
    if message_end is not None:
        special = special | (arr == message_end)
    events = np.flatnonzero(special)
    pos = set(events.tolist())
    pos.update(int(e) + 1 for e in np.flatnonzero(arr == h0) if e + 1 < n)
    assistant = False
    pending = False
    start = None
    spans: list[tuple[int, int]] = []
    for p in sorted(pos):
        token = int(arr[p])
        was = assistant
        if token == h0:
            if turn_boundary and pending:
                out.add(p)
            pending = False
            assistant = False
        elif message_end is not None and token == message_end:
            if turn_boundary:
                pending = was
            assistant = False
        elif p > 0 and int(arr[p - 1]) == h0:
            assistant = token == h1
        if assistant and not was:
            start = p
        elif was and not assistant:
            assert start is not None  # a span opened where the turn became assistant
            spans.append((start, p))
            start = None
    if assistant and start is not None:
        spans.append((start, n))
    for s, e in spans:
        lo, hi = np.searchsorted(bt_pos, [s, e])
        out.update(int(i) + 1 for i in bt_pos[lo:hi])
    return out
