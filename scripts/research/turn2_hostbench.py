"""CPU microbenchmark: host work on the cache-hit path, old per-token loops vs the numpy
helpers (semantic_boundaries, head_boundary, key hashing, draft-vocab prompt scan).

    PYTHONPATH=python python scripts/research/turn2_hostbench.py
"""

from __future__ import annotations

import hashlib
import random
import time

import numpy as np

from yunshu_engine import apc_manager
from yunshu_engine.prompt_ids import ids_array

H0, H1, END = 151644, 77091, 151645
BT = frozenset({198, 271, 13, 382})


def make(n):
    rng = random.Random(1)
    out = [rng.randrange(1000, 150000) for _ in range(n)]
    for i in range(0, n, 700):
        for j, t in enumerate([H0, 872, 198, END, H0, H1, 198, 271, END]):
            if i + j < n:
                out[i + j] = t
    return out


def best(f, reps=5):
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        f()
        t.append(time.perf_counter() - s)
    return min(t) * 1e3


def main():
    cls = next(
        v
        for v in vars(apc_manager).values()
        if isinstance(v, type) and hasattr(v, "_semantic_boundaries_loop")
    )
    m = object.__new__(cls)
    m.prefill_assistant_header, m.prefill_message_end = (H0, H1), END
    m.prefill_turn_boundary, m.prefill_boundary_tokens = True, BT
    m.prefill_alignment, m.head_marker = 64, (H0, 872)
    print(
        "ids      sem old/new   head old/new   hash(key_for) old/new   vocab scan old/new   [ms]"
    )
    for n in (8192, 32768, 131611):
        ids = make(n)
        ids_array(ids)  # one conversion per prompt, counted below
        conv = best(lambda: np.asarray(ids, dtype=np.int32))
        sem = (
            best(lambda: m._semantic_boundaries_loop(ids)),
            best(lambda: m.semantic_boundaries(ids)),
        )
        head = (
            best(lambda: m._head_boundary_loop(ids)),
            best(lambda: m.head_boundary(ids)),
        )
        old_key = best(
            lambda: hashlib.blake2b(
                np.asarray(list(ids[: n - 100]), dtype=np.int64).tobytes(),
                digest_size=16,
            ).digest()
        )
        arr = ids_array(ids)
        new_key = best(
            lambda: hashlib.blake2b(
                arr[: n - 100].astype(np.int64).tobytes(), digest_size=16
            ).digest()
        )
        py = best(
            lambda: (
                sum(1 for t in [int(t) for t in ids] if 1000 <= t < 50000),
                sorted(t for t in set(ids) if 1000 <= t < 50000),
            )
        )
        npv = best(
            lambda: (
                int(np.count_nonzero((arr >= 1000) & (arr < 50000))),
                np.unique(arr[(arr >= 1000) & (arr < 50000)]).tolist(),
            )
        )
        print(
            f"{n:<8} {sem[0]:7.1f}/{sem[1]:<6.1f} {head[0]:7.1f}/{head[1]:<6.1f} {old_key:8.1f}/{new_key:<8.1f} {py:8.1f}/{npv:<8.1f}  (one-time int32 conversion {conv:.1f})"
        )


if __name__ == "__main__":
    main()
