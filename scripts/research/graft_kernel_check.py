"""Device suffix-graft kernel against its host reference on random trees."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def random_case(rng, n, vocab=5):
    parents = [-1]
    for r in range(1, n + 1):
        parents.append(rng.randrange(0, r))
    tokens = [rng.randrange(vocab) for _ in range(n)]
    copied = [rng.randrange(vocab) for _ in range(rng.randrange(0, 20))]
    return tokens, parents, copied, rng.randrange(1, 12)


def main() -> int:
    import mlx.core as mx

    from yunshu_engine.dflash_plan import graft_copy, graft_reference

    rng = random.Random(7)
    bad = 0
    for _ in range(300):
        tokens, parents, copied, rows = random_case(rng, rng.randrange(1, 16))
        if not copied:
            continue
        t, p = graft_copy(
            mx.array(tokens, mx.int32), mx.array(parents, mx.int32), copied, rows
        )
        want_t, want_p = graft_reference(tokens, parents, copied, rows)
        if t.tolist() != want_t or p.tolist() != want_p:
            bad += 1
            print("MISMATCH", tokens, parents, copied, rows, t.tolist(), p.tolist())
    print("graft kernel check: mismatches", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
