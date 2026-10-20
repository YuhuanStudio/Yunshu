import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))


def test_random_cases_have_valid_parents_and_reference_grows_by_rows():
    from graft_kernel_check import random_case

    from yunshu_engine.dflash_plan import graft_reference

    rng = random.Random(3)
    for _ in range(200):
        tokens, parents, copied, rows = random_case(rng, rng.randrange(1, 16))
        t, p = graft_reference(tokens, parents, copied, rows)
        assert len(t) == len(tokens) + rows and len(p) == len(t) + 1
        assert all(0 <= p[r] < r for r in range(1, len(p)))
