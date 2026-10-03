"""Last-use ordering preserves ancestry and bounds retained recurrent states."""

import importlib.util
import itertools
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gdn_tree_order", Path(__file__).parents[2] / "scripts/research/gdn_tree_order.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def check(parents):
    order, new, need = _module.order_plan(parents)
    assert sorted(order) == list(range(len(parents)))
    remaining = [new.count(row) for row in range(len(new))]
    live = set()
    peak = 0
    for row, parent in enumerate(new):
        if parent >= 0:
            assert parent in live
            remaining[parent] -= 1
            if not remaining[parent]:
                live.remove(parent)
        if remaining[row]:
            live.add(row)
        peak = max(peak, len(live))
        old = order[row]
        path, old_path = [], []
        while row >= 0:
            path.append(order[row])
            row = new[row]
        while old >= 0:
            old_path.append(old)
            old = parents[old]
        assert path == old_path
    assert need == peak
    assert peak <= _module.live_bound(len(parents))
    return peak


def test_all_parent_before_child_trees_through_eight_rows():
    for width in range(1, 9):
        for choices in itertools.product(*(range(row) for row in range(1, width))):
            check([-1, *choices])


def test_balanced_and_comb_adversaries():
    assert check([-1, *[(row - 1) // 2 for row in range(1, 31)]]) == 4
    assert check([-1, *range(31)]) == 1
    assert check([-1, *[0] * 31]) == 1
    assert _module.live_bound(16) == 3
    assert _module.live_bound(32) == 4


def test_budget_observations_keep_original_proposal_rank():
    from yunshu_engine.spec_schedule import NodeBudget

    parents = [-1, *[(row - 1) // 2 for row in range(1, 16)]]
    order, _, _ = _module.order_plan(parents)
    path = [15]
    while parents[path[-1]] >= 0:
        path.append(parents[path[-1]])
    path.reverse()
    original = [row - 1 for row in path[1:]]
    reordered = [order.index(row) - 1 for row in path[1:]]
    assert reordered != original
    a, b = NodeBudget(15), NodeBudget(15)
    for _ in range(20):
        a.observe(15, original, 50)
        b.observe(15, _module.remap_landed(reordered, order), 50)
    assert a.p == b.p
    assert a.samples == b.samples
    assert a.best() == b.best()
