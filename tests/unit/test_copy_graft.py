from yunshu_engine.copy_graft import GraftBudget, graft_suffix


def test_graft_shares_prefix_without_repeating_it():
    assert graft_suffix(3, [10, 11, 12, 13, 14, 15], 2) == [13, 14]
    assert graft_suffix(8, [1, 2], 4) == []
    assert graft_suffix(1, [1, 2], -1) == []


def test_mismatched_prefix_cannot_commit_copied_descendants():
    drafts = [10, 99, 12] + graft_suffix(3, [10, 11, 12, 13, 14], 2)
    target = [10, 11, 12, 13, 14]
    accepted = next(
        i for i, (a, b) in enumerate(zip(drafts, target, strict=True)) if a != b
    )
    assert drafts[:accepted] + [target[accepted]] == [10, 11]


def test_growth_requires_full_tail_acceptance():
    budget = GraftBudget()
    assert budget.choose(20) == 4
    budget.observe(5, 4, 3, 60)
    assert budget.width == 4
    budget.observe(5, 4, 9, 60)
    assert budget.width == 8
    assert budget.choose(2) <= 2


def test_unprofitable_tail_backs_off_and_reprobes():
    budget = GraftBudget()
    budget.observe(5, 0, 5, 50)
    budget.observe(5, 4, 5, 80)
    assert budget.choose(20) == 0
    assert any(budget.choose(20) for _ in range(16))


def test_unusable_clock_does_not_train_rate():
    budget = GraftBudget()
    budget.observe(5, 4, 6, None)
    budget.observe(5, 0, 3, None)
    assert not budget.samples and budget.base_ms is None


def test_tree_graft_keeps_copy_branch_when_mtp_diverges(monkeypatch):
    import types

    import mlx.core as mx

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.copy_graft import verify_graft

    seen = {}

    def forward(lm, window, shape, cache):
        seen["parents"] = shape.parents_array().tolist()
        seen["tokens"] = window.reshape(-1).tolist()
        return types.SimpleNamespace(hidden=window[..., None])

    def commit(lm, cache, result, path):
        seen["path"] = path

    monkeypatch.setattr(tv, "tree_forward", forward)
    monkeypatch.setattr(tv, "tree_commit", commit)
    lm = types.SimpleNamespace(
        speculative_argmax_from_hidden=lambda h: mx.where(
            h[..., 0] < 50, h[..., 0] + 10, 60
        )
    )
    result, drafts, rows = verify_graft(
        lm, [], 10, mx.array([[20, 99, 88]]), [20, 30, 40, 50], mx.int32
    )
    assert seen["parents"] == [-1, 0, 1, 2, 0, 1, 5, 6]
    assert rows == 5
    assert drafts.tolist() == [[20, 30, 40, 50]]
    assert result.target_tokens.tolist() == [20, 30, 40, 50, 60]
    result.commit(lm, [], 4, rows)
    assert seen["path"] == [0, 1, 5, 6, 7]
    result.commit(lm, [], 1, rows)
    assert seen["path"] == [0, 1]


def test_grafted_mtp_rounds_match_plain_with_wrong_model_drafts(monkeypatch):
    import types

    import mlx.core as mx
    import mlx_vlm.speculative.mtp as mtp

    from tests.unit.test_mtp_lane_tool_guide import ToyTarget, run_lane
    from yunshu_engine import mtp_lane
    from yunshu_engine import tree_verify as tv

    script = list(range(10, 60)) + [100]
    target = ToyTarget(script, 128, 100)
    monkeypatch.setattr(tv, "supported", lambda lm: True)
    monkeypatch.setattr(tv, "lane_ready", lambda lm, cache: True)
    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda lm, **__: 32)
    monkeypatch.setitem(mtp_lane._STATE, "context", list(range(10, 60)) * 2)

    def tree_forward(lm, window, shape, cache):
        fake = mtp._mtp_verify_target(lm, window, cache, None)
        depths = shape.depth_array()
        lm.speculative_argmax_from_hidden = lambda h: mx.argmax(
            lm.speculative_logits_from_hidden(h), axis=-1
        )
        return types.SimpleNamespace(hidden=fake.hidden[:, depths, :], fake=fake)

    monkeypatch.setattr(tv, "tree_forward", tree_forward)
    monkeypatch.setattr(
        tv,
        "tree_commit",
        lambda lm, cache, res, path: res.fake.commit(
            lm, cache, len(path) - 1, len(path)
        ),
    )
    got = run_lane(monkeypatch, target, None, script[0], 100, 100, 0.0, 7, 6)
    assert got == script


def test_dflash_shared_prefix_leaf_cannot_hide_copy_tail():
    from yunshu_engine.copy_graft import longest_verified_path

    # Model and copied shared-prefix leaf occur in either visit order.
    assert longest_verified_path(
        [1, 2, 2, 3, 4], [-1, 0, 0, 2, 3], [2, 3, 3, 4, 5]
    ) == [0, 2, 3, 4]
    assert longest_verified_path(
        [1, 2, 3, 4, 2], [-1, 0, 1, 2, 0], [2, 3, 4, 5, 3]
    ) == [0, 1, 2, 3]
    assert longest_verified_path([1, 9], [-1, 0], [2, 3]) == [0]


def test_short_copy_branch_does_not_require_exceeding_model_node_count():
    from yunshu_engine.copy_graft import plan_tree_graft

    budget = GraftBudget()
    assert plan_tree_graft([9, 8, 7], 15, budget) == ([9, 8, 7], 3)
    source, rows = plan_tree_graft(list(range(31)), 8, budget)
    assert rows == 4 and len(source) == 12
    assert plan_tree_graft([], 15, budget) == ([], 0)
    assert plan_tree_graft(list(range(31)), 8, budget, room=10)[1] == 2


def test_graft_reference_shares_matching_ancestors_without_duplicates():
    from yunshu_engine.dflash_plan import graft_reference

    # model tree: 1 -> 2 (children of root chain), copied chain 1 2 3 4
    tokens, parents = [1, 2, 9], [-1, 0, 1, 1]
    out_t, out_p = graft_reference(tokens, parents, [1, 2, 3, 4], 2)
    assert out_t == [1, 2, 9, 3, 4]
    assert out_p[:4] == [-1, 0, 1, 1]
    # appended rows chain from the deepest shared node (row 2 = token 2)
    assert out_p[4] == 2 and out_p[5] == 4


def test_graft_reference_pads_and_diverges_at_root():
    from yunshu_engine.dflash_plan import graft_reference

    out_t, out_p = graft_reference([5], [-1, 0], [7, 8], 3)
    assert out_t == [5, 7, 8, 0]
    assert out_p == [-1, 0, 0, 2, 3]


def test_budget_stops_grafting_when_rounds_are_slower_per_token():
    budget = GraftBudget()
    for _ in range(3):
        budget.observe(0, 0, 8, 55.0, tokens=9)
    for _ in range(3):
        budget.observe(4, 8, 12, 105.0, tokens=12)
    assert budget.choose(20) == 0
    assert any(budget.choose(20) for _ in range(16))


def test_budget_keeps_grafting_when_rounds_are_faster_per_token():
    budget = GraftBudget()
    budget.observe(0, 0, 8, 55.0, tokens=9)
    budget.observe(4, 8, 12, 80.0, tokens=17)
    assert budget.choose(20) > 0
