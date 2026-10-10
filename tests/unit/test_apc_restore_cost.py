from yunshu_engine.apc_restore_cost import MediaRestoreCost


def test_short_suffix_loses_but_revisit_and_long_prefix_win():
    cost = MediaRestoreCost()
    assert cost.worth(10_000_000, 96, 31)  # no device observations yet
    cost.observe(108, 0, 0.06)
    cost.observe(108, 107, 0.015)
    assert not cost.worth(10_000_000, 96, 31)
    assert cost.worth(10_000_000, 107, 1)
    assert cost.worth(3 << 30, 32716, 39)
    assert not cost.worth(1000 << 30, 32716, 39)


def test_invalid_observations_and_long_launch_bound():
    cost = MediaRestoreCost()
    for tokens, seconds in [(0, 1), (10, 0), (10, -1), (10, float("nan"))]:
        cost.observe(tokens, 0, seconds)
    assert cost.token_s == cost.cold_s == 0
    cost.observe(108, 0, 0.06)
    cost.observe(32768, 0, 12)
    assert cost.cold_s == 0.06
    assert cost.worth(3 << 30, 32716, 39)
