"""Adaptive MTP depth controller picks the block with the best tokens/second."""

from yunshu_engine.mtp_depth import DepthController


def _drive(ctl, accept_for_block, cost_for_block, rounds=200):
    t, lens, picks = 0.0, [], []
    for _ in range(rounds):
        ctl.observe(lens, t)
        block = ctl.choose(8)
        ctl.last_block, ctl.last_time = block, t
        picks.append(block)
        lens.append(accept_for_block(block))
        t += cost_for_block(block)
    return picks


def test_prefers_deep_blocks_when_drafts_are_accepted():
    ctl = DepthController(min_block=2, max_block=6, start=3)
    # every draft accepted; verify cost grows slowly with block size
    picks = _drive(ctl, lambda b: b - 1, lambda b: 0.040 + 0.003 * b)
    assert max(set(picks[-50:]), key=picks[-50:].count) == 6


def test_prefers_shallow_blocks_when_later_drafts_fail():
    ctl = DepthController(min_block=2, max_block=6, start=5)
    # only the first draft is ever accepted; deeper blocks just cost more
    picks = _drive(ctl, lambda b: min(1, b - 1), lambda b: 0.040 + 0.010 * b)
    assert max(set(picks[-50:]), key=picks[-50:].count) == 2


def test_resets_acceptance_on_new_request():
    ctl = DepthController(max_block=6, start=4)
    ctl.observe([], 0.0)
    ctl.last_block, ctl.last_time = 4, 0.0
    ctl.observe([0], 0.05)
    assert ctl.p[1] < 0.8 and ctl.rounds == 1
    ctl.observe([], 1.0)  # drafter.reset() swapped in a fresh list
    assert ctl.rounds == 0 and ctl.p[1] == 0.8
    assert 4 in ctl.cost  # hardware cost survives the reset
