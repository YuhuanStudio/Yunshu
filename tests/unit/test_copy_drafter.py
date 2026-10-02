"""Prompt-copy drafting: the drafter's proposals, backoff and index scope, and the
speculative lane's output with copy rounds (identical to the model-draft rounds)."""

from __future__ import annotations

import random

import pytest

from yunshu_engine.copy_drafter import CopyConfig, CopyDrafter


def seg(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(10, 200) for _ in range(n)]


def test_proposes_the_continuation_of_the_longest_match():
    s = seg(1, 40)
    d = CopyDrafter(max_draft=7)
    d.extend(seg(2, 30) + s + seg(3, 30))
    d.extend(s[:12])
    assert d.draft() == s[12:16]  # a short match offers the first width
    assert d.last_match >= 12
    d.observe_copy(4, 4)  # fully verified: the next copy gets the whole window
    d.extend(s[12:16])
    assert d.draft() == s[16:23]


def test_index_covers_the_whole_prompt_not_only_the_tail():
    """A prefix-cache hit only computes the tail, but the copy source may sit in the
    skipped part: the drafter is seeded with the full prompt."""
    s = seg(4, 40)
    prompt = seg(5, 20) + s + seg(6, 5000)
    d = CopyDrafter(max_draft=7)
    d.extend(prompt)
    d.extend(s[:10])
    assert d.draft() == s[10:14]
    tail_only = CopyDrafter(max_draft=7)
    tail_only.extend(prompt[-1000:])
    tail_only.extend(s[:10])
    assert tail_only.draft() == []


def test_no_proposal_below_min_match_and_window_is_a_parameter():
    s = seg(7, 40)
    d = CopyDrafter(CopyConfig(min_match=6), max_draft=3)
    d.extend(s + seg(8, 20) + s[:4])
    assert d.draft() == []  # 4 matching tokens < min_match
    d2 = CopyDrafter(max_draft=15, config=CopyConfig(confident_match=8))
    d2.extend(s + seg(8, 20) + s[:10])
    assert d2.draft() == s[10:25]


def test_misses_back_off_and_a_hit_resets():
    s = seg(9, 60)
    d = CopyDrafter(max_draft=7)
    d._model_tpr = 1.0
    d.extend(s + seg(10, 10) + s[:12])
    assert d.draft()
    d.observe_copy(4, 0)
    assert d.draft() == []  # silenced one round
    assert d.draft()  # then probes again
    d.observe_copy(4, 0)
    assert d.draft() == [] and d.draft() == []  # two silent rounds
    assert d.draft()
    d.observe_copy(4, 4)
    d.observe_model(3)
    assert d._misses == 0


def test_low_benefit_copy_yields_to_the_model_unless_the_match_is_long():
    s = seg(11, 80)
    d = CopyDrafter(CopyConfig(min_match=6, confident_match=24), max_draft=7)
    d._copy_gain, d._model_tpr = 1.2, 3.0
    d.extend(s + seg(12, 10) + s[:10])
    assert d.draft() == []  # 10-token match, copy has been paying less than the model
    d.extend(s[10:40])
    assert d.draft()  # a 40-token match is trusted


# ---- the lane ---------------------------------------------------------------

mx = pytest.importorskip("mlx.core")
pytest.importorskip("llguidance")
pytest.importorskip("tokenizers")

from tests.unit.test_mtp_lane_tool_guide import (  # noqa: E402,F401
    ToyTarget,
    run_lane,
    toy,
)
from yunshu_engine import mtp_lane  # noqa: E402


@pytest.mark.parametrize("accuracy", [0.0, 0.7])
@pytest.mark.parametrize("rows", [0, 4, 8, 16])
def test_lane_output_is_identical_with_copy_rounds(
    toy,  # noqa: F811
    monkeypatch,
    accuracy,
    rows,
):
    hf = toy
    eos = hf.eos_token_id
    quoted = seg(21, 60)
    prompt = seg(22, 40) + quoted + seg(23, 40)
    script = seg(24, 6) + quoted[:50] + seg(25, 8) + quoted[10:55] + [eos]
    target = ToyTarget(script, len(hf) + 4, eos, seed=3)
    plain_rows = mtp_lane._STATE["copy_rows"]
    try:
        mtp_lane.set_copy_rows(0)
        mtp_lane.set_context(None)
        ref = run_lane(monkeypatch, target, None, script[0], eos, 400, accuracy, 7, 6)
        assert ref == script
        mtp_lane.set_copy_rows(rows)
        mtp_lane.set_context(prompt)
        heads: list = []
        orig = mtp_lane.CopyDrafter

        def spy(*a, **k):
            heads.append(orig(*a, **k))
            return heads[-1]

        monkeypatch.setattr(mtp_lane, "CopyDrafter", spy)
        got = run_lane(monkeypatch, target, None, script[0], eos, 400, accuracy, 7, 6)
    finally:
        mtp_lane.set_copy_rows(plain_rows)
        mtp_lane.set_context(None)
    assert got == script
    if rows >= 3:
        (cd,) = heads
        assert (
            cd.rounds > 0 and cd.committed > 60
        )  # copy rounds carried the quoted runs
        assert cd.accepted <= cd.proposed
    else:
        assert not heads
