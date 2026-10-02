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
@pytest.mark.parametrize("rows", [0, 4, 8, 16, 24, 32])
@pytest.mark.parametrize("cost_aware", [False, True])
def test_lane_output_is_identical_with_copy_rounds(
    toy,  # noqa: F811
    monkeypatch,
    accuracy,
    rows,
    cost_aware,
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
            if cost_aware:
                from yunshu_engine.copy_cost import CopyCosts

                k["costs"] = CopyCosts({8: 42, 16: 44, 24: 58, 32: 62}, 43)
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
        assert cd.rounds > 0
        if not cost_aware:
            assert cd.committed > 60  # copy rounds carried the quoted runs
        assert cd.accepted <= cd.proposed
    else:
        assert not heads


def test_copy_rows_follow_verify_width():
    from yunshu_engine import mtp_lane

    keep = mtp_lane._STATE["copy_rows"]
    try:
        assert mtp_lane.set_copy_rows(16, 8) == 8  # sg8 / packed verify: 8 rows
        assert mtp_lane.set_copy_rows(16, 32) == 16
        assert mtp_lane.set_copy_rows(64, 32) == 32
        assert mtp_lane.set_copy_rows(0, 32) == 0
        assert mtp_lane.set_copy_rows(-1, 32) == 0
        assert mtp_lane.set_copy_rows(16, -1) == 0
    finally:
        mtp_lane.set_copy_rows(keep)


@pytest.mark.parametrize(
    "lane,tile,expected",
    [(False, False, 8), (False, True, 8), (True, False, 8), (True, True, 32)],
)
def test_copy_width_requires_both_lane_projections_and_tile_attention(
    monkeypatch, lane, tile, expected
):
    from yunshu_engine.kernels import ragged_attention

    monkeypatch.setattr(ragged_attention, "tile_ready", lambda: tile)
    assert mtp_lane.verify_max_rows(lane) == expected


@pytest.mark.parametrize("dim,group,expected", [(256, 8, 32), (128, 8, 8), (256, 9, 8)])
def test_copy_width_respects_the_model_attention_tile_shape(
    monkeypatch, dim, group, expected
):
    from types import SimpleNamespace

    from yunshu_engine.kernels import ragged_attention

    monkeypatch.setattr(ragged_attention, "tile_ready", lambda: True)
    attention = SimpleNamespace(
        head_dim=dim, num_attention_heads=group * 4, num_key_value_heads=4
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(is_linear=False, self_attn=attention)]
        )
    )
    assert mtp_lane.verify_max_rows(True, model) == expected


def test_copy_width_does_not_certify_unconverted_expert_projections(monkeypatch):
    from types import SimpleNamespace

    from yunshu_engine.kernels import ragged_attention

    monkeypatch.setattr(ragged_attention, "tile_ready", lambda: True)
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_linear=True, mlp=SimpleNamespace(switch_mlp=object())
                )
            ]
        )
    )
    assert mtp_lane.verify_max_rows(True, model) == 8


def test_round_rechecks_target_after_another_engine_sets_global_copy_width(monkeypatch):
    from types import SimpleNamespace

    from yunshu_engine.kernels import lane_linear, ragged_attention

    class Lane:
        pass

    monkeypatch.setattr(lane_linear, "LaneLinear", Lane)
    monkeypatch.setattr(ragged_attention, "tile_ready", lambda: True)
    monkeypatch.setitem(mtp_lane._STATE, "copy_rows", 32)
    attention = SimpleNamespace(
        q_proj=object(), head_dim=256, num_attention_heads=32, num_key_value_heads=4
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(is_linear=False, self_attn=attention)]
        )
    )
    assert mtp_lane.copy_rows_for_model(model) == 8
    attention.q_proj = Lane()
    assert mtp_lane.copy_rows_for_model(model) == 32


def test_unknown_target_geometry_keeps_a_conservative_copy_cap(monkeypatch):
    monkeypatch.setitem(mtp_lane._STATE, "copy_rows", 32)
    assert mtp_lane.copy_rows_for_model(object()) == 8
