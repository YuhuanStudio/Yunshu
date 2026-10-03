"""Regressions from the captured tiny smoke, evaluated on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from yunshu_engine.message_adapter import adapt_messages
from yunshu_engine.speculative_decoder import DraftResult, SpeculativeDecoder


def test_renamed_checkpoint_hoists_all_instruction_roles_without_mutation():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "reminder"},
        {"role": "developer", "content": [{"type": "text", "text": "policy"}]},
    ]
    out = adapt_messages(messages, "/tmp/tiny-q4-mtp")
    assert out == [{"role": "system", "content": "reminder\n\npolicy"}, messages[0]]
    assert messages[2]["role"] == "developer"


def test_external_verification_keeps_unprocessed_bonus_context():
    with mx.stream(mx.cpu):
        seen = []
        cache = SimpleNamespace(offset=3)

        def target(ids, cache):
            seen.append(cache[0].offset)
            return mx.array([[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]])

        decoder = SpeculativeDecoder(target, object(), SimpleNamespace(eos_token_id=99))
        with patch(
            "mlx_lm.models.cache.trim_prompt_cache",
            side_effect=lambda c, n: setattr(c[0], "offset", c[0].offset - n),
        ):
            decoder.verify_draft(
                DraftResult([1], [0.0]),
                mx.array([[0]]),
                [cache],
                cache_contains_last_token=False,
            )
        assert seen == [3]


def test_fp32_round_attention_preserves_dtype_and_causal_prefix():
    from yunshu_engine.round_driver.batch import KVPlan, Slots, attend

    with mx.stream(mx.cpu):
        q = mx.ones((1, 1, 2, 4), dtype=mx.float32)
        k = mx.ones((1, 1, 2, 4), dtype=mx.float32)
        v = mx.array([[[[2.0, 2.0, 2.0, 2.0], [6.0, 6.0, 6.0, 6.0]]]])
        at = SimpleNamespace(
            q_proj=lambda x: x,
            k_proj=lambda x: x,
            v_proj=lambda x: x,
            _prepare_projected_qkv=lambda *a: (q, k, v, mx.zeros((1, 2, 4)), None),
            scale=0.5,
            o_proj=lambda x: x,
        )
        slots = Slots(1)
        slot = slots.alloc()
        slots.reserve(2)
        out = attend(at, mx.zeros((1, 2, 4)), slots, 0, KVPlan.make([slot], [0], 2))
        assert out.dtype == mx.float32
        assert out.tolist() == [[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]]


def test_tree_setting_reaches_upstream_hook_instead_of_chain_lane(monkeypatch):
    from mlx_vlm.generate import ar
    from mlx_vlm.speculative import utils

    from yunshu_engine import mtp_lane, settings

    state = dict(mtp_lane._STATE)
    calls = []
    monkeypatch.setattr(
        ar,
        "run_speculative_server_rounds",
        lambda *a, **kw: calls.append("tree") or iter(()),
    )
    monkeypatch.setattr(
        utils, "run_speculative_server_rounds", ar.run_speculative_server_rounds
    )
    monkeypatch.setattr(
        settings, "get", lambda name: "tree" if name == "YUNSHU_SPEC_TREE" else None
    )
    monkeypatch.setattr(
        mtp_lane, "rounds", lambda *a, **kw: calls.append("chain") or iter(())
    )
    mtp_lane._STATE.update(installed=False, enabled=True, guide=None)
    try:
        mtp_lane.install()
        with mx.stream(mx.cpu):
            ar.run_speculative_server_rounds(
                None,
                SimpleNamespace(supports_greedy_draft_argmax=True),
                [],
                None,
                draft_kind="mtp",
                greedy_sampling=True,
                first_bonus=mx.array([1]),
                max_tokens=4,
                sampler=None,
            )
        assert calls == ["tree"]
    finally:
        mtp_lane._STATE.clear()
        mtp_lane._STATE.update(state)
