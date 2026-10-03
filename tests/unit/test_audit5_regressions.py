"""Regressions from the captured tiny smoke, evaluated on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from yunshu_engine.message_adapter import adapt_messages
from yunshu_engine.speculative_decoder import DraftResult, SpeculativeDecoder


@pytest.mark.parametrize("model_name", ["/tmp/tiny-q4-mtp", "Qwen3.8-27B"])
def test_renamed_checkpoint_hoists_all_instruction_roles_without_mutation(model_name):
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "reminder"},
        {"role": "developer", "content": [{"type": "text", "text": "policy"}]},
    ]
    out = adapt_messages(messages, model_name)
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
        q = mx.ones((1, 1, 2, 256), dtype=mx.float32)
        k = mx.ones((1, 1, 2, 256), dtype=mx.float32)
        v = mx.broadcast_to(mx.array([2.0, 6.0])[None, None, :, None], (1, 1, 2, 256))
        at = SimpleNamespace(
            q_proj=lambda x: x,
            k_proj=lambda x: x,
            v_proj=lambda x: x,
            _prepare_projected_qkv=lambda *a: (q, k, v, mx.zeros((1, 2, 256)), None),
            scale=0.5,
            o_proj=lambda x: x,
        )
        slots = Slots(1)
        slot = slots.alloc()
        slots.reserve(2)
        out = attend(at, mx.zeros((1, 2, 256)), slots, 0, KVPlan.make([slot], [0], 2))
        assert out.dtype == mx.float32
        assert out.tolist() == [[[1.0] * 256, [2.0] * 256]]


@pytest.mark.parametrize("case", ["greedy", "guide", "keyed"])
def test_tree_setting_reaches_upstream_hook_instead_of_chain_lane(monkeypatch, case):
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
    mtp_lane._STATE.update(
        installed=False, enabled=True, guide=object() if case == "guide" else None
    )
    from yunshu_engine.keyed_sampling import KeyedSampler

    sampler = KeyedSampler(SimpleNamespace(), 42) if case == "keyed" else None
    try:
        mtp_lane.install()
        with mx.stream(mx.cpu):
            ar.run_speculative_server_rounds(
                None,
                SimpleNamespace(supports_greedy_draft_argmax=True),
                [],
                None,
                draft_kind="mtp",
                greedy_sampling=case != "keyed",
                first_bonus=mx.array([1]),
                max_tokens=4,
                sampler=sampler,
            )
        assert calls == (["tree"] if case == "greedy" else ["chain"])
    finally:
        mtp_lane._STATE.clear()
        mtp_lane._STATE.update(state)


@pytest.mark.parametrize("accepted", [False, True])
def test_generate_leaves_each_alignment_token_uncached(monkeypatch, accepted):
    from yunshu_engine.speculative_decoder import SpecDecodingConfig, VerifyResult

    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _: [])
    with mx.stream(mx.cpu):

        def model(ids, cache=None):
            return mx.array([[[0.0, 10.0, 0.0]]])

        decoder = SpeculativeDecoder(
            model,
            model,
            SimpleNamespace(eos_token_id=99),
            SpecDecodingConfig(draft_length=1, draft_temperature=0.0),
        )
        seen = []

        def verify(draft, ids, cache, **kwargs):
            seen.append(kwargs["cache_contains_last_token"])
            return VerifyResult(
                int(accepted), [1] if accepted else [], -1 if accepted else 0, 1, []
            )

        monkeypatch.setattr(decoder, "verify_draft", verify)
        assert (
            decoder.generate(mx.array([[0]]), max_tokens=5, temperature=0.0) == [1] * 5
        )
        assert seen[0] is False
        assert len(seen) >= 2 and not any(seen)


@pytest.mark.parametrize("draft_bias", [0, 3, 8])
def test_stateful_external_generation_matches_serial_target(draft_bias, monkeypatch):
    from yunshu_engine.speculative_decoder import SpecDecodingConfig

    class HistoryCache:
        def __init__(self):
            self.history = []

        @property
        def offset(self):
            return len(self.history)

        @offset.setter
        def offset(self, value):
            self.history = self.history[:value]

        def is_trimmable(self):
            return True

        def trim(self, n):
            used = min(n, len(self.history))
            self.history = self.history[: len(self.history) - used]
            return used

    def model(bias):
        def forward(ids, cache):
            rows = []
            for token in ids[0].tolist():
                cache[0].history.append(token)
                wanted = (sum(cache[0].history[-3:]) + 7 + bias) % 31
                rows.append([20.0 if i == wanted else 0.0 for i in range(31)])
            return mx.array([rows])

        return forward

    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache", lambda _: [HistoryCache()]
    )
    history = [1, 4, 9, 2]
    expected = []
    for _ in range(32):
        token = (sum(history[-3:]) + 7) % 31
        expected.append(token)
        history.append(token)
    with mx.stream(mx.cpu):
        decoder = SpeculativeDecoder(
            model(0),
            model(draft_bias),
            SimpleNamespace(eos_token_id=99),
            SpecDecodingConfig(draft_length=3, draft_temperature=0.0),
        )
        assert (
            decoder.generate(mx.array([[1, 4, 9, 2]]), max_tokens=32, temperature=0.0)
            == expected
        )

        if draft_bias == 0:
            # An identical deterministic model must propose the target's entire
            # stream. Parity alone hides a broken, perpetually rejected proposer.
            assert decoder.acceptance_rate == 1.0


def test_bf16_verification_matches_upstream_sampler_distribution():
    from mlx_lm.sample_utils import make_sampler

    with mx.stream(mx.cpu):
        logits = mx.zeros((1, 2, 8192), dtype=mx.bfloat16)
        logits[:, :, 0] = 0.1
        logits[:, :, 1] = 0.101
        lp = logits[0, :1, :] - mx.logsumexp(logits[0, :1, :], keepdims=True)
        expected = int(make_sampler(temp=0.0)(lp).item())
        assert expected == 0 and int(mx.argmax(logits[0, 0]).item()) == 1
        decoder = SpeculativeDecoder(
            lambda ids, cache=None: logits, object(), SimpleNamespace(eos_token_id=99)
        )
        result = decoder.verify_draft(
            DraftResult([expected], [0.0]),
            mx.array([[3]]),
            [],
            cache_contains_last_token=False,
        )
        assert result.accepted_ids == [expected]
        assert result.bonus_token_id == expected


def test_bf16_external_stream_matches_upstream_sampler_on_normalization_ties(
    monkeypatch,
):
    from mlx_lm.sample_utils import make_sampler

    from yunshu_engine.speculative_decoder import SpecDecodingConfig

    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _: [])
    with mx.stream(mx.cpu):
        scores = mx.zeros((1, 1, 8192), dtype=mx.bfloat16)
        scores[:, :, 0] = 0.1
        scores[:, :, 1] = 0.101

        def model(ids, cache=None):
            return mx.broadcast_to(scores, (1, ids.shape[1], 8192))

        lp = scores[:, 0, :] - mx.logsumexp(scores[:, 0, :], keepdims=True)
        expected = int(make_sampler(temp=0.0)(lp).item())
        decoder = SpeculativeDecoder(
            model,
            model,
            SimpleNamespace(eos_token_id=99),
            SpecDecodingConfig(draft_length=3, draft_temperature=0.0),
        )
        assert (
            decoder.generate(mx.array([[3, 4]]), max_tokens=8, temperature=0.0)
            == [expected] * 8
        )
        assert decoder.acceptance_rate == 1.0
