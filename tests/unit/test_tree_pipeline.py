"""Tree verification submits earlier layers without changing captured states."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from yunshu_engine import tree_verify as tv  # noqa: E402


def test_tree_forward_submits_live_states_before_later_layers(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35

    submitted = []

    def identity(x):
        return x

    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                is_linear=True,
                linear_attn=None,
                input_layernorm=identity,
                post_attention_layernorm=identity,
                mlp=None,
            )
            for _ in range(9)
        ],
        fa_idx=0,
        embed_tokens=lambda _: np.ones((1, 1, 1)),
        norm=identity,
    )
    monkeypatch.setattr(
        tv,
        "mx",
        SimpleNamespace(
            array=lambda value, **_: np.array(value),
            int32=np.int32,
            async_eval=lambda *values: submitted.append(
                tuple(value.copy() for value in values)
            ),
        ),
    )
    monkeypatch.setattr(tv, "RoundContext", lambda *_: None)
    monkeypatch.setattr(tv, "_rope_delta", lambda _: 0)
    monkeypatch.setattr(tv.ragged_kv, "_lane_length", lambda _: 0)
    monkeypatch.setattr(
        tv, "_gdn_layer", lambda _v, _l, x, _c, _s: (x * 0.125, ("test",))
    )
    monkeypatch.setattr(tv.vq, "_add_rms_eligible", lambda *_: False)
    monkeypatch.setattr(
        q35._EXACT_SPECULATIVE_VERIFIER,
        "_feed_forward",
        lambda _mlp, x: x * 0.25,
    )
    result = tv.tree_forward(
        SimpleNamespace(model=model),
        np.ones((1, 1)),
        tv.TreeShape([-1]),
        [None] * 9,
        [0, 3, 8],
    )
    factor = 1.125 * 1.25
    assert np.all(result.hidden == factor**9)
    assert len(submitted) == 3
    for (hidden, normed), completed in zip(submitted, (1, 4, 8), strict=True):
        assert np.all(hidden == factor**completed)
        assert np.array_equal(hidden, normed)
    for captured, completed in zip(result.captured, (1, 4, 9), strict=True):
        assert np.all(captured == factor**completed)


def test_compact_tail_keeps_live_tile_addresses_at_chunk_boundaries():
    # The tile kernel clamps its final load to capacity - 64. Shrinking the
    # backing allocation must not shift any tile that contributes to softmax.
    for live in range(1, tv.CK + tv.MAX_ROWS + 1):
        compact = tv._tree_tail_capacity(live)
        padded = ((live + tv.CK - 1) // tv.CK) * tv.CK
        assert live <= compact <= padded
        for start in range(0, live, 64):
            assert min(start, compact - 64) == min(start, padded - 64)
    assert tv._tree_tail_capacity(1) == 64
    assert tv._tree_tail_capacity(tv.CK + 1) == tv.CK + 64


def test_early_submission_failure_rolls_back_appended_kv(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35

    class Cache:
        def __init__(self):
            self.length = 0
            self.trimmed = []

        def trim(self, count):
            self.length -= count
            self.trimmed.append(count)

    cache = [Cache() for _ in range(4)]

    def identity(x):
        return x

    def prepare(q, _k, _v, item, *_):
        item.length += 1
        return q.reshape(1, 1, 1, 1), None, None, q, None

    layers = [
        SimpleNamespace(
            is_linear=True,
            linear_attn=None,
            input_layernorm=identity,
            post_attention_layernorm=identity,
            mlp=None,
        )
        for _ in range(4)
    ]
    layers[-1].is_linear = False
    layers[-1].self_attn = SimpleNamespace(
        q_proj=None,
        k_proj=None,
        v_proj=None,
        o_proj=None,
        scale=1,
        _prepare_projected_qkv=prepare,
    )
    model = SimpleNamespace(
        layers=layers,
        fa_idx=3,
        embed_tokens=lambda _: np.ones((1, 1, 1)),
        norm=identity,
    )
    calls = []

    def submit(*_):
        calls.append(None)
        # Use five layers below so layer four has an early submission.
        if len(calls) == 2:
            raise RuntimeError("submission failed")

    model.layers.append(model.layers[0])
    cache.append(Cache())
    monkeypatch.setattr(
        tv,
        "mx",
        SimpleNamespace(
            array=lambda value, **_: np.array(value),
            int32=np.int32,
            sigmoid=np.ones_like,
            async_eval=submit,
        ),
    )
    monkeypatch.setattr(tv, "RoundContext", lambda *_: None)
    monkeypatch.setattr(tv, "_rope_delta", lambda _: 0)
    monkeypatch.setattr(tv.ragged_kv, "_lane_length", lambda _: 0)
    monkeypatch.setattr(
        tv, "_gdn_layer", lambda _v, _l, x, _c, _s: (x * 0.125, ("test",))
    )
    monkeypatch.setattr(tv, "tree_attention", lambda q, *_: q)
    monkeypatch.setattr(tv.vq, "_add_rms_eligible", lambda *_: False)
    verifier = q35._EXACT_SPECULATIVE_VERIFIER
    monkeypatch.setattr(verifier, "_linears", lambda _p, x: (x, x, x))
    monkeypatch.setattr(verifier, "_linear", lambda _p, x: x)
    monkeypatch.setattr(verifier, "_feed_forward", lambda _p, x: x * 0.25)
    with pytest.raises(RuntimeError, match="submission failed"):
        tv.tree_forward(
            SimpleNamespace(model=model),
            np.ones((1, 1)),
            tv.TreeShape([-1]),
            cache,
        )
    assert cache[3].length == 0
    assert cache[3].trimmed == [1]
