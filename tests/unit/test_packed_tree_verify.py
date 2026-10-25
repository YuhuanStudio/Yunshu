"""Packed verification shares projections, never sequence state or rollback."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
from yunshu_engine.round_driver import verify as pv  # noqa: E402


def test_solo_delegates_exact_lane(monkeypatch):
    marker = object()
    calls = []
    monkeypatch.setattr(
        pv.tv, "tree_forward", lambda *a, **k: (calls.append((a, k)), marker)[1]
    )
    w = pv.Window(np.ones((1, 2)), SimpleNamespace(width=2), [])
    assert pv.forward("lm", [w], (1,), verifier="v") == [marker]
    assert calls == [(("lm", w.tokens, w.shape, w.cache, (1,)), {"verifier": "v"})]


@pytest.mark.parametrize("bad", ["width", "budget", "alias"])
def test_rejects_invalid_pack_before_forward(bad):
    w = pv.Window(np.ones((1, 2)), SimpleNamespace(width=2), [])
    if bad == "width":
        w.tokens = np.ones((1, 1))
        windows = [w]
    elif bad == "budget":
        w.shape.width = 33
        windows = [w]
    else:
        windows = [w, w]
    with pytest.raises(ValueError):
        pv.forward(None, windows)


@pytest.mark.parametrize("fail", [None, "attention", "submit"])
def test_projection_once_per_layer_and_transactional_abort(monkeypatch, fail):
    tv = pv.tv
    calls, cleared = [], []
    mx = SimpleNamespace(
        concatenate=np.concatenate,
        broadcast_to=np.broadcast_to,
        sigmoid=lambda x: 1 / (1 + np.exp(-x)),
    )

    def submit(*_):
        if fail == "submit":
            raise RuntimeError("submit")

    mx.async_eval = submit
    monkeypatch.setattr(pv, "mx", mx)
    monkeypatch.setattr(tv, "lane_projections", lambda _: True)
    monkeypatch.setattr(tv, "_rope_delta", lambda _: 0)
    monkeypatch.setattr(tv.ragged_kv, "_lane_length", lambda c: c.length)
    monkeypatch.setattr(tv, "RoundContext", lambda *a: a)
    monkeypatch.setattr(tv.vq, "clear_group_sums", lambda: cleared.append(True))

    class Cache:
        def __init__(self, n):
            self.length = n

        def trim(self, n):
            self.length -= n

    def prepare(q, k, v, c, pos, *_):
        c.length += q.shape[1]
        return q[:, :, None].transpose(0, 2, 1, 3), k, v, np.zeros_like(q), None

    attn = SimpleNamespace(
        q_proj="q",
        k_proj="k",
        v_proj="v",
        o_proj="o",
        scale=1,
        _prepare_projected_qkv=prepare,
    )

    def attention(q, c, *_):
        if fail == "attention" and c.length > 10:
            raise RuntimeError("attention")
        return q

    monkeypatch.setattr(tv, "tree_attention", attention)
    layer = SimpleNamespace(
        is_linear=False,
        self_attn=attn,
        input_layernorm=lambda x: x,
        post_attention_layernorm=lambda x: x,
        mlp="mlp",
    )
    model = SimpleNamespace(
        fa_idx=0,
        layers=[layer, layer],
        embed_tokens=lambda t: t[..., None].astype(float),
        norm=lambda x: x,
    )

    def linears(members, x):
        calls.append((members, x.shape[1]))
        return (x, x, x)

    verifier = SimpleNamespace(
        _linears=linears, _linear=lambda _, x: x, _feed_forward=lambda _, x: 0.25 * x
    )
    windows = []
    for n, width in [(3, 2), (11, 3)]:
        shape = SimpleNamespace(width=width, depth_array=lambda w=width: np.arange(w))
        windows.append(
            pv.Window(np.arange(1, width + 1)[None], shape, [Cache(n), Cache(n)])
        )
    if fail:
        with pytest.raises(RuntimeError, match=fail):
            pv.forward(SimpleNamespace(model=model), windows, verifier=verifier)
        assert [[c.length for c in w.cache] for w in windows] == [[3, 3], [11, 11]]
    else:
        results = pv.forward(
            SimpleNamespace(model=model), windows, (0,), verifier=verifier
        )
        assert calls == [(("q", "k", "v"), 5)] * 2
        for w, res in zip(windows, results, strict=True):
            np.testing.assert_array_equal(
                res.hidden, w.tokens[..., None] * (1.5 * 1.25) ** 2
            )
            np.testing.assert_array_equal(
                res.captured[0], w.tokens[..., None] * (1.5 * 1.25)
            )
            assert res.records == {0: ("kv",), 1: ("kv",)}
    assert cleared == [True]


def test_probe_plan_is_cpu_checkable():
    from scripts.research.phaseb_verify import parser, plan

    a = parser().parse_args(["--tiny", "--output", "unused", "--rows", "1", "2", "4"])
    assert [c["rows"] for c in plan(a)] == [1, 2, 4]
    a.nodes = 9
    with pytest.raises(ValueError, match="32"):
        plan(a)


def test_fair_nodes_reserves_roots_caps_and_rotates_remainder():
    from yunshu_engine.round_driver.allocate import fair_nodes

    assert fair_nodes([15]) == [15]
    assert fair_nodes([15, 15]) == [15, 15]
    assert fair_nodes([15] * 4) == [7] * 4
    assert fair_nodes([0, 1, 15, 15], budget=12) == [0, 1, 4, 3]
    assert fair_nodes([15] * 3, budget=8, turn=0) == [2, 2, 1]
    assert fair_nodes([15] * 3, budget=8, turn=1) == [1, 2, 2]
    assert fair_nodes([], budget=0) == []
    with pytest.raises(ValueError):
        fair_nodes([0, 0], budget=1)


def test_probe_tiny_config_satisfies_installed_model_schema():
    from scripts.research.phaseb_verify import tiny_config

    cfg = tiny_config()
    assert cfg.model_type == "qwen3_5"
    assert cfg.linear_key_head_dim == cfg.linear_value_head_dim == 128
    assert cfg.head_dim == 256


def test_probe_tiny_end_to_end_small_arrays():
    from scripts.research.phaseb_verify import parser, run

    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not tile_ready():
        pytest.skip("needs lane-capable Metal")
    a = parser().parse_args(["--tiny", "--context", "32", "--output", "unused"])
    rows = []
    run(a, rows.append)
    assert rows[-1]["complete"] and rows[-1]["success"]
    assert len(rows) == 8
