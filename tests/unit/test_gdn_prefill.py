"""The chunked GatedDeltaNet core serves prefill chunks only, only when enabled, and names itself
in the APC key."""

import pytest

mx = pytest.importorskip("mlx.core")
gd = pytest.importorskip("mlx_vlm.models.qwen3_5.gated_delta")

from yunshu_engine.kernels import gdn_prefill  # noqa: E402


@pytest.fixture
def patched(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mx.fast,
        "gated_delta_update",
        lambda *a: calls.append("fast") or "fast",
        raising=False,
    )
    monkeypatch.setattr(
        gd, "gated_delta_kernel", lambda *a: calls.append("step") or "step"
    )
    monkeypatch.setitem(gdn_prefill._STATE, "installed", False)
    monkeypatch.setitem(gdn_prefill._STATE, "enabled", False)
    return calls


def test_dispatch_by_chunk_length_and_enable(patched):
    assert gdn_prefill.kernel_id() == "gdn-step"
    assert gdn_prefill.install()
    assert gdn_prefill.kernel_id() == f"gdn-chunked-ge{gdn_prefill.MIN_TOKENS}"
    kernel = gd.gated_delta_kernel
    long = mx.zeros((1, gdn_prefill.MIN_TOKENS, 1, 1))
    short = mx.zeros((1, gdn_prefill.MIN_TOKENS - 1, 1, 1))
    g3 = mx.zeros((1, 1, 1))
    state = mx.zeros((1,))
    assert kernel(long, long, long, g3, g3, state) == "fast"
    assert kernel(short, short, short, g3, g3, state) == "step"
    assert kernel(long, long, long, mx.zeros((1, 1, 1, 1)), g3, state) == "step"
    gdn_prefill.disable()
    assert gdn_prefill.kernel_id() == "gdn-step"
    assert kernel(long, long, long, g3, g3, state) == "step"


def test_prefill_scope_is_nested_and_exception_safe():
    assert not gdn_prefill.in_prefill()
    with gdn_prefill.prefill_scope():
        assert gdn_prefill.in_prefill()
        with pytest.raises(RuntimeError), gdn_prefill.prefill_scope():
            assert gdn_prefill.in_prefill()
            raise RuntimeError("cancelled prompt")
        assert gdn_prefill.in_prefill()
    assert not gdn_prefill.in_prefill()


@pytest.mark.parametrize("rows", [1, 2, 31, 63, 64, 65, 128])
def test_chunked_padding_is_an_identity_suffix(monkeypatch, rows):
    # Inspect lazy arrays on the CPU; no Metal command is needed for the mock.
    with mx.stream(mx.cpu):
        q = mx.ones((1, rows, 2, 4))
        v = mx.ones((1, rows, 4, 4))
        g = mx.full((1, rows, 4), 0.5)
        beta = mx.full(g.shape, 0.75)
        state = mx.ones((1, 4, 4, 4))
        calls = []

        def fast(qp, kp, vp, gp, bp, st, mask):
            pad = (-rows) % 64
            assert qp.shape[1] == rows + pad
            assert st is state
            if pad:
                assert mx.all(qp[:, rows:] == 0).item()
                assert mx.all(kp[:, rows:] == 0).item()
                assert mx.all(vp[:, rows:] == 0).item()
                assert mx.all(gp[:, rows:] == 1).item()
                assert mx.all(bp[:, rows:] == 0).item()
            calls.append(True)
            return vp, st

        monkeypatch.setattr(mx.fast, "gated_delta_update", fast)
        out, got = gdn_prefill.chunked_padded(q, q, v, g, beta, state)
        assert out.shape == v.shape and got is state and calls == [True]


def test_split_gdn_dispatch_is_confined_to_prompt_scope(patched, monkeypatch):
    gdn_prefill.install()
    monkeypatch.setitem(gdn_prefill._STATE, "split", True)
    calls = []
    monkeypatch.setattr(
        gdn_prefill, "chunked_padded", lambda *args: calls.append("padded") or "padded"
    )
    q = mx.zeros((1, 1, 1, 1))
    gates = mx.zeros((1, 1, 1))
    state = mx.zeros((1,))
    kernel = gd.gated_delta_kernel
    assert kernel(q, q, q, gates, gates, state) == "step"
    with gdn_prefill.prefill_scope():
        assert kernel(q, q, q, gates, gates, state) == "padded"
        with gdn_prefill.step_kernel():
            assert kernel(q, q, q, gates, gates, state) == "step"
    assert kernel(q, q, q, gates, gates, state) == "step"
    assert calls == ["padded"]


def test_singleton_prompt_keeps_prefill_forward_and_convolution(monkeypatch):
    from types import SimpleNamespace

    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    monkeypatch.setitem(gdn_prefill._STATE, "enabled", True)
    monkeypatch.setitem(gdn_prefill._STATE, "split", True)
    monkeypatch.setattr(
        Qwen3_5GatedDeltaNet, "_causal_conv1d_decode", lambda *args: "decode-conv"
    )
    lm = SimpleNamespace(_supports_batch_invariant_decode=lambda: True)
    gdn_prefill.install_split_forward(lm)
    layer = SimpleNamespace(conv1d=lambda x: "prefill-conv")
    assert lm._supports_batch_invariant_decode()
    assert Qwen3_5GatedDeltaNet._causal_conv1d_decode(layer, None) == "decode-conv"
    with gdn_prefill.prefill_scope():
        assert not lm._supports_batch_invariant_decode()
        assert Qwen3_5GatedDeltaNet._causal_conv1d_decode(layer, None) == "prefill-conv"
    assert lm._supports_batch_invariant_decode()
