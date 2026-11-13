"""Qwen3.5-MoE (qwen3_5_moe) on the packed round-driver layer: the structure gate and the per-segment
expert call. The arithmetic gates (solo == concurrent, spec on == off, MTP / DFlash drafting on a real
MoE checkpoint) need the GPU and are listed in the parity audit (G4)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from yunshu_engine.round_driver import forward as fwd  # noqa: E402


def _lm(mlps):
    inner = q35.Qwen3_5Model.__new__(q35.Qwen3_5Model)
    inner.layers = [SimpleNamespace(mlp=m) for m in mlps]
    return SimpleNamespace(model=inner)


def _dense():
    return SimpleNamespace(gate_proj=1, up_proj=2, down_proj=3)


def _moe():
    return SimpleNamespace(
        gate=1, switch_mlp=2, shared_expert=3, shared_expert_gate=4, top_k=8
    )


def test_supports_takes_dense_moe_and_mixed_decoders_but_not_unknown_blocks():
    assert fwd.supports(_lm([_dense(), _dense()]))
    assert fwd.supports(_lm([_moe(), _moe()]))
    assert fwd.supports(_lm([_dense(), _moe()]))
    assert not fwd.supports(_lm([_dense(), SimpleNamespace(gate=1)]))
    assert not fwd.supports(SimpleNamespace(model=object()))


class _Moe(nn.Module):
    """A MoE-shaped block: what matters here is that it is called once per segment."""

    def __init__(self):
        super().__init__()
        self.switch_mlp = nn.Linear(4, 4, bias=False)
        self.calls = []

    def __call__(self, x):
        self.calls.append(tuple(x.shape))
        return self.switch_mlp(x) * 2.0


def test_moe_mlp_runs_each_segment_on_its_own_rows():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mlp = _Moe()
        x = mx.arange(1 * 9 * 4, dtype=mx.float32).reshape(1, 9, 4) / 10
        spans = [(0, 3), (3, 4), (4, 9)]
        out = fwd._mlp(mlp, x, spans, [False, True, False])
        assert mlp.calls == [(1, 3, 4), (1, 1, 4), (1, 5, 4)]
        want = mx.concatenate(
            [mlp.switch_mlp(x[:, s:e]) * 2.0 for s, e in spans], axis=1
        )
        assert mx.array_equal(out, want).item()
        assert tuple(out.shape) == (1, 9, 4)
    finally:
        mx.set_default_device(previous)


def test_packed_projection_gate_admits_moe_only_with_engaged_mtp(monkeypatch):
    # Stock MoE stays off the packed path (measured -5% decode); an engaged MTP drafter admits it.
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    monkeypatch.setattr("yunshu_engine.kernels.omlx.is_nax_available", lambda: True)

    class Lang:
        pass

    Lang.__module__ = "mlx_vlm.models.qwen3_5_moe.language"
    assert not pl.enabled(Lang())
    assert pl.enabled(Lang(), mtp_engaged=True)
    Lang.__module__ = "mlx_vlm.models.qwen3_5.language"
    assert pl.enabled(Lang())
    Lang.__module__ = "mlx_vlm.models.gemma4.language"
    assert not pl.enabled(Lang(), mtp_engaged=True)
