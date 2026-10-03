"""Numerical admission and serial floating-point projections on Apple GPUs."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def test_lane_probe_rejects_wrong_row_invariant_arithmetic(monkeypatch):
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    monkeypatch.setattr(lane_qmm, "_READY", [])
    calls = []

    def wrong(x, weight, *args, **kwargs):
        calls.append(True)
        return mx.zeros((x.shape[0], weight.shape[0]), mx.bfloat16)

    monkeypatch.setattr(lane_qmm, "lane_matmul", wrong)
    assert not lane_qmm.ready()
    count = len(calls)
    assert not lane_qmm.ready()
    assert len(calls) == count  # one-time admission, including failure
    linear = nn.Linear(512, 48, bias=False)
    linear.set_dtype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    assert not lane_linear.eligible(quantized)


@pytest.mark.parametrize(
    "arch",
    ["applegpu_g13g", "applegpu_g14g", "applegpu_g15s", "applegpu_g16s", "unknown"],
)
def test_pre_nax_architectures_decline_packed_kernels(monkeypatch, arch):
    import platform

    from yunshu_engine.kernels import omlx

    monkeypatch.setattr(omlx, "_NAX_CACHE", {})
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.3", (), ""))
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    assert not omlx.is_nax_available()


def test_nax_requires_the_installed_metallib(monkeypatch):
    import platform
    from pathlib import Path

    from yunshu_engine.kernels import omlx

    monkeypatch.setattr(omlx, "_NAX_CACHE", {})
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.3", (), ""))
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g17s"})
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert not omlx.is_nax_available()


def test_floating_verify_projection_equals_serial_decode(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language, speculative_verifier
    from mlx_vlm.speculative.ops import linear as ops

    from yunshu_engine.kernels import batch_invariant

    # The install patches upstream symbols. Snapshot each to keep this test local.
    cls = speculative_verifier.Qwen3_5BatchInvariantForward
    for module in (ops, language, speculative_verifier):
        for name in (
            "_target_verify_linear",
            "_target_verify_linears",
            "_target_verify_quantized_linear",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(module, name))
    for name in ("_linear", "_linears", "quantized_linear", "quantized_argmax"):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", nn.QuantizedLinear.__call__)
    monkeypatch.setattr(batch_invariant, "_STATE", {"installed": False, "active": True})
    batch_invariant.install(SimpleNamespace(named_modules=lambda: []))
    projection = nn.Linear(512, 1024, bias=False)
    projection.set_dtype(mx.bfloat16)
    mx.random.seed(77)
    x = mx.random.normal((1, 6, 512)).astype(mx.bfloat16)
    expected = mx.concatenate([projection(x[:, t : t + 1]) for t in range(6)], axis=1)
    got = cls()._linear(projection, x)
    assert mx.array_equal(got, expected).item()
