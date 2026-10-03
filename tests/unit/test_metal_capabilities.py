"""Numerical admission and serial floating-point projections on Apple GPUs."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def test_lane_probe_rejects_wrong_row_invariant_arithmetic(monkeypatch):
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g19s"})
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


@pytest.mark.parametrize(
    "arch, expected",
    [
        ("applegpu_g15s", "portable"),
        ("applegpu_g17s", "m5"),
        ("applegpu_g18g", "m5"),
        ("applegpu_g19s", "portable"),
        ("unknown", "portable"),
    ],
)
def test_lane_hardware_dispatch_once(monkeypatch, arch, expected):
    from yunshu_engine.kernels.tensorfold import lane_qmm, lane_widen

    for name in ("_VARIANT", "_MAIN", "_MAIN_TILED", "_COOP"):
        monkeypatch.setattr(lane_qmm, name, getattr(lane_qmm, name))
    for name in ("NIBBLES", "BYTES"):
        monkeypatch.setattr(lane_widen, name, getattr(lane_widen, name))
    monkeypatch.setattr(lane_qmm, "_VARIANT", None)
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    assert lane_qmm._resolve_variant() == expected
    monkeypatch.setattr(
        mx, "device_info", lambda: pytest.fail("repeated hardware lookup")
    )
    assert lane_qmm._resolve_variant() == expected


@pytest.mark.parametrize("arch", ["applegpu_g15s", "applegpu_g17s", "applegpu_g18g"])
def test_known_lane_generations_skip_probe(monkeypatch, arch):
    from yunshu_engine.kernels.tensorfold import lane_qmm

    monkeypatch.setattr(lane_qmm, "_READY", [])
    monkeypatch.setattr(lane_qmm, "_resolve_variant", lambda: None)
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    monkeypatch.setattr(
        lane_qmm, "lane_matmul", lambda *a, **kw: pytest.fail("known GPU probed")
    )
    assert lane_qmm.ready()


def test_m5_fragment_sources_preserved():
    import hashlib

    from yunshu_engine.kernels.tensorfold import lane_m5, lane_widen_m5

    # Fixed main 38ac21ed sources: detect accidental edits to the proven M5 kernels.
    expected = {
        "_MAIN": "36682369f0a233bbed897a9b294381641551c30896ba1b8603875663d7886a00",
        "_COOP": "ffe5b11f7de152e7871b7e15f599d592d9e67e4ab67a37ac81faed109499a13b",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(getattr(lane_m5, name).encode()).hexdigest() == digest
    for name, digest in {
        "NIBBLES": "247e4fe348eddb9689d852b09d472d3529316c34cd00c3ebc7a1ec1cb29598c6",
        "BYTES": "18c899d98329488444ff3714939b109b85732a66ba1f98adc370680010722951",
    }.items():
        assert (
            hashlib.sha256(getattr(lane_widen_m5, name).encode()).hexdigest() == digest
        )
