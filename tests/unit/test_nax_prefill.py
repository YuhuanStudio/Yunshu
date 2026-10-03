"""Fail-closed geometry/version guards for the measured prefill path."""

import pytest

pytest.importorskip("mlx.core")
from yunshu_engine.kernels import nax_prefill  # noqa: E402


@pytest.mark.parametrize(
    "m,k,n,bits,group,tiled",
    [
        (512, 5120, 17408, 4, 64, True),
        (600, 5120, 17408, 4, 64, True),
        (16384, 5120, 17408, 4, 64, True),
        (2048, 5121, 17408, 4, 64, True),
        (2048, 5120, 48, 5, 64, False),
        (2048, 5120, 248320, 4, 64, True),
        (2048, 5120, 17408, 6, 64, True),
        (2048, 5120, 17408, 4, 32, True),
        (2048, 5120, 17408, 4, 64, False),
    ],
)
def test_unmeasured_shapes_keep_stock(monkeypatch, m, k, n, bits, group, tiled):
    monkeypatch.setattr(nax_prefill, "_enabled", True)
    assert not nax_prefill.eligible(m, k, n, bits, group, tiled)


def test_measured_shapes_require_enable(monkeypatch):
    monkeypatch.setattr(nax_prefill, "_enabled", False)
    assert not nax_prefill.eligible(2048, 5120, 17408, 4, 64, True)
    monkeypatch.setattr(nax_prefill, "_enabled", True)
    assert nax_prefill.eligible(2048, 5120, 17408, 4, 64, True)
    assert nax_prefill.eligible(8192, 17408, 5120, 5, 64, True)


def test_unknown_version_disables_even_a_previous_enable(monkeypatch):
    monkeypatch.setattr(nax_prefill, "_enabled", True)
    monkeypatch.setattr(nax_prefill.importlib.metadata, "version", lambda _: "0.33.0")
    assert not nax_prefill.enable()
    assert not nax_prefill.enabled()


def test_known_device_and_abi_can_enable(monkeypatch):
    monkeypatch.setattr(nax_prefill, "_enabled", False)
    monkeypatch.setattr(nax_prefill.importlib.metadata, "version", lambda _: "0.32.3")
    monkeypatch.setattr(
        nax_prefill.mx, "device_info", lambda: {"device_name": "Apple M5 Max"}
    )
    monkeypatch.setattr(nax_prefill, "lane_header", lambda: "known ABI")
    assert nax_prefill.enable()
    assert nax_prefill.enabled()


def test_source_change_changes_arithmetic_namespace(monkeypatch):
    nax_prefill.arithmetic_id.cache_clear()
    monkeypatch.setattr(nax_prefill, "lane_header", lambda: "kernel A")
    first = nax_prefill.arithmetic_id()
    nax_prefill.arithmetic_id.cache_clear()
    monkeypatch.setattr(nax_prefill, "lane_header", lambda: "kernel B")
    assert nax_prefill.arithmetic_id() != first
    nax_prefill.arithmetic_id.cache_clear()


def test_long_narrow_dispatch_does_not_widen_general_row_limit(monkeypatch):
    import mlx.core as mx

    from yunshu_engine.kernels.tensorfold import lane_qmm

    def fake_kernel(name):
        def run(**kwargs):
            return [
                mx.zeros(kwargs["output_shapes"][0], dtype=kwargs["output_dtypes"][0])
            ]

        return run

    monkeypatch.setattr(lane_qmm, "_kernel", fake_kernel)
    monkeypatch.setattr(lane_qmm, "_xs_cache", {})
    x = mx.zeros((1024, 64), dtype=mx.bfloat16)
    weight = mx.zeros((48, 8), dtype=mx.uint32)
    sbt = mx.zeros((1, 48, 2), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="at most"):
        lane_qmm.lane_matmul(x, weight, sbt, row_block=32, row_limit=512)
    got = lane_qmm.lane_matmul(x, weight, sbt, row_block=32, prefill_narrow=True)
    assert got.shape == (1024, 48)
    with pytest.raises(ValueError, match="long narrow"):
        lane_qmm.lane_matmul(
            x,
            mx.zeros((256, 8), dtype=mx.uint32),
            sbt,
            row_block=32,
            prefill_narrow=True,
        )
