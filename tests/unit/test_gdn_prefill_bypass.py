"""The round driver's prefill runs the step GDN kernel inside ``step_kernel``;
upstream-path prefill keeps the chunked kernel (its spans are 2048 rows)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


def test_step_kernel_bypasses_the_chunked_kernel(monkeypatch):
    from mlx_vlm.models.qwen3_5 import gated_delta as gd

    from yunshu_engine.kernels import gdn_prefill

    calls = []
    monkeypatch.setattr(gd, "gated_delta_kernel", lambda *a: calls.append("step"))
    monkeypatch.setattr(
        mx.fast, "gated_delta_update", lambda *a: calls.append("chunked"), raising=False
    )
    monkeypatch.setitem(gdn_prefill._STATE, "installed", False)
    monkeypatch.setitem(gdn_prefill._STATE, "enabled", False)
    assert gdn_prefill.install()
    q = SimpleNamespace(shape=(1, 128))
    g = SimpleNamespace(ndim=3)
    args = (q, None, None, g, None, object())
    gd.gated_delta_kernel(*args)
    with gdn_prefill.step_kernel():
        gd.gated_delta_kernel(*args)
    gd.gated_delta_kernel(*args)
    assert calls == ["chunked", "step", "chunked"]
