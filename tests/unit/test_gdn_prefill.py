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
