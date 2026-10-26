"""Flash-Next dense projections as lane matmuls: when ``configure_lane`` turns them on."""

import pytest

from yunshu_engine import qwen4_mtp, settings


class FakeLM:
    def __init__(self, head_ok):
        self.head_ok = head_ok

    def speculative_verify_hidden(self, *a):
        return None

    def speculative_draft_hidden(self, *a):
        return None

    def _supports_batch_invariant_decode(self):
        return self.head_ok


@pytest.fixture(autouse=True)
def reset():
    yield
    settings.clear_overrides()


def test_default_keeps_upstream_arithmetic(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qwen4_mtp,
        "lane_projections",
        lambda lm, head: calls.append(head) or {"lane_linear": 3},
    )
    settings.set_override("YUNSHU_QWEN4_LANE", False)
    kernels, block = qwen4_mtp.configure_lane(FakeLM(True), object(), None)
    assert kernels == {"qwen4_native": True} and block == 2 and calls == []


def test_flag_converts_dense_and_keeps_the_upstream_head(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qwen4_mtp,
        "lane_projections",
        lambda lm, head: calls.append(head) or {"lane_linear": 3},
    )
    settings.set_override("YUNSHU_QWEN4_LANE", True)
    kernels, _ = qwen4_mtp.configure_lane(FakeLM(True), None, 4)
    assert kernels["lane_linear"] == 3 and calls == [False]


def test_unreadable_head_turns_the_lane_on_with_the_head(monkeypatch):
    class Lane(FakeLM):
        converted = False

        def _supports_batch_invariant_decode(self):
            return self.converted

    lm = Lane(False)

    def convert(lm_, head):
        lm_.converted = (
            True  # a LaneLinear head is not a QuantizedLinear: upstream accepts it
        )
        return {"lane_linear": 5, "head": head}

    monkeypatch.setattr(qwen4_mtp, "lane_projections", convert)
    settings.set_override("YUNSHU_QWEN4_LANE", False)
    kernels, _ = qwen4_mtp.configure_lane(lm, None, None)
    assert kernels["lane_linear"] == 5 and kernels["head"] is True


def test_refuses_when_the_lane_cannot_take_an_unsupported_head(monkeypatch):
    monkeypatch.setattr(qwen4_mtp, "lane_projections", lambda lm, head: {})
    settings.set_override("YUNSHU_QWEN4_LANE", False)
    with pytest.raises(RuntimeError, match="batch-invariant plain decode"):
        qwen4_mtp.configure_lane(FakeLM(False), None, None)


def test_dense_skip_keeps_fused_kernel_owned_modules_quantized():
    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Module() for _ in range(1)]
            self.layers[0].mlp = nn.Module()
            self.layers[0].mlp.shared_expert = nn.Module()
            self.layers[0].mlp.shared_expert.gate_proj = nn.Linear(128, 64, bias=False)
            self.layers[0].attn_hyper_connection = nn.Module()
            self.layers[0].attn_hyper_connection.input_mix_weight_down = nn.Linear(
                128, 64, bias=False
            )
            self.layers[0].linear_attn = nn.Module()
            self.layers[0].linear_attn.in_proj_qkv = nn.Linear(128, 64, bias=False)

    m = M()
    m.set_dtype(nn.Linear(1, 1).weight.dtype)
    import mlx.core as mx

    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=64, bits=8)
    out = lane_linear.convert(m, skip=qwen4_mtp.DENSE_SKIP)
    assert out["converted"] == 1
    assert isinstance(m.layers[0].linear_attn.in_proj_qkv, lane_linear.LaneLinear)
    assert isinstance(m.layers[0].mlp.shared_expert.gate_proj, nn.QuantizedLinear)
    assert isinstance(
        m.layers[0].attn_hyper_connection.input_mix_weight_down, nn.QuantizedLinear
    )
