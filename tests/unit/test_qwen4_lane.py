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
